from .. import _monkey as monkey

from .._fusion_common import TRITON_MAX_RBLOCK

from typing import Any, Sequence, Iterable, Callable

import collections
import contextlib
import dataclasses
import itertools
import functools
import operator

import torch
import torch._inductor.config

from torch._inductor import ir, scheduler
from torch._inductor.virtualized import V
from torch._inductor.loop_body import LoopBody
from torch._inductor.scheduler import SchedulerNode, FusedSchedulerNode, BaseSchedulerNode
from torch._inductor.codegen.common import Kernel, InplacedBuffer
from torch._inductor.codegen.triton import TritonScheduling, TritonKernel, Placeholder
from torch._inductor.codegen.simd import IterationRangesRoot, EnableReduction, DisableReduction, CantSplit
from torch._inductor.utils import sympy_index_symbol, sympy_product, sympy_subs, IndentedBuffer
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv

import sympy

from triton.language import TRITON_MAX_TENSOR_NUMEL

# add 'fusion' to comma-separated TORCH_LOG to enable
fusion_log = torch._logging.getArtifactLogger('torch._inductor', "fusion")  # fusion_log.debug(...)
schedule_log = torch._logging.getArtifactLogger('torch._inductor.codegen.simd', "schedule")

aten = torch.ops.aten


def override_bmm_decomposition() -> None:
    f"""Adapted from torch._decomp.remove_decompositions.
    Override torch._decomp.decompositions::pw_cast_for_opmath decorated torch._inductor.decomposition::bmm,
    which forces input type promotion (bf16 -> fp32). Prefer bf16 matmul for efficiency.
    """
    from torch.utils import _pytree as pytree
    from torch.utils._pytree import tree_map
    def pw_cast_for_bf16_bmm(
        f: Callable,
        pw_cast_for_opmath_bmm: Callable,
    ):
        """Adapted from torch._decomp.decompositions"""
        @functools.wraps(f)
        def inner(*args, **kwargs):
            if all(x.dtype != torch.bfloat16 for x in pytree.arg_tree_leaves(*args, **kwargs) if isinstance(x, torch.Tensor)):
                return pw_cast_for_opmath_bmm(*args, **kwargs)

            def to_bf16_prec(x):
                if isinstance(x, torch.Tensor):
                    return x.to(torch.bfloat16)
                else:
                    return x

            r = f(*tree_map(to_bf16_prec, args), **tree_map(to_bf16_prec, kwargs))
            return r

        return inner


    from torch._inductor.decomposition import decompositions
    assert isinstance(aten.bmm, torch._ops.OpOverloadPacket)
    for overload_name in aten.bmm.overloads():
        opo = getattr(aten.bmm, overload_name)
        old_decomp = decompositions.pop(opo)
        decompositions[opo] = pw_cast_for_bf16_bmm(old_decomp.__wrapped__, old_decomp)


def override_meta_bmm():
    """
    Override torch._meta_registrations::meta_bmm, which uses metadata of the second input for output.
    Output of bf16 @ bf16 will be fp32.
    """
    from torch._meta_registrations import common_meta_baddbmm_bmm, activate_meta
    def meta_bmm(batch1: torch.Tensor, batch2: torch.Tensor):
        """Adapted from torch._meta_registrations::common_meta_baddbmm_bmm"""
        if batch1.dtype != torch.bfloat16 and batch2.dtype != torch.bfloat16:
            return common_meta_baddbmm_bmm(batch1, batch2, True)
        torch._check(batch1.dim() == 3, lambda: "batch1 must be a 3D tensor")
        torch._check(batch2.dim() == 3, lambda: "batch2 must be a 3D tensor")

        batch1_sizes = batch1.size()
        batch2_sizes = batch2.size()

        bs = batch1_sizes[0]
        contraction_size = batch1_sizes[2]
        res_rows = batch1_sizes[1]
        res_cols = batch2_sizes[2]
        output_size = (bs, res_rows, res_cols)

        torch._check(
            batch2_sizes[0] == bs and batch2_sizes[1] == contraction_size,
            lambda: f"Expected size for first two dimensions of batch2 tensor to be: [{bs}"
            f", {contraction_size}] but got: [{batch2_sizes[0]}, {batch2_sizes[1]}].",
        )

        output = batch2.new_empty(output_size, dtype=torch.float32)

        return output

    from torch._decomp import meta_table, _convert_out_params
    fn = _convert_out_params(meta_bmm)
    meta_table[aten.bmm.default] = fn
    # torch._meta_registrations::activate_meta
    k = torch._C.DispatchKey.Meta
    aten.bmm.default.py_kernels.pop(k)
    aten.bmm.default.py_impl(k)(fn)


def override_bmm_need_realized_inputs():
    from torch._inductor.lowering import needs_realized_inputs
    for op in [aten.bmm] + [getattr(aten.bmm, ol) for ol in aten.bmm.overloads()]:
        needs_realized_inputs.remove(op)


override_bmm_decomposition()
override_meta_bmm()
override_bmm_need_realized_inputs()



@monkey.patch(scheduler.Scheduler)
def merge_loops(self) -> None:
    """
    - For fused nodes, loop fusion should not be done individually
    """
    from torch._inductor import config
    for node in self.nodes:
        if not config.loop_ordering_after_fusion:
            continue

        # Even for CPU, if we are using the halide backend, we still need
        # the merge loops steps below
        if not isinstance(node, (SchedulerNode, FusedSchedulerNode)) or (
            node.get_device().type != "cuda" and config.cpu_backend != "halide"
        ):
            continue
        if isinstance(node, FusedSchedulerNode):
            continue  # TODO @bozhiyou loop fusion for entire fused node, not individually
        for snode in node.get_nodes():
            # merge loops for the scheduler node
            if not isinstance(snode, SchedulerNode) or snode.is_template():
                continue

            snode._body = snode._body.merge_loops()
            snode._sizes = snode._body.sizes

            # merge_loops is called after loop reordering.
            # We still need retain fake dependencies since codegen the
            # estimated amount of memory access rely on them.
            snode.refresh_dependencies(normalize=True)

            # Note that for CPU backend, merging loops will change
            # snode.group. It's fine for Triton backend.
            # But if we simplify update snode.group like this:
            #   group_fn = self.get_backend(snode.node.get_device()).group_fn
            #   snode.group = (snode.node.get_device(), group_fn(snode._sizes))
            # There is still an issue due to different snode in a
            # FusedSchedulerNode having different merged loops.
            # Skip CPU backend for now.


#####
# stage 1: matching
# check range compatibility
# implemented in dependent reduction fusion


def reduction_can_fuse(node1, node2):
    _, (numel1, rnumel1) = node1.group
    _, (numel2, rnumel2) = node2.group
    if rnumel1 != 1:
        if rnumel2 != 1:
            reduction_can_fuse = (numel1 == numel2 and rnumel1 == rnumel2  # default condition
                            ) or (numel1 == numel2 * rnumel2  # reduction over previous result
                            )
        else:
            reduction_can_fuse = (numel1 * rnumel1 == numel2 and rnumel2 == 1)  # split second range
        return reduction_can_fuse
    return False


# @monkey.patch(scheduler.Scheduler)
def can_fuse(self: scheduler.Scheduler,
             node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
    if monkey.fallback(self, node1, node2):
        return True
    
    if node1 is node2:
        return False

    if node1.is_reduction() and reduction_can_fuse(node1, node2):
        fusion_log.debug(f"\033[33mfuse\033[0m {node1.get_name()} {node1.group[1]} with {node2.get_name()} {node2.group[1]}: {node1.group[1]}, {node2.group[1]}")
        return True

    return False


#####
# stage 2: scheduling
# collect iteration context and do metadata fusion


def get_numels(self: SchedulerNode):
    _, (numel, rnumel) = self.group
    one_shot_ranges = getattr(self, 'one_shot', {})
    for s in one_shot_ranges.values():
        numel //= s
    return numel, rnumel


def find_main_body_group(nodes: Sequence[SchedulerNode]):
    """
    Used to be the group of first reduction or first node if no reduction:
        _, (numel, rnumel) = max(nodes, key=lambda x: int(x.is_reduction())).group
    Now rnumel is the outter-most reduction range.

    assumption: nodes are compatible to fuse
    """
    assert nodes
    # _, (numel, rnumel) = nodes[0].group
    common_numel = min(get_numels(n)[0] for n in nodes)
    common_rnumel = min([r for x, r in [get_numels(n) for n in nodes] if x == common_numel and r != 1] or [1])
    return common_numel, common_rnumel
    numel, rnumel = get_numels(nodes[0])
    for node in nodes[1:]:
        node_numel, node_rnumel = get_numels(node)
        if node_numel == numel and node_rnumel == rnumel:
            continue  # fits both loops
        elif node_numel == numel * rnumel:
            rnumel = node_rnumel if rnumel == 1 else rnumel
            continue  # fits inner loop, may require more nesting
        elif node_numel * node_rnumel == numel:
            numel, rnumel = node_numel, (rnumel if node_rnumel == 1 else node_rnumel)
            continue  # fits outer loop, split
        else:
            raise NotImplementedError(
                f"unexpected group: ({numel}, {rnumel}) != {node.group[1]}"
            )

    return numel, rnumel


def remap_index(prior: SchedulerNode, node: SchedulerNode):
    _, (numel, rnumel) = prior.group
    _, (node_numel, node_rnumel) = node.group

    if numel == node_numel * node_rnumel:   # reduction over previous result
        assert len(node._body.sizes[-1]) or node_rnumel == 1, f"{node._body.sizes} {node.group}"
        assert node_rnumel == 1 or functools.reduce(
            operator.mul, itertools.chain.from_iterable(prior._body.sizes[:-1])
            ) % node._body.sizes[-1][0] == 0, "cannot align reductions"

        def find_var_mapping(finer_body: LoopBody, coarser_body: LoopBody):
            if len(finer_body.iter_vars) < len(coarser_body.iter_vars):
                finer_body, coarser_body = coarser_body, finer_body
            var_mapping = {}
            base_level = len(finer_body.iter_vars) - 1
            target_range = base_range = 1
            for var in reversed(list(itertools.chain.from_iterable(coarser_body.vars))):
                if base_level < 0:
                    break
                expr = sympy.Integer(0)
                target_range *= coarser_body.var_ranges[var]
                while target_range > base_range:
                    base_var = finer_body.iter_vars[base_level]
                    base_level -= 1
                    # represents current var with prior var (loop fusion)
                    expr += base_range * base_var
                    base_range *= finer_body.var_ranges[base_var]
                
                if target_range != base_range:
                    raise RuntimeError(f"incompatible ranges {finer_body.var_ranges, coarser_body.var_ranges}")
                assert target_range == base_range, f"unmatch ranges {finer_body.var_ranges} {coarser_body.var_ranges}"
                if var != expr:
                    var_mapping[var] = expr
                target_range = base_range = 1
            return var_mapping

        def merge_indexing(finer_body: LoopBody, coarser_body: LoopBody, var_remapping):
            if len(finer_body.iter_vars) < len(coarser_body.iter_vars):
                finer_body, coarser_body = coarser_body, finer_body
            index_remapping = {}
            indexing_exprs = {n: e for n, e in finer_body.indexing_exprs.items() if  # filter reduction only indexing
                            any(s in e.free_symbols for s in finer_body.iter_vars)}
            for name, ind in coarser_body.indexing_exprs.items():
                mapped_ind = sympy_subs(ind, var_remapping)
                for known_name, known_ind in indexing_exprs.items():
                    if mapped_ind == known_ind and name != known_name:
                        index_remapping[name] = known_name
                        break
                # if name not in index_remapping:  # indexing not in prior node
                #     new_name = f"index{len(indexing_exprs)}"
                #     if name != new_name:
                #         index_remapping[name] = new_name
                #     indexing_exprs[new_name] = mapped_ind
            # append prior leftover exprs to prevent merging
            for known_name, known_ind in indexing_exprs.items():
                if known_name not in index_remapping.values():
                    new_name = f"index{len(coarser_body.indexing_exprs)}"
                    assert new_name not in coarser_body.indexing_exprs, f"non-contiguous keys {coarser_body.indexing_exprs}"
                    coarser_body.indexing_exprs[new_name] = known_ind

        var_remapping = find_var_mapping(prior._body, node._body)
        merge_indexing(prior._body, node._body, var_remapping)

    # node._sizes = anc._sizes
    # node._body.iter_vars = anc._body.iter_vars
    # node._body.reduce_vars = anc._body.reduce_vars
    # node._body.var_ranges = anc._body.var_ranges
    # node._body.sizes = anc._body.sizes
    # node._body.indexing_exprs = indexing_exprs

@monkey.patch(FusedSchedulerNode)
def __init__(self:FusedSchedulerNode, scheduler: scheduler.Scheduler, snodes: list[BaseSchedulerNode]) -> None:
    """
    # + fuse metadata  # remapping seems not necessary given patched split range method
    + override fused group
    """
    # for node1, node2 in zip(snodes, snodes[1:]):
    #     remap_index(node1, node2)
    monkey.fallback(self, scheduler, snodes)
    self.group = (self.group[0], # max(snodes, key=lambda x: int(x.is_reduction())).group
                  find_main_body_group(snodes))


@monkey.patch(TritonScheduling)
def generate_node_schedule(self: TritonScheduling, nodes, numel, rnumel):
    """
    + general conditions for fusion
    """
    node_schedule: list = []  # if rnumel == 1 else [EnableReduction]
    done: OrderedSet[scheduler.BaseSchedulerNode] = OrderedSet()
    # Writes with a reduced shape, meaning they are only present once the
    # reduction loop has ended
    not_ready_yet_nodes = [OrderedSet[str]()]
    rnumels = [rnumel] if rnumel != 1 else []  # stack of reduction ranges
    EnableReduction.context = [rnumels]  # a log of rnumels history

    # TODO @bozhiyou one shot range should be kernel level metadata
    one_shot_range = {}
    for n in itertools.chain(node.get_nodes() for node in nodes):
        for prefix, size in getattr(n, 'one_shot', {}).items():
            if prefix in one_shot_range:
                assert size == one_shot_range[prefix], f"{prefix}: {size}/{one_shot_range[prefix]}"
                continue
            one_shot_range[prefix] = size

    def fits_in_main_body(n):
        """fits in kernel"""
        node_numel, node_rnumel = get_numels(n)
        return node_numel % numel == 0

    # def fits_outside_reduction(n):
    #     _, (node_numel, node_rnumel) = n.group
    #     return (node_numel % numel) == 0
    #     # return node_numel == numel and node_rnumel == 1 and rnumel != 1

    def schedule_node_in_loop(n):
        nonlocal numel, rnumel, rnumels

        nlevels = requires_closing_previous_reduction(node)
        for _ in range(nlevels):
            # end_current_reduction_loop
            assert not (node_schedule and node_schedule[-1] in (EnableReduction, DisableReduction)), f"reduction enabled/disabled with noop {node_schedule}"
            node_schedule.append(DisableReduction)
            not_ready_yet_nodes.pop()
            rnumels.pop()

        # TODO @bozhiyou EnableReduction/DisableReduction can be instances with ranges to avoid recomputation
        ranges = n.get_ranges(one_shot_range)
        (new_numels, *new_rnumels), _ = self.kernel_type._split_iteration_ranges([numel, *rnumels], ranges)#[[node_numel], [node_rnumel]])
        new_numel = sympy_product(new_numels)
        new_rnumels = list(rnumel for rnumel in itertools.chain.from_iterable(new_rnumels) if rnumel != 1)
        assert numel == new_numel, f"{new_numel} (expect {numel})"
        if len(new_rnumels) > len(rnumels):
            assert new_rnumels[:len(rnumels)] == rnumels, f"fits_outside_reduction: {rnumels} {new_rnumels}"
            for rnumel in new_rnumels[len(rnumels):]:
                if rnumel != 1:  # TODO @bozhiyou and not persistent_reduction
                    node_schedule.append(EnableReduction)
                    not_ready_yet_nodes.append(OrderedSet())
                    break  # mark once even for multiple levels
            EnableReduction.context.append(new_rnumels)
        elif len(new_rnumels) < len(rnumels):
            assert rnumels[:len(new_rnumels)] == new_rnumels, f"fits_outside_reduction: {rnumels} {new_rnumels}"
            for rnumel in reversed(rnumels[len(new_rnumels):]):
                if rnumel != 1:  # TODO @bozhiyou and not persistent_reduction
                    node_schedule.append(DisableReduction)
                    not_ready_yet_nodes.pop()
                    break  # mark once even for multiple levels
            EnableReduction.context.append(new_rnumels)
        else:
            assert rnumels == new_rnumels, f"may need another reduction {rnumels} {new_rnumels}"
        rnumels = new_rnumels
        rnumel = rnumels[-1] if len(rnumels) else 1

        node_schedule.append(n)
        # A scan is modelled as a reduction in the scheduler but has a
        # full sized output that can be used inside the loop body
        if (
            n.is_reduction()
            and isinstance(n, scheduler.SchedulerNode)
            and isinstance(n.node, ir.ComputedBuffer)
            and not isinstance(n.node.data, ir.Scan)
        ):
            not_ready_yet_nodes[-1].add(n.get_name())

    # @contextlib.contextmanager
    # def end_current_reduction_loop():
    #     if node_schedule and node_schedule[-1] is EnableReduction:
    #         node_schedule.pop()
    #     else:
    #         node_schedule.append(DisableReduction)
    #     prior = rnumels.pop()
    #     not_ready_yet_nodes.pop()
    #     yield
    #     node_schedule.append(EnableReduction)
    #     not_ready_yet_nodes.append(OrderedSet())
    #     rnumels.append(prior)

    def requires_closing_previous_reduction(node: SchedulerNode):
        for i, nryn in enumerate(reversed(not_ready_yet_nodes)):
            if not_ready_yet_ancestors:= nryn & node.ancestors:
                if node.is_reduction() and all(node.scheduler.name_to_node[n].is_reduction() for n in not_ready_yet_ancestors):
                    continue  # fused dependent reduction does not require closing previous reduction
                return i + 1  # number of levels to pop
        return 0

    for node in nodes:
        if node in done:
            continue
        done.add(node)

        if fits_in_main_body(node):
            schedule_node_in_loop(node)
        # elif fits_outside_reduction(node):
        #     with end_current_reduction_loop():
        #         node_schedule.append(node)
        #         raise NotImplementedError("@bozhiyou maybe new loop")
        else:
            raise NotImplementedError(
                f"unexpected group: ({numel}, {rnumel}) != {node.group[1]}"
            )

    # if rnumel != 1:
    #     node_schedule.append(DisableReduction)
    return node_schedule


def create_kernel(self: TritonScheduling, kernel_type: type, nodes, *kernel_args, **kernel_kwargs):
    """Like ComboKernel.create_triton_kernel"""
    numel, rnumel = find_main_body_group(nodes)
    node_schedule = self.generate_node_schedule(nodes, numel, rnumel)

    tiled_groups = self.select_tiling(node_schedule, numel, rnumel)
    reduction_hint_val, mutations, index_dtype = self.get_kernel_args(
        node_schedule, numel, rnumel
    )    

    def _create_kernel(*kernel_args, **kernel_kwargs) -> Kernel:
        kernel_kwargs = dict(
            reduction_hint=reduction_hint_val,
            mutations=mutations,
            index_dtype=index_dtype,
        ) | kernel_kwargs
        return kernel_type(
                *tiled_groups,
                *kernel_args,
                reduction_hint=reduction_hint_val,
                mutations=mutations,
                index_dtype=index_dtype,
            )

    # if kernel_args or kernel_kwargs:
    #     return node_schedule, _create_kernel(*kernel_args, **kernel_kwargs), _create_kernel

    return node_schedule, _create_kernel()



@monkey.patch(TritonScheduling)
def generate_kernel_code_from_nodes(self: TritonScheduling, nodes, benchmark_kernel=False):
    """
    + override kernel creation
    """
    @dataclasses.dataclass
    class LastUsageHolder:
        n: Any
        last_usage: Any

        def __del__(self) -> None:
            self.n.last_usage = self.last_usage

    last_usage_holders = [LastUsageHolder(n, n.last_usage) for n in nodes]

    # empty last_usage. May cause more aggressive 'evict_last'. Should be fine.
    for n in nodes:
        n.last_usage = OrderedSet()

    # nodes are output of snode.get_nodes()
    if not nodes[0].is_template():
        # _, (numel, rnumel) = max(nodes, key=lambda x: int(x.is_reduction())).group
        # node_schedule = self.generate_node_schedule(nodes, numel, rnumel)

        # tiled_groups = self.select_tiling(node_schedule, numel, rnumel)
        # reduction_hint_val, mutations, index_dtype = self.get_kernel_args(
        #     node_schedule, numel, rnumel
        # )

        # kernel = self.kernel_type(
        #     *tiled_groups,
        #     reduction_hint=reduction_hint_val,
        #     mutations=mutations,
        #     index_dtype=index_dtype,
        # )
        node_schedule, kernel = create_kernel(self, self.kernel_type, nodes)

        self.codegen_node_schedule_with_kernel(node_schedule, kernel)
        with torch._inductor.config.patch(
            "benchmark_kernel", benchmark_kernel
        ), V.set_kernel_handler(kernel):
            src_code = kernel.codegen_kernel()
    else:
        template_node, *epilogue_nodes = nodes

        with torch._inductor.config.patch("benchmark_kernel", benchmark_kernel):
            src_code = self.codegen_template(
                template_node, epilogue_nodes, only_gen_src_code=True
            )

    src_code = src_code.replace(str(Placeholder.KERNEL_NAME), "triton_")
    return src_code


# scheduling to final codegen
@monkey.patch(TritonScheduling)
def codegen_node(
    self: TritonScheduling, node: scheduler.FusedSchedulerNode|scheduler.SchedulerNode
):
    """
    + new node schedule heuristics
    TODO @bozhiyou use create_kernel in codegen_node_schedule
    """

    nodes: List[scheduler.SchedulerNode] = node.get_nodes()  # type: ignore[assignment]

    numel, rnumel = find_main_body_group(nodes)
    node_schedule = self.generate_node_schedule(nodes, numel, rnumel)

    buf_accesses = collections.defaultdict(list)
    for node in nodes:
        for access in node.read_writes.reads | node.read_writes.writes:
            buf_accesses[access.name].append(access)

    schedule_log.debug("Schedule:\n %s", node_schedule)

    return self.codegen_node_schedule(node_schedule, buf_accesses, numel, rnumel)



#####
# stage 3: codegen



@dataclasses.dataclass
class TritonCodeGenBuffer:
    body: IndentedBuffer
    indexing_code: IndentedBuffer
    loads: IndentedBuffer
    compute: IndentedBuffer
    stores: IndentedBuffer
    suffix: IndentedBuffer

kernel_code_cache = collections.defaultdict[TritonKernel, list[TritonCodeGenBuffer]](list[TritonCodeGenBuffer])

def push_code(self: TritonKernel):
    cache = TritonCodeGenBuffer(
        self.body,
        self.indexing_code,
        self.loads,
        self.compute,
        self.stores,
        self.suffix,
    )
    kernel_code_cache[self].append(cache)
    self.body = IndentedBuffer()
    self.indexing_code = IndentedBuffer()
    self.loads = IndentedBuffer()
    self.compute = IndentedBuffer()
    self.stores = IndentedBuffer()
    self.suffix = IndentedBuffer()
    return cache

def pop_code(self: TritonKernel):
    assert not (
        self.indexing_code
        or self.loads
        or self.stores
        or self.compute
        or self.suffix
    ), "unflushed code"
    cache = kernel_code_cache[self].pop()
    self.indexing_code = cache.indexing_code
    self.loads = cache.loads
    self.loads.splice(self.body)  # flushed code after load
    self.compute = cache.compute
    self.stores = cache.stores
    self.suffix = cache.suffix
    self.body = cache.body
    return cache

    


@monkey.patch(TritonKernel)
def disable_reduction(self: TritonKernel):
    """
    TODO @bozhiyou Changed semantics to close previous reduction?
    + reduction code indentation
    """
    should_flush = self.range_trees[-1].is_loop

    @contextlib.contextmanager
    def ctx():
        if self.numels[-1] == 1:
            assert not self.inside_reduction
            yield
            return
        if should_flush:
            # calling codegen_body() will flush all the pending buffers
            # and write out a reduction loop
            self.codegen_body()
        self.inside_reduction |= self.range_trees[-1].is_loop
        try:
            yield
            if should_flush:
                # flush out any code before opening the next loop
                self.codegen_body()
        finally:
            self.inside_reduction |= self.range_trees[-1].is_loop

    return ctx()


@monkey.patch(IterationRangesRoot)
def index_sym(self: IterationRangesRoot):
    return self.symbol()


@monkey.patch(TritonKernel)
def codegen_range_tree(self: TritonKernel) -> None:
    f"""Also patched in load_block.
    {TritonKernel} does codegen for range trees right after they are initialized
    which assumes linear blocking for each range tree (thus single block size).

    Delay this until range tree hierarchy is known.
    Also delay rbase generation until codegen_body.
    """
    @monkey.patch(self)
    def finalize_indexing(indices: Sequence[sympy.Expr]):
        # NOTE @bozhiyou by this point, `indices` should include blocking info
        monkey.fallback(indices)
        for tree in self.range_trees:
            # reduction indexing goes inside a loop
            if not tree.is_loop:
                self.iteration_ranges_codegen_header(tree, self.body)


@monkey.patch(TritonKernel)
def filter_masks(self: TritonKernel, mask_vars) -> None:
    """
    + add mask suffix
    """
    masks = OrderedSet[str](f"{tree.prefix}mask" for tree in self.range_trees)
    if mask_vars != masks:
        return monkey.fallback(self, mask_vars)
    mask_vars.clear()
    ndims = self.triton_tensor_ndim()
    for tree in self.range_trees:
        if self._has_constant_mask(tree):
            continue
        if tree.tensor_dim is not None:
            sizes = ["None"] * ndims
            sizes[tree.tensor_dim] = ":"
            size_suffix = f"[{', '.join(sizes)}]"
        else:
            size_suffix = ''
        if len(tree.var_list) > 1:
            for var in tree.var_list:
                if block_meta:= getattr(tree, 'block_meta', None):
                    if block_meta.var_to_block(var).block == 1:
                        continue
                mask_vars.add((f"{tree.prefix}mask{var.name[1:]}{size_suffix}"))
            continue
        mask_vars.add(f"{tree.prefix}mask{tree.name.split('index')[-1]}{size_suffix}")



@monkey.patch(TritonKernel)
def iteration_ranges_ranges_code(self: TritonKernel, entry, block_size=''):
    """Also patched in load_block."""
    assert entry.tensor_dim is not None or entry.prefix == 'r'
    # size = self.indexing_size_str(entry.tensor_dim)
    index_dtype = self.index_dtype
    convert = f".to({index_dtype})" if index_dtype != "tl.int32" else ""
    block_size = block_size or f"{entry.prefix.upper()}BLOCK"
    # return f"tl.arange(0, {block_size}){size}{convert}"
    return f"tl.arange(0, {block_size}){convert}"


@monkey.patch(TritonKernel)
def codegen_body(self: TritonKernel):
    """Flush the codegen buffers.
    + rbase generated here
    + inline rnumel and adaptive `roffset` and `RBLOCK` symbols
    """
    if not (
        self.indexing_code
        or self.loads
        or self.stores
        or self.compute
        or self.suffix
    ):
        return

    if self.inside_reduction and self.range_trees[-1].is_loop:
        prefix, suffix = self.range_trees[-1].name.split('index')
        block = f'{prefix.upper()}BLOCK{suffix}'
        constexprs = getattr(self.args, 'constexprs', dict[str, int|None]())
        constexprs[block] = self.range_trees[-1].numel
        setattr(self.args, 'constexprs', constexprs)
        self.body.writeline(
            f"{prefix}base{suffix} = {self.iteration_ranges_ranges_code(self.range_trees[-1], block_size=f'{prefix.upper()}BLOCK{suffix}')}"
        )
        self.body.writeline(f"for {prefix}offset{suffix} in range(0, {self.range_trees[-1].numel}, {prefix.upper()}BLOCK{suffix}):")
        with self.body.indent():
            # last range tree is always reduction
            self.iteration_ranges_codegen_header(self.range_trees[-1], self.body)
            self.body.splice(self.indexing_code)
            self.body.splice(self.loads)
            self.body.splice(self.compute)
            self.body.splice(self.stores)

        # invalidate any caches that came from inside the reduction loop
        self.cse.invalidate(self.outside_loop_vars)
        self.range_trees[-1].cache_clear()
    else:
        self.body.splice(self.indexing_code)
        self.body.splice(self.loads)
        self.body.splice(self.compute)
        self.body.splice(self.stores)
    self.body.splice(self.suffix)
    self.indexing_code.clear()
    self.loads.clear()
    self.compute.clear()
    self.stores.clear()
    self.suffix.clear()



def lookup_one_shot_var(self: IterationRangesRoot, size: sympy.Integer):
    """
    kernel/rangetree level one shot range
    """
    one_shot_ranges = getattr(self, 'one_shot', {})
    if size in one_shot_ranges:
        return one_shot_ranges[size]
    # create new symbol
    var = sympy_index_symbol(f"{self.prefix}{next(self.kernel.iter_vars_count)}")
    one_shot_ranges[size] = var
    setattr(self, 'one_shot', one_shot_ranges)
    if not self.is_loop:
        self.var_ranges[var] = size  # no need to keep loop range for oneshot var, but needed for masking parallel one shot var
    fusion_log.debug(f"add {var}: {size} to tree {self.name}")
    return var

def set_one_shot_ranges(self: TritonKernel, vars: list[list[sympy.Symbol]], one_shot_ranges: dict[int, int]):
    processed = set()
    lo = sympy.Integer(1)
    hi = lo
    for itree, tree in enumerate(self.range_trees):
        hi *= tree.numel
        for prefix, size in one_shot_ranges.items():
            if prefix not in processed and prefix <= hi:
                one_shot_var = lookup_one_shot_var(self.range_trees[itree], size)
                vars[itree].append(one_shot_var)
                processed.add(prefix)
                if treex:= getattr(tree, 'block_meta', None):
                    dense_numel_list = [r.dense_numel for r in treex.ranges]
                    for i, ran9e in enumerate(treex.ranges):
                        if lo == prefix:
                            break
                        lo *= ran9e.dense_numel
                    assert lo == prefix, f"{lo}, {prefix}"
                    treex.insert_range(i, numel=sympy.Integer(1), stride=ran9e.stride, var_list=[one_shot_var])
        lo = hi


def handle_one_shot_ranges(self: TritonKernel, itervars):
    """
    Node level one shot range
    """
    for tree in self.range_trees:
        # reset: remove all one shot ranges
        if treex:= getattr(tree, 'block_meta', None):
            treex.ranges = [r for r in treex.ranges if r.numel != 1]

    if one_shot_ranges:= getattr(self.current_node, 'one_shot', {}):
        set_one_shot_ranges(self, itervars, one_shot_ranges)


@monkey.patch(TritonKernel)
def split_and_set_ranges(self: TritonKernel, lengths: list[list[sympy.Expr]]):
    """
    + Apply block hints
    + Adapt rnumel to the current lengths
    """
    # TODO @bozhiyou this is ad hoc to fix rnumel == 1; should be no_r_dim if rnumel == 1
    self.range_trees = [tree for tree in self.range_trees if tree.numel != 1]

    groups = [rt.numel for rt in self.range_trees]  # [:1] + self.reduction_range_trees]
    if not self.inside_reduction:
        groups = [sympy.Integer(1) if rt.is_loop else rt.numel for rt in self.range_trees]

    if len(lengths) == len(self.range_trees) and all(
        V.graph.sizevars.simplify(sympy_product(x) - g) == 0
        for x, g in zip(lengths, groups)
    ):
        itervars = self.set_ranges(*lengths)
        handle_one_shot_ranges(self, itervars)
        fusion_log.debug(f"{itervars}")
        return itervars

    rtree = self.range_trees[-1]
    rnumels = [rt.numel for rt in self.range_trees if rt.is_loop]
    new_ranges, return_getters_groups = self._split_iteration_ranges(
        groups, lengths
    )
    # assert V.graph.sizevars.simplify(sympy_product(lengths[-1])) == V.graph.sizevars.simplify(new_ranges[-1][-1]), "TODO @bozhiyou store list of rnumels"
    new_rnumels = new_ranges[len(groups) - len(rnumels):]
    new_rnumels = list(itertools.chain.from_iterable(new_rnumels))
    if len(new_rnumels) > len(rnumels):
        assert new_rnumels[:len(rnumels)] == rnumels, f"range hierarchy mismatch: {rnumels} {new_rnumels}"
        for i, rnumel in enumerate(new_rnumels[len(rnumels):]):
            code_buffer = push_code(self)
            if self.range_trees:
                setattr(self.range_trees[-1], 'code_buffer', code_buffer)
            # if rnumel < TRITON_MAX_RBLOCK:
            #     raise
            new_tree = IterationRangesRoot(
                name = f"rindex{(len(rnumels) + i) or ''}",
                numel = rnumel,
                prefix= 'r',
                index = len(self.range_trees),
                kernel=self,
                pid_cache=rtree.pid_cache,
                is_loop=rnumel != 1 and not self.persistent_reduction,
                tensor_dim=rtree.tensor_dim + 1 + i,
                grid_dim=None,
                has_zdim=rtree.has_zdim,
            )
            self.range_trees.append(new_tree)
            # self.reduction_range_trees.append(self.range_trees[-1])
    elif len(new_rnumels) < len(rnumels):
        assert rnumels[:len(new_rnumels)] == new_rnumels, f"range hierarchy mismatch: {rnumels} {new_rnumels}"
        for _ in rnumels[len(new_rnumels):]:
            # self.reduction_range_trees.pop()
            self.range_trees.pop()
            if len(kernel_code_cache[self]):
                pop_code(self)
    else:
        assert rnumels == new_rnumels, f"may need another reduction {rnumels} {new_rnumels}"
    # self.range_trees[-1] = self.reduction_range_trees[-1]
    if len(new_rnumels):
        self.numels[-1] = self.range_trees[-1].numel
    self.inside_reduction |= self.numels[-1] != 1

    # SEEMS WRONG, TO BE REMOVED: tensor_dim calibration 
    # if self.current_node.is_reduction():
    #     reduction_hint = self.current_node.node.data.reduction_hint
    #     tensor_dim = itertools.count()
    #     for i, tree in enumerate(reversed(self.range_trees)):
    #         if reduction_hint == ir.ReductionHint.INNER and i < len(lengths[-1]):
    #             tree.tensor_dim = None
    #             continue
    #         tree.tensor_dim = next(tensor_dim)

    itervars = list(itertools.chain.from_iterable(self.set_ranges(*new_ranges)))
    itervars = [[fn(itervars) for fn in fns] for fns in return_getters_groups]
    handle_one_shot_ranges(self, itervars)
    fusion_log.debug(f"{itervars}")
    return itervars


@monkey.patch(TritonKernel)
@staticmethod
def _split_iteration_ranges(
    groups: Iterable[sympy.Expr], lengths: Sequence[Sequence[sympy.Expr]]
):
    """
    Re-group `lengths` into `groups`.
    `lengths`: groups of lengths/ranges.
    `groups`: kernel group sizes.
    - handles out-of-range sizes in add_range
    """
    sv = V.graph.sizevars
    new_ranges: list[list[sympy.Expr]] = []  # [[] for _ in groups]
    remaining = [sv.simplify(g) for g in groups]
    var_count = itertools.count()

    def add_range(i, expr):
        """Consume i-th group by expr"""
        nonlocal current_group
        expr = sv.simplify(expr)
        if i == len(new_ranges):
            new_ranges.append([])
        if i >= len(remaining):
            new_ranges[i].append(expr)
            current_group += 1
            return next(var_count)
        if not sv.statically_known_multiple_of(remaining[i], expr):
            raise CantSplit
        # guard on the last item out
        remaining[i] = FloorDiv(remaining[i], expr)
        new_ranges[i].append(expr)
        return next(var_count)

    def make_combined(size, idx1, idx2):
        def getter(flat_vars):
            return size * flat_vars[idx1] + flat_vars[idx2]

        return getter

    return_getters_groups = []
    current_group = 0
    for length_group in lengths:
        return_getters = []
        for size in length_group:
            if sv.statically_known_equals(size, 1):  # type: ignore[arg-type]
                return_getters.append(lambda _: sympy.Integer(0))
                continue

            while current_group < len(remaining) and sv.statically_known_equals(
                remaining[current_group], 1  # type: ignore[arg-type]
            ):
                # scroll to next group with remaining elements
                current_group += 1

            if current_group < len(remaining) and sv.statically_known_gt(
                size, remaining[current_group]
            ):
                # need to break size in two groups
                if not sv.statically_known_multiple_of(
                    size, remaining[current_group]
                ):
                    raise CantSplit
                size1 = remaining[current_group]
                size2 = FloorDiv(size, remaining[current_group])
                return_getters.append(
                    make_combined(
                        size2,
                        add_range(current_group, size1),
                        add_range(current_group + 1, size2),
                    )
                )
            else:  # consume the size
                return_getters.append(
                    operator.itemgetter(add_range(current_group, size))
                )

        return_getters_groups.append(return_getters)

    assert all(
        V.graph.sizevars.size_hint(s) == 1 for s in remaining[:current_group + 1]
    ), f"failed to set ranges {remaining} {lengths} {groups} {current_group}"

    return new_ranges, return_getters_groups
