from .. import _monkey as monkey
from typing import Any, Sequence, Iterable

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
from torch._inductor.utils import sympy_product, sympy_subs, IndentedBuffer
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv

import sympy


# add 'fusion' to comma-separated TORCH_LOG to enable
fusion_log = torch._logging.getArtifactLogger('torch._inductor', "fusion")  # fusion_log.debug(...)
schedule_log = torch._logging.getArtifactLogger('torch._inductor.codegen.simd', "schedule")

from triton.language import TRITON_MAX_TENSOR_NUMEL
#####
# stage 1: matching
# check range compatibility

def reduction_can_fuse(node1, node2):
    _, (numel1, rnumel1) = node1.group
    _, (numel2, rnumel2) = node2.group
    if node1.is_reduction():
        if node2.is_reduction():
            assert rnumel1 != 1 and rnumel2 != 1, f"trivial reduction {rnumel1=} {rnumel2=}"
            reduction_can_fuse = (numel1 == numel2 and rnumel1 == rnumel2  # default condition
                            ) or (numel1 == numel2 * rnumel2  # reduction over previous result
                            )
        else:
            reduction_can_fuse = (numel1 * rnumel1 == numel2 and rnumel2 == 1)  # split second range
        return reduction_can_fuse
    return False


@monkey.patch(scheduler.Scheduler)
def can_fuse(self: scheduler.Scheduler,
             node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
    _, (numel1, rnumel1) = node1.group
    _, (numel2, rnumel2) = node2.group

    if node1.is_reduction():
        return reduction_can_fuse(node1, node2)
    return monkey.fallback(self, node1, node2)


@monkey.patch(scheduler.Scheduler)
def can_fuse_vertical(self: scheduler.Scheduler,
    node1: BaseSchedulerNode, node2: BaseSchedulerNode
):
    if monkey.fallback(self, node1, node2):
        return True
    return reduction_can_fuse(node1, node2)


@monkey.patch(TritonScheduling)
def can_fuse_vertical(self: TritonScheduling, node1, node2) -> bool:
    """Add one reduction fusion rule."""
    if monkey.fallback(self, node1, node2):
        return True
    return reduction_can_fuse(node1, node2)


#####
# stage 2: scheduling
# collect iteration context and do metadata fusion

def find_main_body_group(nodes: Sequence[SchedulerNode]):
    """
    Used to be the group of first reduction or first node if no reduction:
        _, (numel, rnumel) = max(nodes, key=lambda x: int(x.is_reduction())).group
    Now rnumel is the outter-most reduction range.
    """
    assert nodes
    _, (numel, rnumel) = nodes[0].group
    for node in nodes[1:]:
        _, (node_numel, node_rnumel) = node.group
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


@monkey.patch(FusedSchedulerNode)
def __init__(self:FusedSchedulerNode, scheduler: scheduler.Scheduler, snodes: list[BaseSchedulerNode]) -> None:
    monkey.fallback(self, scheduler, snodes)
    self.group = (self.group[0], # max(snodes, key=lambda x: int(x.is_reduction())).group
                  find_main_body_group(snodes))


def remap_index(prior: SchedulerNode, node: SchedulerNode):
    _, (numel, rnumel) = prior.group
    _, (node_numel, node_rnumel) = node.group

    if numel == node_numel * node_rnumel:   # reduction over previous result
        assert functools.reduce(
            operator.mul, itertools.chain.from_iterable(prior._body.sizes[:-1])
            ) % node._body.sizes[-1][0] == 0, "cannot align reductions"

        def find_var_mapping(prior_body: LoopBody, node_body: LoopBody):
            backward_mapping = {}
            anc_i = len(prior_body.iter_vars) - 1
            ran9e = anc_range = 1
            for var in reversed(list(itertools.chain.from_iterable(node_body.vars))):
                if anc_i < 0:
                    break
                expr = sympy.Integer(0)
                ran9e *= node_body.var_ranges[var]
                while ran9e > anc_range:
                    anc_var = prior_body.iter_vars[anc_i]
                    anc_i -= 1
                    expr += anc_range * anc_var
                    anc_range *= prior_body.var_ranges[anc_var]
                
                if ran9e < anc_range:
                    raise NotImplementedError("should remap from the other direction")
                assert ran9e == anc_range, f"unmatch ranges {prior_body.var_ranges} {node_body.var_ranges}"
                if var != expr:
                    backward_mapping[var] = expr
                ran9e = anc_range = 1
            return backward_mapping
        var_remapping = find_var_mapping(prior._body, node._body)

        index_remapping = {}
        indexing_exprs = {n: e for n, e in prior._body.indexing_exprs.items() if  # filter reduction only indexing
                          any(s in e.free_symbols for s in prior._body.iter_vars)}
        for name, ind in node._body.indexing_exprs.items():
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
                new_name = f"index{len(node._body.indexing_exprs)}"
                assert new_name not in node._body.indexing_exprs, f"non-contiguous keys {node._body.indexing_exprs}"
                node._body.indexing_exprs[new_name] = known_ind

    # node._sizes = anc._sizes
    # node._body.iter_vars = anc._body.iter_vars
    # node._body.reduce_vars = anc._body.reduce_vars
    # node._body.var_ranges = anc._body.var_ranges
    # node._body.sizes = anc._body.sizes
    # node._body.indexing_exprs = indexing_exprs

@monkey.patch(FusedSchedulerNode)
def __init__(self: FusedSchedulerNode, scheduler: scheduler.Scheduler, snodes: list[scheduler.BaseSchedulerNode]) -> None:
    """
    + fuse metadata
    + override fused group
    """
    for node1, node2 in zip(snodes, snodes[1:]):
        remap_index(node1, node2)
    monkey.fallback(self, scheduler, snodes)
    self.group = (self.group[0], find_main_body_group(snodes))


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
    rnumels = [rnumel]  # stack of reduction ranges
    EnableReduction.context = [rnumels]  # a log of rnumels history

    def fits_in_main_body(n):
        """fits in kernel"""
        _, (node_numel, node_rnumel) = n.group
        return node_numel % numel == 0

    # def fits_outside_reduction(n):
    #     _, (node_numel, node_rnumel) = n.group
    #     return (node_numel % numel) == 0
    #     # return node_numel == numel and node_rnumel == 1 and rnumel != 1

    def schedule_node_in_loop(n):
        nonlocal numel, rnumel, rnumels

        nlevels = requires_closing_previous_reduction(node)
        if nlevels:
            for _ in range(nlevels):
                # end_current_reduction_loop
                assert not (node_schedule and node_schedule[-1] in (EnableReduction, DisableReduction)), f"reduction enabled/disabled with noop {node_schedule}"
                node_schedule.append(DisableReduction)
                not_ready_yet_nodes.pop()
                rnumels.pop()

        # TODO @bozhiyou EnableReduction/DisableReduction can be instances with ranges to avoid recomputation
        _, (node_numel, node_rnumel) = n.group
        ((new_numel,), *new_rnumels), _ = self.kernel_type._split_iteration_ranges([numel, *rnumels], [[node_numel], [node_rnumel]])
        new_rnumels = list(itertools.chain.from_iterable(new_rnumels))
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

    def requires_closing_previous_reduction(node):
        for i, nryn in enumerate(reversed(not_ready_yet_nodes)):
            if nryn & node.ancestors:
                return i + 1  # number of levels to pop
        return None

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

def tuple_append(t, x):
    if isinstance(t, tuple):
        return (*t, x)
    return (t, x)


@monkey.patch(TritonScheduling)
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


#####
# stage 3: codegen

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
        node_schedule, kernel = self.create_kernel(self.kernel_type, nodes)

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


#####
# final codegen

@monkey.patch(TritonScheduling)
def codegen_node(
    self: TritonScheduling, node: scheduler.FusedSchedulerNode|scheduler.SchedulerNode
):
    """
    + new node schedule heuristics
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



def codegen_node_schedule(
    self, node_schedule, buf_accesses, numel, reduction_numel
):
    from torch._inductor.codegen.triton_split_scan import TritonSplitScanKernel

    # tiled_groups = self.select_tiling(node_schedule, numel, reduction_numel)
    # (
    #     reduction_hint_val,
    #     mutations,
    #     index_dtype,
    # ) = self.get_kernel_args(node_schedule, numel, reduction_numel)

    is_split_scan = any(
        isinstance(node, BaseSchedulerNode) and node.is_split_scan()
        for node in node_schedule
    )
    kernel_type: type = self.kernel_type
    if is_split_scan and issubclass(TritonSplitScanKernel, kernel_type):
        kernel_type = TritonSplitScanKernel

    # kernel_args = tiled_groups
    # kernel_kwargs = dict(
    #     reduction_hint=reduction_hint_val,
    #     mutations=mutations,
    #     index_dtype=index_dtype,
    # )

    def _node_has_sort(node):
        if node in (EnableReduction, DisableReduction):
            return False

        sort_nodes = node._body.root_block.graph.find_nodes(
            op="call_method", target="sort"
        )
        return bool(sort_nodes)

    # ops.sort only works with persistent reduction, and is not bandwidth bound anyway
    # so taking the hit of non-coalesced loads is okay
    has_sort = any(_node_has_sort(node) for node in node_schedule)
    if has_sort:
        kernel_kwargs["override_persistent_reduction"] = True

    # kernel = kernel_type(
    #     *kernel_args,
    #     **kernel_kwargs,
    # )
    kernel = create_kernel()
    kernel.buf_accesses = buf_accesses

    kernel2: Optional[SIMDKernel] = None
    if kernel.persistent_reduction and config.triton.multi_kernel and not has_sort:
        kernel2 = self.kernel_type(
            *kernel_args,
            **kernel_kwargs,
            override_persistent_reduction=False,
        )
        self.codegen_node_schedule_with_kernel(node_schedule, kernel2)
        with V.set_kernel_handler(kernel2):
            src_code2 = kernel2.codegen_kernel()
        kernel_name2 = self.define_kernel(src_code2, node_schedule, kernel)
        kernel2.kernel_name = kernel_name2
        kernel2.code_hash = code_hash(src_code2)

        # Keep buffers needed by the non-persistent reduction so both
        # kernels have the same arguments
        kernel.must_keep_buffers = set(kernel2.must_keep_buffers)

    self.codegen_node_schedule_with_kernel(node_schedule, kernel)

    with V.set_kernel_handler(kernel):
        src_code = kernel.codegen_kernel()

    kernel_name = self.define_kernel(src_code, node_schedule, kernel)
    log.debug("Generating kernel code with kernel_name: %s", kernel_name)
    kernel.kernel_name = kernel_name
    kernel.code_hash = code_hash(src_code)

    final_kernel = MultiKernel([kernel, kernel2]) if kernel2 is not None else kernel

    with V.set_kernel_handler(final_kernel):
        for node in node_schedule:
            if node not in (EnableReduction, DisableReduction):
                node.mark_run()

    self.codegen_comment(node_schedule)
    final_kernel.call_kernel(final_kernel.kernel_name)

    if config.nan_asserts:
        final_kernel.codegen_nan_check()
    if config.warn_mix_layout:
        final_kernel.warn_mix_layout(kernel_name)

    V.graph.removed_buffers |= final_kernel.removed_buffers
    V.graph.inplaced_to_remove |= final_kernel.inplaced_to_remove

    if (
        V.graph.wrapper_code.supports_intermediate_hooks
        and config.generate_intermediate_hooks
    ):
        # Not every node in the schedule will actually be live on output;
        # we can't check dead buffers.
        live_outs = kernel.args.live_output_buffers()
        for node in node_schedule:
            if not isinstance(node, scheduler.BaseSchedulerNode):
                continue
            name = node.get_name()
            if name not in live_outs:
                continue
            assert node.node is not None
            origin_node = node.node.get_origin_node()
            if origin_node is not None:
                counters["inductor"]["intermediate_hooks"] += 1
                V.graph.wrapper_code.writeline(
                    f"run_intermediate_hooks({origin_node.name!r}, {name})"
                )

    self.scheduler.free_buffers()

# final codegen
#####


# @monkey.patch(TritonKernel)
def reduction_loop_context(self: TritonKernel):
    """Adapted from disable_reduction.
    + reduction code indentation
    """
    should_flush = self.range_trees[-1].is_loop
    @contextlib.contextmanager
    def ctx():
        if should_flush and (
            self.indexing_code
            or self.loads
            or self.stores
            or self.compute
            or self.suffix
        ):
            self.codegen_body()
        self.body.writeline(f"for roffset in range(0, rnumel, RBLOCK):")
        with self.body.indent():
            yield

    return ctx()

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
    self.loads.splice(self.body)
    self.compute = cache.compute
    self.stores = cache.stores
    self.suffix = cache.suffix
    self.body = cache.body

    


@monkey.patch(TritonKernel)
def disable_reduction(self: TritonKernel):
    """
    TODO @bozhiyou Changed semantics to close previous reduction
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
        self.inside_reduction = self.range_trees[-1].is_loop
        try:
            yield
            if should_flush:
                # flush out any code before opening the next loop
                self.codegen_body()
        finally:
            self.inside_reduction = self.range_trees[-1].is_loop

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
    """
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
        constexprs = getattr(self.args, 'constexprs', set[str]())
        constexprs.add(block)
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


# @monkey.patch(TritonKernel)
# def initialize_range_tree(self: TritonKernel, pid_cache) -> None:
#     try:
#         return monkey.fallback(self, pid_cache)
#     finally:
#         self.reduction_range_trees = [rt for rt in self.range_trees if rt.is_loop]



@monkey.patch(TritonKernel)
def split_and_set_ranges(self: TritonKernel, lengths: list[list[sympy.Expr]]):
    """
    + Apply block hints
    + Adapt rnumel to the current lengths
    """
    groups = [rt.numel for rt in self.range_trees]  # [:1] + self.reduction_range_trees]
    if not self.inside_reduction:
        groups[-1] = sympy.Integer(1)

    if len(lengths) == len(self.range_trees) and all(
        V.graph.sizevars.simplify(sympy_product(x) - g) == 0
        for x, g in zip(lengths, groups)
    ):
        return self.set_ranges(*lengths)

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
            self.range_trees.append(IterationRangesRoot(
                name = f"rindex{len(rnumels) + i}",
                numel = rnumel,
                prefix= 'r',
                index = len(self.range_trees),
                kernel=self,
                pid_cache=rtree.pid_cache,
                is_loop=rnumel != 1 and not self.persistent_reduction,
                tensor_dim=rtree.tensor_dim,
                grid_dim=rtree.grid_dim,
                has_zdim=rtree.has_zdim,
            ))
            # self.reduction_range_trees.append(self.range_trees[-1])
            push_code(self)
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
    self.numels[-1] = self.range_trees[-1].numel
    self.inside_reduction = self.numels[-1] != 1

    # tensor_dim calibration 
    if self.current_node.is_reduction():
        reduction_hint = self.current_node.node.data.reduction_hint
        tensor_dim = itertools.count()
        for i, tree in enumerate(reversed(self.range_trees)):
            if reduction_hint == ir.ReductionHint.INNER and i < len(lengths[-1]):
                tree.tensor_dim = None
                continue
            tree.tensor_dim = next(tensor_dim)

    itervars = list(itertools.chain.from_iterable(self.set_ranges(*new_ranges)))
    fusion_log.debug(f"{itervars}")
    return [[fn(itervars) for fn in fns] for fns in return_getters_groups]

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
        expr = sv.simplify(expr)
        if i == len(new_ranges):
            new_ranges.append([])
        if i >= len(remaining):
            new_ranges[i].append(expr)
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
            else:  # consume the group
                return_getters.append(
                    operator.itemgetter(add_range(current_group, size))
                )

        return_getters_groups.append(return_getters)

    assert all(
        V.graph.sizevars.size_hint(s) == 1 for s in remaining[:current_group + 1]
    ), f"failed to set ranges {remaining} {lengths} {groups} {current_group}"

    return new_ranges, return_getters_groups
