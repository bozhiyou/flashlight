from .. import _monkey as monkey
from typing import Any, Sequence, Callable

import collections
import contextlib
import dataclasses
import itertools

import torch
import torch._inductor.config
from torch._inductor import ir, lowering as L
from torch._inductor.virtualized import V
from torch._inductor.kernel.bmm import tuned_bmm, mm_args
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.codegen.common import SizeArg
from torch._inductor.codegen.triton import TritonScheduling, TritonKernel, TritonKernelOverrides
from torch._inductor.codegen.simd import IterationRangesRoot, IterationRangesEntry, EnableReduction, DisableReduction
from torch._inductor.optimize_indexing import indexing_dtype_strength_reduction
from torch._inductor.utils import ceildiv, sympy_index_symbol, sympy_product, IndentedBuffer
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.wrapper_benchmark import _kernel_category_choices

import sympy
from sympy import Expr
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._ordered_set import OrderedSet

from torch._inductor.virtualized import ops, OpsWrapper, OpsValue

# add 'fusion' to comma-separated TORCH_LOG to enable
fusion_log = torch._logging.getArtifactLogger('torch._inductor', "fusion")  # fusion_log.debug(...)

# do not merge loops before fusion
torch._inductor.config.loop_ordering_after_fusion = True

# TODO @bozhiyou debug persistent reduction
@monkey.patch(TritonKernel)
def should_use_persistent_reduction(self):
    """persistent reduction not debugged yet; disable it for now"""
    return False


#####
# new IR for bmm, which used to be templates for block indexing
#####

@dataclasses.dataclass
class BlockReduction(ir.Reduction):
    """
    BlockReduction performs persistent reduction on every block and accumulates the result. No final reduction is required.
        (x0+x1+x2)+(x3+x4+x5)
    Reduction (the original Inductor IR) performs multi-lane reduction. A final reduction is required.
        (x0+x3, x2+x4, x3+x5) -> (x0+x3)+(x2+x4)+(x3+x5)
    """
    @classmethod
    def create(  # type: ignore[override]
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        ranges: list[Expr],
        reduction_ranges: list[Expr],
        reduction_type: str,
        reduction_hint = ir.ReductionHint.DEFAULT,
        input_node: None|ir.IRNode|tuple[ir.IRNode, ...] = None,
    ):
        reduction_numel = V.graph.sizevars.simplify(sympy_product(reduction_ranges))

        # TODO @bozhiyou these currently fall back to Pointwise; make Blockwise?
        if reduction_numel <= 1 or (
            isinstance(reduction_numel, sympy.Integer)
            and V.graph.sizevars.size_hint(reduction_numel)
            < torch._inductor.config.unroll_reductions_threshold
            and sympy_product(ranges) != 1
        ):
            super().create(
                device,
                dst_dtype,
                src_dtype,
                inner_fn,
                ranges,
                reduction_ranges,
                reduction_type,
                reduction_hint,
                input_node,
            )

        # triton doesn't support reduce to single element well, so break it up
        hint, split = cls.num_splits(
            device,
            dst_dtype,
            src_dtype,
            inner_fn,
            ranges,
            reduction_ranges,
            reduction_type,
            reduction_numel,
            input_node,
        )
        # intermediate reduction in split can contain complex indexing,
        # and num_splits will fail to correctly set the hint
        # reuse the passed hint if available
        if reduction_hint == ir.ReductionHint.DEFAULT:
            reduction_hint = hint
        if split == -1:
            assert input_node is not None
            new_ranges, new_reduction_ranges = ir.extract_input_node_reduction_ranges(
                input_node  # type: ignore[arg-type]
            )
            assert new_ranges is not None
            assert new_reduction_ranges is not None
            return cls.create_multilayer_existing_ranges(
                device,
                dst_dtype,
                src_dtype,
                inner_fn,
                ranges,
                reduction_ranges,
                new_ranges,
                new_reduction_ranges,
                reduction_type,
                reduction_hint,
            )
        elif split > 1:
            # triton doesn't support reduce to single element well, so break it up
            return cls.create_multilayer(
                device,
                dst_dtype,
                src_dtype,
                inner_fn,
                ranges,
                reduction_ranges,
                reduction_type,
                split,
                reduction_hint,
            )

        return ir.TensorBox.create(
            cls(
                device,
                dst_dtype,
                inner_fn,
                ranges,
                reduction_ranges,
                reduction_type,
                src_dtype,
                reduction_hint,
            )
        )

    def store_reduction(self, output_name, indexer, vars, reduction_vars):
        value = ops.block_reduction(
            self.dtype,
            self.src_dtype,
            self.reduction_type,
            self.inner_fn(vars, reduction_vars),
        )
        return ops.store_reduction(output_name, indexer(vars), value)



def _make_bmm_inner(m, n, k, layout: ir.Layout, mat1, mat2):
    f"""Adapted from {L._make_reduction_inner}."""
    # x = L.to_dtype(x, dtype)
    def loader(index, reduction_index):
        assert len(reduction_index) == 1
        assert len(index) == len(layout.size)
        mat1_loader = mat1.make_loader()
        mat2_loader = mat2.make_loader()
        mat1_index = index[:-1] + reduction_index
        mat2_index = index[:-2] + reduction_index + index[-1:]
        # return mat1_loader(mat1_index) * mat2_loader(mat2_index)
        return ops.dot(mat1_loader(mat1_index), mat2_loader(mat2_index))

    return dict(
        input_node=(mat1, mat2),
        device=layout.device,
        dst_dtype=layout.dtype,  # TODO @bozhiyou match torch._inductor.kernel.mm_common.acc_type
        src_dtype=layout.dtype,
        inner_fn=loader,
        ranges=layout.size,
        reduction_ranges=[k],
    )


# Unregister original bmm (as template)
# torch.ops.aten.bmm: torch._ops.OpOverloadPacket
for overload in torch.ops.aten.bmm.overloads():
    other_fn = getattr(torch.ops.aten.bmm, overload)
    L.lowerings.pop(other_fn)  # tuned_bmm


@L.register_lowering(torch.ops.aten.bmm)
def bmm(mat1, mat2, *, layout=None):
    f"""Adapted from {L.make_reduction}.<locals>.inner.
    + block hint for scheduling
    """
    if all(x.get_device().type == "cpu" for x in [mat1, mat2]):
        return tuned_bmm(mat1, mat2, layout=layout)

    kwargs = _make_bmm_inner(*mm_args(mat1, mat2, layout=layout))
    result = BlockReduction.create(
        reduction_type='sum',
        reduction_hint = ir.ReductionHint.INNER,
        **kwargs,
    )
    if isinstance(
        result.data.data, ir.Reduction
    ):  # Only realize if reduction isn't unrolled
        result.realize()
    result.data.data.block_hint = [None for _ in range(len(result.data.get_size()) - 2)] + [1, 0]  # per-dimension blocking, no blocking for batch dimensions
    return result



#####
# Implementation of new IR interpretation
#####

def ops_wrapper(f) -> Callable[..., OpsValue]:
    f"""Adapted from {ir.ops_wrapper}."""
    assert callable(f)

    def fn(*args: object, **kwargs: object) -> OpsValue:
        new_args = [OpsWrapper._unwrap(a) for a in args]
        new_kwargs = {k: OpsWrapper._unwrap(v) for k, v in kwargs.items()}
        return OpsWrapper._wrap(f(*new_args, **new_kwargs))

    return fn
    
INPLACE_REDUCTION_COMBINE_FN: dict[str, Callable[..., OpsValue]] = {
    "sum": ops_wrapper(lambda *args: "{} += {}".format(*args)),
    # TODO @bozhiyou support others in ir.REDUCTION_COMBINE_FN
}


@monkey.patch(TritonKernelOverrides)
@staticmethod
def dot(a, b):
    return f"tl.dot({a}, {b})"



from torch._inductor.ops_handler import ReductionType
from torch._inductor.codegen.common import CSEVariable
from torch._inductor.utils import is_welford_reduction, IndentedBuffer
from torch._inductor.codegen.triton import TritonKernelOverrides, TritonKernel, TritonCSEVariable
from torch._inductor.codegen.triton import triton_acc_type, triton_compute_type
from torch._inductor.codegen.simd import constant_repr

@monkey.patch(TritonKernelOverrides)
def block_reduction(
    _self: TritonKernelOverrides,
    dtype: torch.dtype,
    src_dtype: torch.dtype,
    reduction_type: ReductionType,
    value: CSEVariable|tuple[CSEVariable, ...],
) -> CSEVariable|tuple[CSEVariable, ...]:
    self: TritonKernel = V.kernel

    assert self.inside_reduction
    masks = OrderedSet(f"{tree.prefix}mask" for tree in self.range_trees)
    self.filter_masks(masks)
    masks = sorted(masks)
    if self._load_mask:
        masks.append(self._load_mask)
    # reduction_range_prefix = self.range_trees[-1].prefix

    # Say we have
    #     tmp0 = ops.constant(1, torch.int64)
    #     tmp1 = ops.reduction(torch.int64, torch.int64, "sum", tmp0)
    # tmp0 in the triton code is either a scalar, or single-element tensor
    # so if we emit tl.sum directly, it will only give 1 instead of RBLOCK * 1
    # To avoid this, we broadcast to the expected shape first.
    dense_size_str = self.dense_size_str()
    value = self._map_tuple_or_scalar(
        lambda v: self.cse.generate(
            self.compute, f"tl.broadcast_to({v}, {dense_size_str})"
        ),
        value,
    )

    # dim: int

    # def final_reduction(value):
    #     use_helper = reduction_type in {"any", "max", "min", "prod"}
    #     module = "triton_helpers" if use_helper else "tl"
    #     if reduction_type in {"max", "min"}:
    #         return self.reduction_resize(
    #             f"{module}.{reduction_type}2({value}, {dim})"
    #         )
    #     return self.reduction_resize(f"{module}.{reduction_type}({value}, {dim})")

    cache_key = (src_dtype, reduction_type, value)
    if cache_key in self.cse.reduction_cache:
        return self.cse.reduction_cache[cache_key]

    # dim = self.triton_tensor_ndim() - 1
    acc_type = triton_acc_type(src_dtype)
    result_var: Any = self.cse.newvar()
    result_var.mask_vars = OrderedSet(var for var in masks if var[0] != "r")
    cond = " & ".join(masks)

    def where_cond(tval, fval):
        if not cond:
            return tval
        return TritonKernelOverrides.where(cond, tval, fval)

    if self.persistent_reduction:
        raise NotImplementedError("persistent blockwise reduction not implemented")

    # accumulator = f"_{result_var}"
    accumulator = result_var
    default = ir.Reduction.default_accumulator(reduction_type, src_dtype)
    default = self._map_tuple_or_scalar(constant_repr, default)
    if not isinstance(default, tuple):
        self.body.writeline(
            f"{accumulator} = tl.full({self.dense_size_str()}, {default}, {acc_type})"
        )

    if reduction_type in {"argmax", "argmin"} or is_welford_reduction(reduction_type):
        raise NotImplementedError(f"{reduction_type}")

    if reduction_type in INPLACE_REDUCTION_COMBINE_FN: # TODO check masking
        inplace_combine_fn = INPLACE_REDUCTION_COMBINE_FN[reduction_type]
        self.compute.writeline(f"{inplace_combine_fn(accumulator, value)}")
    else:
        combine_fn = ir.get_reduction_combine_fn(reduction_type, src_dtype)
        updated = combine_fn(accumulator, value)
        self.compute.writeline(
            f"{accumulator} = {where_cond(updated, accumulator)}"
        )

    # blockwise reduction does not need final reduction
    # if src_dtype == torch.bool:
    #     # This is only really used for aten.any. It changes the
    #     # final reduction of a non-persistent reduction from
    #     #     tmp5 = triton_helpers.max(_tmp5, 1)[:, None]
    #     # to
    #     #     tmp5 = triton_helpers.max(_tmp5.to(tl.int8), 1)[:, None].to(tl.int1)
    #     # which is needed because tl.reduce doesn't support tl.int1
    #     accumulator = f"{accumulator}.to(tl.int8)"
    #     result_type = triton_compute_type(dtype)
    #     self.suffix.writeline(
    #         f"{result_var} = {final_reduction(accumulator)}.to({result_type})"
    #     )
    # else:
    #     self.suffix.writeline(
    #         f"{result_var} = {final_reduction(accumulator)}"
    #     )

    self.cse.reduction_cache[cache_key] = result_var

    if isinstance(result_var, tuple):
        assert all(isinstance(x, TritonCSEVariable) for x in result_var)
        self.outside_loop_vars |= OrderedSet(result_var)
    else:
        assert isinstance(result_var, TritonCSEVariable)
        self.outside_loop_vars.add(result_var)

    return result_var


######
# Scheduler support for block reduction
######

@monkey.patch(SchedulerNode)
def _init_from_node(self: SchedulerNode, node: ir.Operation) -> None:
    """relay block_hint from inductor ir to scheduler"""
    self.block_hint = getattr(node, 'block_hint', None)
    return monkey.fallback(self, node)


# @bozhiyou
# optional: reduction tiling
# Separate batch dimension
# @monkey.patch(SIMDScheduling)
# @classmethod  # NOTE @bozhiyou this seems not
# def select_tiling(self: TritonScheduling, node_schedule, numel, reduction_numel=sympy.Integer(1)):
#     """
#     Heuristics to decide how to tile kernels.
#     Currently, we tile based on stride-1 dimensions.

#     Returns:
#         `(tile1, tile2, reduction_numel)` s.t. `tile1 * tile2 == numel`

#     """
#     import torch._inductor.config as config
#     if config.triton.max_tiles != 2:
#         return monkey.fallback(self, node_schedule, numel, reduction_numel=reduction_numel)
#         # # TODO(jansel): should we tile reductions?
#         # # do perf hint here if stride-1 dim is not being reduced
#         # if perf_hint_log.level <= logging.WARNING:
#         #     for node in EnableReduction.filter(node_schedule):
#         #         if len(cls.candidate_tilings(node)) > 0:
#         #             perf_hint_log.info("reduction over non-contiguous dims")
#         #             break
#         # return (numel, reduction_numel)

#     import collections
#     from collections import Counter
#     from torch._inductor.codegen.simd import SIMDKernel, EnableReduction, perf_hint_log
#     import torch._inductor.scheduler as scheduler

#     node_ranges = [
#         node.get_ranges()[0]
#         for node in EnableReduction.filter(node_schedule)
#         if isinstance(node, scheduler.SchedulerNode)
#     ]
#     new_tilings: OrderedSet[tuple[sympy.Expr]] = OrderedSet()
#     for node_range in node_ranges:
#         if len(node_range) < 2:
#             return monkey.fallback(self, node_schedule, numel, reduction_numel=reduction_numel)
#         # 2D tiling of last two dims in X; flattened others in Y
#         tiling = [sympy_product(node_range[:-2]), tuple(node_range[-2:])]
#         new_tilings.add(tuple(tiling))
#         return (*tiling, reduction_numel)  # FIXME pass the following check

#     for tiled_groups in new_tilings:
#         new_groups = (*tiled_groups, reduction_numel)
#         if all(
#             SIMDKernel.is_compatible(new_groups, node.get_ranges())
#             for node in node_schedule
#             if isinstance(node, scheduler.SchedulerNode)
#         ):
#             return new_groups

#     return (numel, reduction_numel)


# @monkey.patch(SizeVarAllocator)
# def simplify(self: SizeVarAllocator, expr: Expr|tuple):
#     """Allow virtual tiling when initializing numels"""
#     if isinstance(expr, tuple):
#         return tuple(monkey.fallback(self, e) for e in expr)
#     return monkey.fallback(self, expr)


# @monkey.patch(TritonScheduling)
# def get_kernel_args(self: TritonScheduling, node_schedule, numel, reduction_numel):
#     is_block_reduction = False
#     reductions = list(
#         filter(
#             lambda n: n not in (EnableReduction, DisableReduction)
#             and n.is_reduction(),
#             node_schedule,
#         )
#     )
#     if any(isinstance(node.node.data for node in reductions)):
#         is_block_reduction = True

#     """ piggy-back block reduction hint """
#     reduction_hint_val, mutations, index_dtype = monkey.fallback(
#         self, node_schedule, numel, reduction_numel
#     )
#     pass
#     return reduction_hint_val, mutations, index_dtype



# TODO @bozhiyou this main (debug) codegen loop can be staticmethod
@monkey.patch(TritonScheduling)
def codegen_node_schedule_with_kernel(self: TritonScheduling, node_schedule, kernel: TritonKernel):
    """
    + set_current_node
    """
    def current_reduction_nodes(nodes):
        return itertools.takewhile(lambda n: n is not DisableReduction, nodes)

    with kernel:
        stack = contextlib.ExitStack()
        kernel.set_last_usage(current_reduction_nodes(node_schedule))
        all_indexing = {}

        # First pass to collect indexing and decide inplace updates
        for node in node_schedule:
            if node is DisableReduction:
                stack.enter_context(kernel.disable_reduction())
            elif node is EnableReduction:
                stack.close()
            else:
                with kernel.set_current_node(node):
                    node.decide_inplace_update()
                    index_vars = kernel.split_and_set_ranges(node.get_ranges())
                    all_indexing.update(
                        dict.fromkeys(
                            node._body.indexing_from_args(index_vars).values()
                        )
                    )

        kernel.finalize_indexing(all_indexing.keys())

        # Second pass to do codegen
        for i, node in enumerate(node_schedule):
            if node is DisableReduction:
                stack.enter_context(kernel.disable_reduction())
            elif node is EnableReduction:
                stack.close()
                kernel.set_last_usage(current_reduction_nodes(node_schedule[i:]))
            else:
                with kernel.set_current_node(node):
                    # TODO - use split ranges ?
                    indexing_dtype_strength_reduction(node._body)
                    index_vars = kernel.split_and_set_ranges(node.get_ranges())
                    node.codegen(index_vars)


#####
# Kernel support for block reduction
#####

@monkey.patch(TritonKernel)
@contextlib.contextmanager
def set_current_node(self: TritonKernel, node: SchedulerNode):
    """
    + relay block_hint from scheduler to kernel
    + reset tensor_dim
    """
    with monkey.fallback(self, node):
        block_hint = getattr(node, 'block_hint', None)
        if block_hint:
            setattr(self, 'block_hint', block_hint)
            # block reduction does not do multi-lane reduction
            self.range_trees[-1].tensor_dim = None
        elif self.inside_reduction:  # default: multi-lane reduction
            # tensor_dim = itertools.count()
            for i, tree in enumerate(self.range_trees):
                if tree.tensor_dim != i:
                    tree.tensor_dim = i
        yield


@monkey.patch(TritonKernel)
def set_ranges(self: TritonKernel, *lengths) -> list[list[Expr]]:
    if blockhint:= getattr(self, 'block_hint', getattr(self.current_node, 'block_hint', None)):
        hint_offset = 0
        for i, ran9e in enumerate(lengths):
            if hint_offset >= len(blockhint):
                break
            self.range_trees[i].block_hint = blockhint[hint_offset:hint_offset + len(ran9e)]
            hint_offset += len(ran9e)
    return monkey.fallback(self, *lengths)


#####
# Range tree semantics
#####

@monkey.patch(IterationRangesRoot)
def index_sym(self: IterationRangesRoot):
    if any(tree != self and tree.prefix == self.prefix for tree in V.kernel.range_trees):
        return sympy_index_symbol(f"{self.prefix}index{self.index}")
    return sympy_index_symbol(f"{self.prefix}index")


def lookup(self: IterationRangesRoot, divisor, length, parent):
    """Not patching the original method because of incompatible signature.
    + parent setting to reflect tree structure
    """
    if V.graph.sizevars.statically_known_equals(divisor * length, self.numel):
        expr = FloorDiv(self.index_sym(), divisor)
    else:
        expr = ModularIndexing(self.index_sym(), divisor, length)

    if expr not in self.nodes:
        node = IterationRangesEntry(
            f"{self.prefix}{next(V.kernel.iter_vars_count)}",
            divisor,
            length,
            expr,
            parent,
        )
        V.kernel.range_tree_nodes[node.symbol()] = node
        self.var_list.append(node.symbol())
        self.var_ranges[node.symbol()] = length
        self.nodes[expr] = node
    return self.nodes[expr]

@monkey.patch(IterationRangesRoot)
def construct_entries(self: IterationRangesRoot, lengths: list[sympy.Expr]):
    """
    + maintain (reversed) parentship
    """
    divisor = sympy.Integer(1)
    itervars = []
    for length in reversed(lengths):
        itervars.append(lookup(self, divisor, length, itervars[-1] if itervars else self))
        divisor = divisor * length
    return list(reversed(itervars))
    # """
    # + construct entries in original order to maintain parentship
    # """
    # reversed_divisors = [sympy.Integer(1)]
    # for length in reversed(lengths[1:]):
    #     reversed_divisors.append(reversed_divisors[-1] * length)
    # itervars = []
    # for length, divisor in zip(lengths, reversed(reversed_divisors)):
    #     assert V.graph.sizevars.statically_known_multiple_of(self.numel, divisor), f"{self} cannot be split by {divisor}"
    #     itervars.append(lookup(self, divisor, length, itervars[-1] if itervars else self))
    # return itervars

@monkey.patch(TritonKernel)
def is_broadcasted(self: TritonKernel, index: sympy.Expr):
    """
    + fix use of entry.parent to entry.root
    + fix use of self.numels to dynamic calculation from self.range_trees
    """
    # Note. This may not be correct when there is indirect indexing
    if self.is_indirect_indexing(index):
        return False

    index_numels = [1] * len(self.range_trees)
    for symbol in index.free_symbols:
        if symbol not in self.range_tree_nodes:
            # Non-iterated variables, e.g. strides
            continue
        entry = self.range_tree_nodes[symbol]  # type: ignore[index]
        assert isinstance(entry.root, IterationRangesRoot)
        index_numels[entry.root.index] *= entry.length

    # If the index variables only iterate over a subset of the kernel
    # numels, then it must be broadcasted.
    simplify = V.graph.sizevars.simplify
    return any(
        simplify(idx_range) != simplify(iter_range)  # type: ignore[arg-type]
        for idx_range, iter_range in zip(index_numels, [tree.numel for tree in self.range_trees])
    )


#####
# Block tiling and indexing
#####


def _is_var_list_ordered(var_list: list[sympy.Symbol]) -> bool:
    if len(var_list) <= 1:
        return True
    expect = var_list[-1]
    for var in reversed(var_list):
        if expect != var:
            return False
        expect = V.kernel.range_tree_nodes[var].parent.symbol()
    return True

def _get_range_hierarchy(tree: IterationRangesRoot):
    """return entry list ordered from stride 1."""
    children = collections.defaultdict(list)
    for entry in tree.nodes.values():
        if entry.parent != entry.root:
            children[entry.parent].append(entry)
    leaves = {entry for entry in tree.nodes.values() if entry not in children}
    def path_to_root(n: IterationRangesEntry) -> list[IterationRangesEntry]:
        if n.parent == n.root:
            return [n]
        return path_to_root(n.parent) + [n]
    assert len(leaves) == 1, f"inconsistent hierarchy: {[path_to_root(e) for e in leaves]}"
    return path_to_root(next(iter(leaves)))


class BlockMeta:
    """Per-rangetree meta info for blocking/tiling"""
    @dataclasses.dataclass()
    class BlockedRange:
        range_tree: IterationRangesRoot
        meta: 'BlockMeta'
        numel: sympy.Symbol | sympy.Integer | Expr
        stride: Expr
        var_list: list[sympy.Symbol] = dataclasses.field(default_factory=list)
        suffix: str = ''

        def __post_init__(self):
            self.meta.ranges.append(self)
            self.pid = sympy_index_symbol(f"pid{self.suffix}")
            if self.numel.is_Integer:
                self.block = sympy.Integer(1)
                self.offset = self.pid  # pid * 1
                self.base = sympy.Integer(0)  # tl.arange(0, 1)
            else:
                self.block = sympy.Symbol(f"{self.range_tree.prefix.upper()}BLOCK{self.suffix}", integer=True, positive=True)
                self.offset = sympy_index_symbol(f"{self.range_tree.prefix}offset{self.suffix}")
                self.base = sympy_index_symbol(f"{self.range_tree.prefix}base{self.suffix}")
        
        def __hash__(self):
            return hash(tuple(self.var_list))


    def __init__(self, tree: IterationRangesRoot):
        self.range_tree = tree
        self.size: tuple[sympy.Symbol, ...]

        if not _is_var_list_ordered(tree.var_list):
            fusion_log.debug("\033[033m @bozhiyou should not reach here but in case the invariance is violated \033[0m")
            var_list = [entry.symbol() for entry in _get_range_hierarchy(tree)]
            assert set(var_list) == set(tree.var_list), f"{tree.var_list} {var_list}"
            tree.var_list = var_list
        var_list = tree.var_list

        hint = getattr(tree, 'block_hint', [])
        if 0 < len(hint) < len(var_list):
            raise NotImplementedError(f"insufficient hints {hint} {len(var_list)}")
        self.hint = tuple(reversed(hint[-len(var_list):]))


        self.block_group = self.group(self.hint)  # dim groups that require blocking

        suffix = itertools.count()
        block_suffix = collections.defaultdict(lambda: f"{next(suffix)}")
        self.block_suffix = lambda b: block_suffix.get(b, '')

        self.ranges: list[BlockMeta.BlockedRange] = []
        processed_label = set()
        if len(set(self.hint)) <= 1 and None not in self.hint:
            # no hint or same blocking for all
            block = self.BlockedRange(tree, self,
                                      sympy.Symbol(f"{tree.prefix}numbl", integer=True, positive=True),  # TODO @bozhiyou tree.numel can be inlined here
                                      sympy.Integer(1), var_list)
            self.var_to_block = lambda _: block
        else:
            var_to_block = {}
            size = []
            last = self.hint[0]
            for var, b in zip(var_list, self.hint):
                if b is None:
                    block = self.BlockedRange(tree, self, tree.var_ranges[var], sympy_product(size), [var])
                    var_to_block[var] = block
                    size.append(tree.var_ranges[var])
                    continue
                if b in processed_label:
                    assert b == last, f"block hint must be consecutive labels {hint}; non-consecutive {b}"
                    block = self.ranges[-1]
                    block.var_list.append(var)
                    var_to_block[var] = block
                    continue
                numel = sympy.Symbol(f"{tree.prefix}numbl{block_suffix[b]}", integer=True, positive=True)
                block = self.BlockedRange(tree, self, numel, sympy_product(size), [var], block_suffix[b])
                var_to_block[var] = block
                size.append(numel)
                processed_label.add(b)
            self.var_to_block = lambda v: var_to_block[v]
            self.size = tuple(size)

        self._var_to_hint = {var: b for var, b in zip(var_list, self.hint)}
        self.var_to_suffix = lambda var: self.block_suffix(self._var_to_hint.get(var, None))


    @staticmethod
    def group(hint: Sequence) -> Sequence:
        if not hint:
            return []
        grouped = []
        last = hint[0]
        for h in hint:
            if h in grouped:
                continue
            if h is not None:
                grouped.append(h)
            last = h
        return grouped
    
    
    def triton_tensor_ndim(self):
        """number of blocked dimensions"""
        if self.range_tree.tensor_dim is None:
            return 0
        return sum(1 for r in self.ranges if r.block != 1)



@monkey.patch(TritonKernel)
def finalize_indexing(self: TritonKernel, indices: Sequence[sympy.Expr]):
    """
    Hook called right before codegen with every index that will be
    used in the fused kernel.
    """
    for tree in self.range_trees:
        setattr(tree, 'block_meta', BlockMeta(tree))
        

#####
# Codegen
#####


@monkey.patch(TritonKernel)
def codegen_range_tree(self: TritonKernel) -> None:
    f"""
    {TritonKernel} does codegen for range trees right after they are initialized
    which assumes linear blocking for each range tree (thus single block size).

    Delay this until range tree hierarchy is known.
    """
    codegen_range_tree = monkey.get_fallback_hook()
    @monkey.patch(self)
    def finalize_indexing(indices: Sequence[sympy.Expr]):
        # NOTE @bozhiyou by this point, `indices` should include blocking info
        monkey.fallback(indices)
        codegen_range_tree(self)



@monkey.patch(TritonKernel)
def dense_size_list(self: TritonKernel) -> list[str]:
    sizes = []
    for tree in self.range_trees:
        if tree.tensor_dim is None:
            continue
        block_meta = getattr(tree, 'block_meta', None)
        if block_meta is None and (tree.prefix != 'r' or self.inside_reduction):
            prefix, suffix = tree.name.split('index')
            sizes.append(f"{tree.prefix.upper()}BLOCK{suffix}")
            continue
        for r in block_meta.ranges:
            if not r.block.is_symbol:
                continue
            sizes.append(f"{r.block}")
    return sizes

@monkey.patch(TritonKernel)
def iteration_ranges_ranges_code(self: TritonKernel, entry, block_size=''):
    assert entry.tensor_dim is not None or entry.prefix == 'r'
    # size = self.indexing_size_str(entry.tensor_dim)
    index_dtype = self.index_dtype
    convert = f".to({index_dtype})" if index_dtype != "tl.int32" else ""
    block_size = block_size or f"{entry.prefix.upper()}BLOCK{entry.name.split('index')[-1]}"
    # return f"tl.arange(0, {block_size}){size}{convert}"
    return f"tl.arange(0, {block_size}){convert}"


def add_constexpr(self: TritonKernel, key, value) -> None:
    constexpr = getattr(self, 'constexpr', {})
    if not constexpr:
        setattr(self, 'constexpr', {key: value})
        return
    constexpr[key] = value


def iteration_ranges_pid(entry: IterationRangesRoot):
    self: TritonKernel = entry.kernel
    grid_ndim = sum(1 for tree in self.range_trees if tree.grid_dim is not None)
    return sympy_index_symbol(f"{entry.prefix if grid_ndim > 1 else ''}pid")

@monkey.patch(TritonKernel)
def iteration_ranges_codegen_header(
    self: TritonKernel,
    entry: IterationRangesRoot,
    code: IndentedBuffer,
) -> None:
    """
    + handle block_hint
    """
    x = entry.prefix
    *_, suffix = entry.name.split('index')
    if entry.is_loop:
        code.writeline(f"{entry.name} = {x}offset{suffix} + {x}base{suffix}")
    elif entry.grid_dim is None:
        # no need to "{x}offset = "
        code.writeline(f"{entry.name} = {self.iteration_ranges_ranges_code(entry)}")
        code.writeline(f"{x}offset = 0")
    else:
        pid = iteration_ranges_pid(entry)
        code.writeline(f"{pid} = {self.iteration_ranges_get_pid(entry)}")
        block_meta: BlockMeta = getattr(entry, 'block_meta')
        block_hint = getattr(entry, 'block_hint', [])
        if len(set(block_hint)) <= 1:  # all indexing share same pid
            if None in block_hint:  # no blocking
                code.writeline(f"{entry.name} = {pid}")
            else:  # default 1D blocking
                xblock = f"{x.upper()}BLOCK"
                xoffset = f"{x}offset"
                code.writeline(f"{xoffset} = {pid} * {xblock}")
                if entry.tensor_dim is not None:
                    xbase = f"{self.iteration_ranges_ranges_code(entry, block_size=xblock)}"
                    code.writeline(f"{entry.name} = {xoffset} + {xbase}")
                else:
                    code.writeline(f"{entry.name} = {self.iteration_ranges_scalar_code(entry, xoffset)}")
            for expr, node in entry.nodes.items():
                code.writeline(f"{node.name} = {self.kexpr(self.rename_indexing(expr))}")
        else:  # pid partitioning
            for i, ran9e in enumerate(reversed(block_meta.ranges)):
                if ran9e == block_meta.ranges[1] and (
                    len(block_meta.ranges[0].var_list) == len(block_meta.ranges[1].var_list) == 1 and block_meta.ranges[0].block != 1 and block_meta.ranges[1].block != 1
                ):  # TODO @bozhiyou add a flag to enable this L2 optimized scheme
                    # https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations
                    group_m = "GROUP_SIZE"
                    add_constexpr(self, group_m, 8)  # TODO @bozhiyou autotune this param
                    range1, range0 = ran9e, block_meta.ranges[0]
                    pid1, pid0 = range1.pid, range0.pid
                    code.writelines(['',
                        f"{pid} = {pid} % ({sympy_product(block_meta.size[:2])})",
                        f"width = {group_m} * {block_meta.size[0]}",
                        f"group_id = {pid} // width",
                        f"group_size = min({block_meta.size[1]} - group_id * {group_m}, {group_m})",
                        f"{pid0} = ({pid} % width) // (group_size)",
                        f"{pid1} = group_id * {group_m} + ({pid} % group_size)",
                    '',])
                    for ran9e in [range1, range0]:
                        code.writeline(f"{ran9e.offset} = {self.kexpr(ran9e.pid * ran9e.block)}",)
                        var = ran9e.var_list[0]
                        if entry.tensor_dim is not None:
                            xbase = f"{self.iteration_ranges_ranges_code(entry, block_size=ran9e.block)}"
                            code.writeline(f"{var} = {ran9e.offset} + {xbase}")
                        else:
                            code.writeline(f"{var} = {self.iteration_ranges_scalar_code(entry, ran9e.offset)}")
                    break

                if ran9e.block == 1 and len(ran9e.var_list) == 1:
                    code.writeline(f"{ran9e.var_list[0]} = {self.kexpr(ModularIndexing(pid, ran9e.stride, ran9e.numel))}")
                else:
                    code.writeline(f"{ran9e.pid} = {self.kexpr(ModularIndexing(pid, ran9e.stride, ran9e.numel))}")
                    # The following capping is not necessary since `x % (a*b) // a % b` = `x // a % b`
                    # code.writeline(f"{pid} = {self.kexpr(ModularIndexing(pid, 1, range.stride))}")
                    code.writeline(f"{ran9e.offset} = {self.kexpr(ran9e.pid * ran9e.block)}",)
            
                    # lifted var def
                    for var in ran9e.var_list:
                        if entry.tensor_dim is not None:
                            xbase = f"{self.iteration_ranges_ranges_code(entry, block_size=ran9e.block)}"
                            code.writeline(f"{var} = {ran9e.offset} + {xbase}")
                        else:
                            code.writeline(f"{var} = {self.iteration_ranges_scalar_code(entry, ran9e.offset)}")

    if self._has_constant_mask(entry):
        sizes = self.dense_size_str()
        code.writeline(f"{x}mask{suffix} = tl.full({sizes}, True, tl.int1)")
    else:
        # per dim/var mask
        if len(entry.var_list) > 1:
            code.writelines([
                f"{x}mask{suffix}{var.name[1:]} = {var} < {entry.var_ranges[var]}"
                for var in entry.var_list
            ])
        else:  # only one var, no need of suffix
            code.writeline(f"{x}mask{suffix} = {entry.name} < {entry.numel}")



@monkey.patch(TritonKernel)
def triton_tensor_ndim(self: TritonKernel) -> int:
    """
    + poll tree meta for one-to-many tree-to-tensor_dim mapping
    """
    ndim = 0
    for tree in self.range_trees:
        if tree.tensor_dim is None:
            continue
        if block_meta:= getattr(tree, 'block_meta', None):
            ndim += block_meta.triton_tensor_ndim()
            continue
        ndim += 1
    return ndim


# @monkey.patch(TritonKernel)
# def reduction_resize(self: TritonKernel, value):
#     """resize final reduction result.
#     - dimmension expansion processed when store
#     """
#     ndims = self.triton_tensor_ndim()
#     if ndims == 1:
#         return f"triton_helpers.promote_to_tensor({value})"

#     return f"{value}"
#     # sizes = [":"] * ndims
#     # sizes[-1] = "None"
#     # return f"{value}[{', '.join(sizes)}]"


def index_var_to_blocked_dim(self: TritonKernel, index: Expr):
    if isinstance(index, sympy.Symbol) or not index.args:  # singleton
        index_vars = (index,)
    else:
        assert all(len(arg.free_symbols) <= 1 for arg in index.args), f"{[arg.free_symbols for arg in index.args]}"
        index_vars = tuple(var for arg in index.args for var in arg.free_symbols)

    dim = itertools.count()
    var_to_blocked_dim = dict[sympy.Symbol, int]()
    if self.range_trees[-1].tensor_dim is not None:
        # reserve reduction dimension
        block_meta = getattr(self.range_trees[-1], 'block_meta', None)
        if not block_meta:
            # single block size by default
            next(dim)  # 0
            for var in self.range_trees[-1].var_list:
                var_to_blocked_dim[var] = 0
        else:
            blocked_range = block_meta.ranges[-1]
            if blocked_range.block != 1:
                next(dim)  # 0
                for var in blocked_range.var_list:
                    var_to_blocked_dim[var] = 0

    for i, var in enumerate(index_vars):  # `args` is ordered, `free_symbols` is not
        if var in var_to_blocked_dim:
            continue
        tree = self.range_tree_nodes[var].root
        block_meta = getattr(tree, 'block_meta', None)
        if not block_meta:
            # single block size by default
            d = next(dim)
            for var in tree.var_list:
                var_to_blocked_dim[var] = d
        else:
            ran9e = block_meta.var_to_block(var)
            if ran9e.block != 1:
                d = next(dim)
                for var in ran9e.var_list:
                    var_to_blocked_dim[var] = d
    return var_to_blocked_dim


@monkey.patch(TritonKernel)
def indexing(
    self: TritonKernel,
    index: sympy.Expr,
    *,
    copy_shape=None,
    dense_indexing=False,
    override_mask: str|None =None,
    block_ptr=False,
):
    """
    + add size expansion to masks
    """
    if override_mask is None:
        var_to_blocked_dim = index_var_to_blocked_dim(self, index)
        ndim = len(set(var_to_blocked_dim.values()))

        var_ranges = self.var_ranges()
        if ndim <= 1:
            override_mask = ' & '.join(f"({var} < {var_ranges[var]})" for arg in index.args for var in arg.free_symbols)
        else:
            masks = []
            for arg in (index.args or (index,)):  # `args` is ordered, `free_symbols` is not            
                assert len(arg.free_symbols) == 1
                var = next(iter(arg.free_symbols))
                if var not in var_to_blocked_dim:
                    masks.append(f"({var} < {var_ranges[var]})")
                    continue
                i = var_to_blocked_dim[var]
                dims = ["None"] * ndim
                dims[-i-1] = ':'
                masks.append(f"({var} < {var_ranges[var]})[{', '.join(dims)}]")
            override_mask = ' & '.join(masks)
    return monkey.fallback(self, index, copy_shape=copy_shape, dense_indexing=dense_indexing, override_mask=override_mask, block_ptr=block_ptr)


@monkey.patch(TritonKernel)
def index_to_str(self: TritonKernel, index: sympy.Expr) -> str:
    """
    Convert an index expr to a string that can be used in output code.
    e.g. a sympy expression "s2" may actually appear as "ks1" in the generated kernel.

    Index expressions often need to be passed in as arguments to the triton kernel.
    Rename_indexing and codegen_indexing keep track of the needed indices and add
    new parameters to the function signature.
    """
    index_str: str = monkey.fallback(self, index)
    if isinstance(index, list):
        return index_str

    var_to_blocked_dim = index_var_to_blocked_dim(self, index)
    ndim = len(set(var_to_blocked_dim.values()))

    if ndim <= 1:
        return index_str
    args = []
    for arg in (index.args or (index,)):
        assert len(arg.free_symbols) == 1
        var = next(iter(arg.free_symbols))
        if var not in var_to_blocked_dim:
            args.append(f"{arg}")
            continue
        i = var_to_blocked_dim[var]
        dims = ["None"] * ndim
        dims[-i-1] = ':'
        args.append(f"({arg})[{', '.join(dims)}]")
    index_str = ' + '.join(args)
    return index_str


@monkey.patch(TritonKernel)
def combine_contiguous_dims(self, index: sympy.Expr, tree: IterationRangesRoot):
    """Disable combining in ND tiling"""
    return index


@monkey.patch(TritonKernel)
def codegen_iteration_ranges_entry(self: TritonKernel, entry: IterationRangesEntry):
    """Entries are used as bases."""
    line = f"{entry.name} = {self.kexpr(self.rename_indexing(entry.expr))}"
    if entry.root.is_loop:
        self.indexing_code.writeline(line)
        return
    # non-reduction indexing lifted outside loop
    self.body.writeline(f"# {line}")
    return


@monkey.patch(TritonKernel)
def codegen_static_numels(self, code) -> None:
    """Insert constexpr at the start of kernel."""
    monkey.fallback(self, code)
    for key, val in getattr(self, 'constexpr', {}).items():
        code.writeline(f"{key}: tl.constexpr = {val}")


@monkey.patch(TritonKernel)
def _get_heuristic(self: TritonKernel):
    if getattr(self, 'block_hint', None):
        assert self.inside_reduction
        return "blockreduction"
    if self.persistent_reduction:
        assert self.inside_reduction
        return "persistent_reduction"  # TODO @bozhiyou blockwise persistent_reduction?
    elif self.inside_reduction:
        return "reduction"
    return "pointwise"


@monkey.patch(TritonKernel)
def codegen_kernel(self: TritonKernel, name=None) -> str:
    """
    - range without tensor_dim may still need BLOCK, e.g. if is_loop
    """
    from torch._inductor import config
    from torch._inductor.codegen.common import WorkspaceArg
    from torch._inductor.codegen.triton import gen_common_triton_imports, DeviceProperties, Placeholder
    from torch._inductor.codegen.triton_utils import signature_to_meta, signature_of, config_of
    from torch._inductor.runtime.runtime_utils import next_power_of_2
    from typing import cast
    code = IndentedBuffer()

    size_hints = []
    for numel in self.numels:
        numel_hint = V.graph.sizevars.symbolic_hint(numel)
        if not isinstance(numel_hint, (int, sympy.Integer)):
            # This default heuristic hint was picked carefully: it is
            # large, to ensure that we don't shrink the block size (since
            # if you don't have many elements, it'd be wasteful to pick a
            # large block size).  Since we don't know how many elements we
            # might have, we should be OK with some inefficiency to make
            # sure we handle the large case well.  8192 is the largest
            # block size we support, so we pick that.
            #
            # If we have a better hint for unbacked SymInts (e.g., because
            # a user told us, or we are tracking upper bounds) we could
            # use that here.
            size_hint = 8192
        else:
            size_hint = next_power_of_2(int(numel_hint))
        size_hints.append(size_hint)

    if not self.inside_reduction:
        size_hints.pop()

    heuristics = self._get_heuristic()

    if name is None:
        code.splice(gen_common_triton_imports())

        if config.benchmark_kernel:
            code.splice(self.imports_for_benchmark_kernel())

    argdefs, _, signature, _ = self.args.python_argdefs()
    # maps actual expression to SizeArg if it is in sizevars replacements
    for i, arg in enumerate(signature):
        if isinstance(arg, SizeArg):
            # mypy is unhappy about the sympy.Expr
            # type for the key of the dict below
            symbol = cast(sympy.Symbol, arg.expr)
            if symbol in V.graph.sizevars.inv_precomputed_replacements:
                signature[i] = SizeArg(
                    arg.name, V.graph.sizevars.inv_precomputed_replacements[symbol]
                )

    mutated_args: OrderedSet[str] = OrderedSet()
    for mutation in self.mutations:
        if mutation in self.args.input_buffers:
            mutated_args.add(self.args.input_buffers[mutation])
        if (
            mutation in self.args.inplace_buffers
            and mutation not in V.graph.removed_buffers
            and mutation not in self.removed_buffers
        ):
            mutated_args.add(self.args.inplace_buffers[mutation].inner_name)
        if mutation in self.args.output_buffers:
            mutated_args.add(self.args.output_buffers[mutation])

    # workspace arguments are mutated, but are not marked as mutations in self.mutations
    # because their buffers are added during codegen, and aren't tracked during
    # lowering/scheduling. So we add them as mutated_args explicitly below.
    #
    # In the logic below, we only mark the workspaces a mutated if they are marked with
    # zero_fill: that's because, if we don't expect the buffer to be pre-filled with
    # zeros, then, although we still mutate the data, we don't care about those
    # mutations because we don't make any assumptions about the contents of the
    # workspace buffer.
    for argname, arg in zip(argdefs, signature):
        if isinstance(arg, WorkspaceArg) and arg.zero_fill:
            mutated_args.add(argname)

    mutated_args = sorted(mutated_args)

    triton_meta_signature = signature_to_meta(
        signature, size_dtype=self.index_dtype
    )
    triton_meta = {
        "signature": triton_meta_signature,
        "device": DeviceProperties.create(
            V.graph.scheduler.get_current_device_or_throw()
        ),
        "constants": {},
    }

    inductor_meta = {
        "autotune_hints": set(self.autotune_hints),
        "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
        "mutated_arg_names": mutated_args,
        "no_x_dim": self.no_x_dim,
        "num_load": self.num_load,
        "num_reduction": self.num_reduction,
        **self.inductor_meta_common(),
    }

    num_gb = None
    if config.benchmark_kernel or config.profile_bandwidth:
        num_gb = self.estimate_kernel_num_bytes() / 1e9
        inductor_meta["kernel_num_gb"] = num_gb

    for tree in self.active_range_trees():
        sizearg = SizeArg(f"{tree.prefix}numel", tree.numel)
        signature.append(sizearg)
        triton_meta_signature[len(argdefs)] = signature_of(
            sizearg, size_dtype=self.index_dtype
        )
        argdefs.append(f"{tree.prefix}numel")
        # constexpr version causes issues, see
        # https://github.com/pytorch/torchdynamo/pull/1362
        # triton_meta["constants"][len(argdefs)] = V.graph.sizevars.size_hint(
        #     tree.numel
        # )
        # argdefs.append(f"{tree.prefix}numel: tl.constexpr")
    triton_meta["configs"] = [config_of(signature)]

    # Triton compiler includes equal_to_1 args into constants even
    # when they are not constexpr. otherwise there may be a segfault
    # during launching the Inductor-compiled Triton kernel.
    # https://github.com/pytorch/pytorch/issues/120478#issuecomment-1962822307
    # https://github.com/openai/triton/blob/231efe9ed2d200be0f69a07c298e4342b08efe3d/python/triton/runtime/jit.py#L384
    for arg_num in triton_meta["configs"][0].equal_to_1:  # type: ignore[index]
        triton_meta["constants"][arg_num] = 1  # type: ignore[index]

    self.triton_meta = triton_meta

    inductor_meta['block_hints'] = {}
    for tree in self.range_trees:
        if tree.prefix == "r" and (self.persistent_reduction or not self.inside_reduction):
            # RBLOCK for persistent_reduction is defined in codegen_static_numels
            continue
        # if tree.tensor_dim is None:
        #     continue
        prefix, suffix = tree.name.split('index')
        assert prefix == tree.prefix
        numel = f"{tree.prefix}numel{suffix}"
        inductor_meta['block_hints'][numel] = tree.numel
        if 'block' in heuristics:
            if block_meta:= getattr(tree, 'block_meta', None):
                for ran9e in block_meta.ranges:
                    if not ran9e.block.is_symbol:
                        inductor_meta['block_hints'][numel] = ceildiv(inductor_meta['block_hints'][numel], ran9e.numel)
                        continue
                    argdefs.append(f"{ran9e.block}: tl.constexpr")
                    if not tree.is_loop:
                        argdefs.append(f"{ran9e.numel}: tl.constexpr")
                        inductor_meta['block_hints'][f"{ran9e.numel}"] = sympy_product(tree.var_ranges[var] for var in ran9e.var_list)
        # NOTE @bozhiyou still keep XBLOCK as grid scalar
        block = f"{tree.prefix.upper()}BLOCK{suffix}: tl.constexpr"
        if block not in argdefs:
            argdefs.append(block)
        
    # for kernel fusion
    for arg in getattr(self.args, 'constexprs', set()):
        inductor_meta['block_hints'][arg] = None
        arg = f"{arg}: tl.constexpr"
        if arg not in argdefs:
            argdefs.append(arg)

    self.codegen_body()

    for helper in self.helper_functions:
        code.writeline("")
        code.splice(helper)

    if self.inside_reduction:
        reduction_hint = self.reduction_hint
        heuristics_line = f"""
            @triton_heuristics.{heuristics}(
                size_hints={size_hints!r},
                reduction_hint={reduction_hint},
                filename=__file__,
                triton_meta={triton_meta!r},
                inductor_meta={inductor_meta!r}
            )
            @triton.jit
        """
    else:
        tile_hint = ""
        if len(size_hints) == 2:
            if len(signature) == 4:  # input, output and 2 args
                tile_hint = "tile_hint=TileHint.SQUARE,"
            else:
                tile_hint = "tile_hint=TileHint.DEFAULT,"
        heuristics_line = f"""
            @triton_heuristics.{heuristics}(
                size_hints={size_hints!r}, {tile_hint}
                filename=__file__,
                triton_meta={triton_meta!r},
                inductor_meta={inductor_meta!r},
                min_elem_per_thread={self.min_elem_per_thread}
            )
            @triton.jit
        """
    code.splice(heuristics_line)
    code.writeline(
        f"def {name or str(Placeholder.KERNEL_NAME)}({', '.join(argdefs)}):"
    )
    with code.indent():
        self.codegen_static_numels(code)
        for old, new in self.args.aliases():
            code.writeline(f"{old} = {new}")
        code.splice(self.body)

    if config.benchmark_kernel:
        code.splice(self.codegen_kernel_benchmark(num_gb))

    return code.getvalue()


# @monkey.patch(TritonKernel)
# def add_numel_to_call_args_and_grid(self: TritonKernel, name, call_args, arg_types, grid):
#     meta = V.graph.wrapper_code.add_meta_once(self.meta)
#     for tree in self.range_trees:
#         if isinstance(tree.numel, (sympy.Integer, sympy.Symbol)):
#             expr = tree.numel
#         else:
#             expr = V.graph.wrapper_code.generate_numel_expr(name, tree)

#         if tree.prefix != "r" or self.inside_reduction:
#             call_args.append(expr)
#             arg_types.append(type(expr))
#         if tree.grid_dim is not None:
#             grid.append(expr)





#####
# Runtime autotune
#####
import torch._inductor.runtime.triton_heuristics
from torch._inductor.runtime.triton_heuristics import cached_autotune, get_max_y_grid
from torch._inductor.runtime.hints import HeuristicType

import triton

@monkey.patch(torch._inductor.runtime.triton_heuristics)
def grid(*numels):
    """Helper function to compute triton grids"""
    if len(numels) == 1:
        xnumel, ynumel, znumel = numels[0], None, None
    elif len(numels) == 2:
        xnumel, ynumel, znumel = numels[1], numels[0], None
    elif len(numels) == 3:
        xnumel, ynumel, znumel = numels[2], numels[1], numels[0]
    else:
        raise AssertionError(f"invalid size for numels {len(numels)}")

    def get_grid_dim(numel, block):
        if numel is None:
            return 1
        if block is None:
            return numel
        return ceildiv(numel, block)

    def grid_fn(meta):
        x_grid = get_grid_dim(xnumel, meta.get("XBLOCK", 1))
        for k, v in meta.items():
            if k.startswith('xnumbl'):
                x_grid *= v
        y_grid = get_grid_dim(ynumel, meta.get("YBLOCK", None))

        max_y_grid = get_max_y_grid()
        if znumel is None:
            div = ceildiv(y_grid, max_y_grid)
            y_grid = ceildiv(y_grid, div)
            z_grid = div
        else:
            z_grid = get_grid_dim(znumel, meta.get("ZBLOCK", None))
            torch._check(
                y_grid <= max_y_grid,
                lambda: f"Generated y grid beyond 2^16 ({y_grid}) not supported with z dimension present. File issue",
            )

        return (
            x_grid,
            y_grid,
            z_grid,
        )

    setattr(grid_fn, "grid_fn_str", f"grid{numels}")  # noqa: B010

    return grid_fn



def blockreduction_configs(
    *,
    size_hints=None,
    inductor_meta={},
):
    f"""Config space from {torch._inductor.kernel.flex_attention} and {torch._inductor.kernel.mm_common}."""
    from torch._inductor.kernel.flex_attention import _get_default_config_fwd
    # (BLOCK_M, BLOCK_N, num_warps, num_stages)
    configs: list[tuple[int, int, int, int]] = [(64, 32, 4, 3)]  # TODO @bozhiyou default to max or 1?
    # configs.append(_get_default_config_fwd(query))
    if torch._inductor.config.max_autotune:
        configs += [
            (128, 64, 4, 3),
            (128, 128, 4, 3),
            (128, 128, 8, 2),
            (64, 128, 4, 3),
            (64, 64, 4, 3),
        ]
    
    if 'block_hints' in inductor_meta:
        block_hints = inductor_meta['block_hints']
        def block_config(xblock, rblock):
            block = {'x': xblock, 'r': rblock}
            c = {}
            for k, v in block_hints.items():
                if 'numbl' in k:
                    prefix, suffix = k.split('numbl')
                    c[k] = ceildiv(v, block[prefix])
                    c[f"{prefix.upper()}BLOCK{suffix}"] = block[prefix]
                if 'BLOCK' in k:
                    c[k] = block[k[0].lower()]
            return c

        return [
            triton.Config({
                **block_config(XBLOCK, RBLOCK),
                "XBLOCK": inductor_meta['block_hints']["xnumel"],
                "RBLOCK": RBLOCK,
            }, num_stages=num_stages, num_warps=num_warps)
            for (XBLOCK, RBLOCK, num_warps, num_stages) in configs
        ]

    return [
        triton.Config({
            "XBLOCK": XBLOCK,
            "RBLOCK": RBLOCK,
        }, num_stages=num_stages, num_warps=num_warps)
        for (XBLOCK, RBLOCK, num_warps, num_stages) in configs
    ]


@monkey.patch(torch._inductor.runtime.triton_heuristics)
def blockreduction(
    size_hints=None,
    reduction_hint=False,
    filename=None,
    inductor_meta={},
    triton_meta={},
):
    """args to @triton.heuristics()"""
    inductor_meta["reduction_hint"] = reduction_hint
    # reduction requires: assert triton_meta is not None

    if size_hints:
        if inductor_meta.get("no_x_dim"):
            size_hints = [1, *size_hints[1:]]
        rnumel = size_hints[-1]
        if len(size_hints) != 2:
            raise NotImplementedError(f"size_hints: {size_hints}")

    configs = blockreduction_configs(size_hints=size_hints, inductor_meta=inductor_meta)

    # def template(num_stages, num_warps, triton_meta, filename=None, inductor_meta=None):
    return cached_autotune(
        size_hints,
        configs=configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.TEMPLATE,
        filename=filename,
    )

_kernel_category_choices.append('blockreduction')









###
# debug handles

@monkey.property(IterationRangesRoot)
def tensor_dim(self):
    return self._tensor_dim

@tensor_dim.setter
def tensor_dim(self, value):
    self._tensor_dim = value


from torch._inductor.loop_body import LoopBody
@monkey.property(LoopBody)
def indexing_exprs(self):
    return self._indexing_exprs

@indexing_exprs.setter
def indexing_exprs(self, value):
    self._indexing_exprs = value



@monkey.property(TritonKernel)
def args(self):
    return self._args

@args.setter
def args(self, value):
    self._args = value

@monkey.property(TritonKernel)
def numels(self):
    return self._numel

@numels.setter
def numels(self, value):
    self._numel = value


from torch._inductor.bounds import BoundVars
@monkey.property(BoundVars)
def replacement_vals(self):
    return self._replacement_vals

@replacement_vals.setter
def replacement_vals(self, value):
    self._replacement_vals = value


@monkey.property(BoundVars)
def _bounds(self):
    return self._replacement_vals

@_bounds.setter
def _bounds(self, value):
    self._replacement_vals = value

from torch._inductor.sizevars import SizeVarAllocator
@monkey.property(SizeVarAllocator)
def inv_precomputed_replacements(self):
    return self._inv_precomputed_replacements

@inv_precomputed_replacements.setter
def inv_precomputed_replacements(self, value):
    self._inv_precomputed_replacements = value

from torch._inductor.codegen.wrapper import WrapperCodeGen
@monkey.property(WrapperCodeGen)
def computed_sizes(self):
    return self._computed_sizes

@computed_sizes.setter
def computed_sizes(self, value):
    self._computed_sizes = value

