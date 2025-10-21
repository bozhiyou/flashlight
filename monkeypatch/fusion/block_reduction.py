"""
New ReductionExt IR node for bmm.

ReductionExt IR: the ReductionExt class is a new type of reduction. It represents a reduction that is fully completed within a single thread block, fit for the inner dot-product accumulation along the K-dimension of a matrix multiplication. This contrasts with Inductor's default ir.Reduction, which often implies a parallel reduction across threads that requires a second-stage reduction.

BMM Lowering: Instead of using a fixed Triton template for aten.bmm, lower it to a ReductionExt where the inner function is a dot product. This makes bmm a first-class citizen in the fusion system.
"""
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
from torch._inductor.kernel.bmm import tuned_bmm
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.loop_body import LoopBody
from torch._inductor.codegen.common import SizeArg
from torch._inductor.codegen.triton import TritonScheduling, TritonKernel, TritonKernelOverrides
from torch._inductor.codegen.simd import IterationRangesRoot, IterationRangesEntry, EnableReduction, DisableReduction
from torch._inductor.optimize_indexing import indexing_dtype_strength_reduction
from torch._inductor.utils import ceildiv, sympy_index_symbol, sympy_product, IndentedBuffer, sympy_subs, VarRanges
from torch._inductor.sizevars import SimplifyIndexing


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

from ._reduction import ReductionExt

@monkey.patch(TritonKernelOverrides)
@staticmethod
def dot(a, b, **kwargs):
    return f"tl.dot({', '.join([str(a), str(b)] + [f'{k}={repr(v)}' for k,v in kwargs.items()])})"


@monkey.patch(TritonKernelOverrides)
@staticmethod
def exp(x):
    return f"tl_math.exp2(({x}) * 1.44269504)"

@monkey.patch(TritonKernelOverrides)
@staticmethod
def maximum(a, b):
    return f"tl.maximum({a}, {b})"


def _to_dtype(x: ir.TensorBox, dtype: torch.dtype, copy=False, use_compute_types=True):
    """Adapted from L.to_dtype.
    + use_compute_types
        True: triton_compute_type
        False: triton_store_type
    """
    src_dtype = x.get_dtype()
    if src_dtype == dtype:
        return L.clone(x) if copy else x

    def _to_dtype(x):
        return ops.to_dtype(x, dtype, src_dtype=src_dtype, use_compute_types=use_compute_types)

    return L.make_pointwise(_to_dtype, override_return_dtype=dtype)(x)

@monkey.patch(L)
def transform_args(args, broadcast, type_promotion_kind, convert_input_to_bool):
    if V.graph.current_node.target in [getattr(op, ov) for op in (
            torch.ops.aten.bmm,
            # torch.ops.aten.mm,
            # torch.ops.aten.matmul,
        ) for ov in op.overloads()]:
        a, b = args[:2]
        assert isinstance(a, ir.TensorBox) and isinstance(b, ir.TensorBox)
        # For mixed precision, disable promotion and downcast to bf16
        if {a.get_dtype(), b.get_dtype()} == {torch.bfloat16, torch.float32}:
            type_promotion_kind = None  # disable/override
            args = (
                [a, _to_dtype(b, torch.bfloat16, use_compute_types=False)]
                if a.get_dtype() == torch.bfloat16
                else [_to_dtype(a, torch.bfloat16, use_compute_types=False), b]
            ) + args[2:]
    return monkey.fallback(args, broadcast, type_promotion_kind, convert_input_to_bool)


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
        m1 = mat1_loader(mat1_index)
        m2 = mat2_loader(mat2_index)
        # return m1 * m2
        return ops.dot(m1, m2, input_precision='ieee')
        # return ops.dot(
        #     # operations in mat1_loader might have mat1 dtype promoted
        #     ops.to_dtype(m1, mat1.dtype, use_compute_types=False),
        #     ops.to_dtype(m2, mat2.dtype, use_compute_types=False),
        # )

    return dict(
        input_node=(mat1, mat2),
        device=layout.device,
        dst_dtype=layout.dtype,  # TODO @bozhiyou match torch._inductor.kernel.mm_common.acc_type
        src_dtype=layout.dtype,
        inner_fn=loader,
        ranges=layout.size,
        reduction_ranges=[k],
    )


def mm_args(
    mat1: ir.TensorBox,
    mat2: ir.TensorBox,
    *others: tuple[ir.TensorBox],
    layout=None,
    out_dtype=None,
    use_4x2_dim=False,
    mat2_transposed=False,
):
    f"""Adapted from torch._inductor.kernel::bmm.mm_args
    - remove input realization
    """
    # mat1, mat2 = realize_inputs(mat1, mat2)
    *b1, m, k1 = mat1.get_size()
    if mat2_transposed:
        *b2, n, k2 = mat2.get_size()
    else:
        *b2, k2, n = mat2.get_size()
    b = [V.graph.sizevars.guard_equals(a, b) for a, b in zip(b1, b2)]
    if use_4x2_dim:
        k2 = k2 * 2
    k = V.graph.sizevars.guard_equals(k1, k2)
    if layout is None:
        from torch._inductor.ir import FixedLayout

        if out_dtype is None:
            out_dtype = mat1.get_dtype()
            # tl.dot: bf16 x bf16 -> fp32
            if {mat1.get_dtype(), mat2.get_dtype()} == {torch.bfloat16}:
                out_dtype = torch.float32

        layout = FixedLayout(
            mat1.get_device(),
            out_dtype,
            [*b, m, n],
        )
    else:
        assert out_dtype is None, "out_dtype is ignored if layout is specified."

    # from torch._inductor.lowering import expand
    # others = [realize_inputs(expand(x, layout.size)) for x in others]

    return [m, n, k, layout, mat1, mat2, *others]


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
    result = ReductionExt.create(
        reduction_type='sum',
        reduction_hint = ir.ReductionHint.INNER,
        **kwargs,
    )
    if isinstance(
        result.data.data, ir.Reduction
    ):  # Only realize if reduction isn't unrolled
        result.realize()
    # The `block_hint` attribute is a static heuristic that guides the tiling strategy for the kernel.
    # It's currently implemented as a list of labels, one for each dimension of the output tensor.
    # - `None`: The dimension is not tiled and is iterated over normally (e.g., batch dimensions).
    # - `int`: Dimensions with the same integer label are fused into a single logical tiled dimension.
    #   Dimensions with different integer labels are treated as separate, orthogonal tiled dimensions.
    # For `bmm`, we assign `[None, ..., 1, 0]` to create a 2D tiling scheme.
    result.data.data.block_hint = [None for _ in range(len(result.data.get_size()) - 2)] + [1, 0]
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
    # torch.fx.graph.inplace_methods
}








######
# Scheduler support for block reduction
######

@monkey.patch(SchedulerNode)
def _init_from_node(self: SchedulerNode, node: ir.Operation) -> None:
    """relay block_hint from inductor ir to scheduler"""
    self.block_hint = getattr(node, 'block_hint', None)
    return monkey.fallback(self, node)


def _simplify_modular_indexing(indexing):
    def _visitor(*args):
        x = args[0]
        if (tree_node := V.kernel.range_tree_nodes.get(x)) is None:
            return ModularIndexing(*args)
        new_index = sympy_subs(ModularIndexing(*args), {x: tree_node.expr})
        # new_index = V.graph.sizevars.combine_modular_indexing_pairs(new_index)
        # the index now contains xindex/etc, which is nonstandard, fix it up
        return sympy_subs(
            new_index,
            {
                new_index: tree_node.root.lookup(
                    new_index.args[1], new_index.args[2]
                ).symbol()
            },
        )
    return indexing.replace(ModularIndexing, _visitor)


# TODO @bozhiyou this main (debug) codegen loop can be staticmethod
@monkey.patch(TritonScheduling)
def codegen_node_schedule_with_kernel(self: TritonScheduling, node_schedule, kernel: TritonKernel):
    """
    + set_current_node
    + induction variable elimination
    """
    def current_reduction_nodes(nodes):
        return itertools.takewhile(lambda n: n is not DisableReduction, nodes)

    fusion_log.debug(f"codegen for {node_schedule}")
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
                    assert len(index_vars) == len(node.get_ranges())
                    all_indexing.update(
                        dict.fromkeys(
                            indexing for indexing in node._body.indexing_from_args(index_vars).values()
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
    + relay block_hint from scheduler to kernel # TODO @bozhiyou this might not be necessary; irnode->snode->tree
    + reset tensor_dim
    """
    with monkey.fallback(self, node):
        if self.inside_reduction:  # default: multi-lane reduction
            # tensor_dim = itertools.count()
            for i, tree in enumerate(self.range_trees):
                assert tree.tensor_dim == i or tree.numel == 1, "broken assumption that each tree associates an output tensor dimension"
                # if tree.tensor_dim != i:
                #     tree.tensor_dim = i

        if block_hint:= getattr(node, 'block_hint', None):
            # block reduction does not do multi-lane reduction: now handled by ir
            # self.range_trees[-1].tensor_dim = None
            if old_block_hint:= getattr(self, 'block_hint', None):
                for new, old in zip(block_hint, old_block_hint):
                    assert new == old, f"incompatible block hint {old_block_hint} {block_hint}"  # TODO @bozhiyou is_compatible method
                if len(block_hint) > len(old_block_hint):
                    setattr(self, 'block_hint', block_hint)
            else:
                setattr(self, 'block_hint', block_hint)
        yield
        # has_old = hasattr(self, 'block_hint')
        # if has_old:
        #     old_block_hint = getattr(self, 'block_hint')
        # setattr(self, 'block_hint', block_hint)
        # yield
        # if has_old:
        #     setattr(self, 'block_hint', old_block_hint)
        # else:
        #     delattr(self, 'block_hint')



@monkey.patch(TritonKernel)
def set_ranges(self: TritonKernel, *lengths) -> list[list[Expr]]:
    """
    + also set range block hint
    """
    if blockhint:= getattr(self.current_node, 'block_hint', None):
        hint_offset = 0
        for i, ranges in enumerate(lengths):
            if hint_offset >= len(blockhint):
                break
            tree_hint = blockhint[hint_offset:hint_offset + len(ranges)]
            if old_tree_hint:= getattr(self.range_trees[i], 'block_hint', None):
                assert old_tree_hint == tree_hint, f"{tree_hint} overwrites {old_tree_hint} on {self.range_trees[i]}"
            setattr(self.range_trees[i], 'block_hint', tree_hint)
            hint_offset += len(ranges)
    return monkey.fallback(self, *lengths)


#####
# Range tree semantics
# (moved to reduction_kernel_fusion)
#####


@monkey.patch(TritonKernel)
def is_broadcasted(self: TritonKernel, index: sympy.Expr):
    """
    + fix use of entry.parent to entry.root
    + TODO @bozhiyou use total numel (ad hoc)
    Originally, kernel's numels is used as output shape here
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
    return simplify(sympy_product(index_numels)) == simplify(sympy_product(self.numels)) or any(
        simplify(idx_range) != simplify(iter_range)  # type: ignore[arg-type]
        for idx_range, iter_range in zip(index_numels, self.numels)
    )


#####
# Block tiling and indexing
#####


def _sort_var_list_by_stride(var_list: list[sympy.Symbol]) -> dict[int, list[sympy.Symbol]]:
    stride_to_vars = collections.defaultdict[int, list[sympy.Symbol]](list)
    for var in sorted(var_list, key=lambda var: V.kernel.range_tree_nodes[var].divisor):
        stride_to_vars[V.kernel.range_tree_nodes[var].divisor].append(var)
    return stride_to_vars


class RangeTreeExt:
    """Extends Inductor's "range tree" (IterationRangesRoot) for flexible, multi-dimensional tiling/blocking strategies.

    Inductor linearizes/flattens multi-dimensional iteration space into a single, contiguous iteration space (e.g., an index space from 0 to M*N, iterated by a single block size).
    High-performance Triton kernels require explicit, multi-dimensional blocking.

    This class takes a single, flat iteration dimension from Inductor's IterationRangesRoot (aka range tree) and logically re-groups its constituent loop variables into multiple, independent "blocked ranges". This allows the code generator to treat a single logical loop as a set of nested, tiled loops.
    """
    @dataclasses.dataclass()
    class BlockedRange:
        """
        Represents a logical, tiled dimension within a larger, linearized iteration space.

        Inductor's scheduler often linearizes a multi-dimensional problem into a
        1D iteration space. For example, a pointwise operation on a tensor of shape
        `(M, N)` becomes a single loop over `M*N` elements, tiled by a single
        block size, say `BLOCK_X`, then indexed by a single
        variable `x` from `0` to `M*N - 1`. While simple, this approach doesn't map well to
        the multi-dimensional nature of memory access patterns in backends like Triton.

        `BlockedRange` helps reconstruct this multi-dimensional view. It allows us
        to treat the iteration space as logically two-dimensional, with one dimension
        of size `M` and another of size `N`,
        tiled independently with block sizes `XBLOCK0` and `XBLOCK1` (then index by vars `x0` and `x1` respectively).

        This provides two main benefits:
        1.  **Flexibility**: A 2D tile of shape `(XBLOCK0, XBLOCK1)` is more
            flexible and maps better to 2D data layouts than a 1D tile of
            shape `(XBLOCK,)`.
        2.  **Configurability**: The block sizes can be different and tuned independently to find
            the optimal configuration for a given hardware architecture.

        Conversely, contiguous dimensions can be grouped into a single logical
        `BlockedRange`. For example, an iteration space with two contiguous
        dimensions (e.g., indexed by `x0` and `x1`) can be treated as a single
        linearized dimension (indexed by `x`) and tiled with a single block size
        `XBLOCK`.
        """
        range_tree: IterationRangesRoot
        meta: 'RangeTreeExt'
        numel: sympy.Symbol | sympy.Integer | Expr  # number of blocks
        stride: Expr
        var_list: list[sympy.Symbol] = dataclasses.field(default_factory=list)
        suffix: str = ''

        def __post_init__(self):
            """
            self.pid (only apply to grid dimensions e.g. x): A symbolic program ID, representing which block a particular GPU thread block is responsible for.
            self.block: The symbolic size (e.g., XBLOCK, RBLOCK) of this range/dimension. This allows for auto-tuning.
                For a dimension with block == 1, the corresponding loop variable (e.g., x0) in the kernel will represent a single value for each thread, rather than a vector or block of values (like tl.arange(0, BLOCK_SIZE)).
                `block == 1` is the primary mechanism to distinguish between a tiled ("sparse") dimension and a non-tiled ("dense") dimension during code generation
            self.offset: The starting offset for the current block (typically calculated as pid * block for grid dimensions).
            self.base: The base for intra-block indexing, usually tl.arange(0, block).
            """
            self.pid = sympy_index_symbol(f"pid{self.suffix}")
            # index = offset + base = offset + [0, block]
            if self.numel == 1:
                assert len(self.var_list) == 1, f"{self.var_list} expect a single var"
                one_shot = getattr(self.range_tree, 'one_shot')
                size = 0
                for size, var in one_shot.items():
                    if var in self.var_list:
                        break
                assert size, f"{self.var_list} {one_shot}"
                self.block = size  # tl.arange(0, dense_numel)
                self.offset = self.pid  # pid * 1
                self.base = sympy.Integer(0)  # 0 + tl.arange(0, dense_numel)
                return
            if self.numel.is_Integer:
                self.block = sympy.Integer(1)  # tl.arange(0, 1)
                self.offset = self.pid  # pid * 1
                self.base = sympy.Integer(0)  # 0 + tl.arange(0, 1)
                return
            self.block = sympy.Symbol(f"{self.range_tree.prefix.upper()}BLOCK{self.suffix}", integer=True, positive=True)
            self.offset = sympy_index_symbol(f"{self.range_tree.prefix}offset{self.suffix}")
            self.base = sympy_index_symbol(f"{self.range_tree.prefix}base{self.suffix}")

        def __hash__(self):
            return hash(tuple(self.var_list))

        @property
        def dense_numel(self):
            """
            number of elements at this logical level
            could be multiple vars where other vars covers across multiple ranges
            """
            stride_to_vars = _sort_var_list_by_stride(self.var_list)
            return sympy_product([min(self.range_tree.var_ranges[var] for var in vars) for vars in stride_to_vars.values()])


    def __init__(self, tree: IterationRangesRoot):
        """
        Initializes the RangeTreeExt by analyzing an IterationRangesRoot and
        logically grouping its loop variables into multi-dimensional blocks
        based on stride and optional blocking hints. This is a key step in
        translating Inductor's flat iteration space into a tiled structure
        suitable for high-performance Triton kernels.

        Args:
            tree (IterationRangesRoot): The 1D iteration space to be blocked.
        """
        self.range_tree = tree
        self.size: tuple[sympy.Symbol, ...]

        # 1. Group loop variables by their stride.
        # This helps in identifying contiguous dimensions that can be fused.
        var_list = tree.var_list
        stride_to_vars = _sort_var_list_by_stride(var_list)
        # Assumption: each stride value is unique among variables.
        assert all(len(v) < 2 for v in stride_to_vars.values())

        # 2. Process blocking hints provided by the scheduler.
        # The hint guides how variables are grouped into logical blocks.
        hint = getattr(tree, 'block_hint', [])
        if 0 < len(hint) < len(stride_to_vars):
            # Ad-hoc padding for incomplete hints.
            # TODO(bozhiyou): A more robust way might be to trace range splits.
            # raise NotImplementedError(f"insufficient hints: len({hint}) < {len(stride_to_vars)}")
            hint = hint[:1] * (len(stride_to_vars) - len(hint)) + hint  # ad hoc; TODO @bozhiyou trace range split
        self.hint = tuple(reversed(hint[-len(stride_to_vars):]))

        # Identify unique block groups from the hints.
        self.block_group = self.group(self.hint)

        # 3. Set up utilities for generating unique names for blocks.
        suffix = itertools.count()
        block_suffix = collections.defaultdict(lambda: f"{next(suffix)}")
        self.block_suffix = lambda b: block_suffix.get(b, '')

        # 4. Create BlockedRange objects based on hints.
        # This is where the 1D space is partitioned into a multi-dimensional
        # tiled structure.
        self.ranges: list[RangeTreeExt.BlockedRange] = []
        processed_label = set()
        if len(set(self.hint)) <= 1 and None not in self.hint:
            # Simple case: No hint or a uniform hint for all dimensions.
            # Blocking the entire iteration space with a single logical block.
            block = self.BlockedRange(tree, self,
                                      sympy.Symbol(f"{tree.prefix}numbl", integer=True, positive=True),  # Number of blocks is symbolic.
                                      sympy.Integer(1), var_list)
            self.ranges.append(block)
            self.var_to_block = lambda _: block
        else:
            # Complex case: Multi-dimensional or non-uniform blocking.
            var_to_block = {}
            size = []
            last = self.hint[0]
            for (stride, vars), b in zip(stride_to_vars.items(), self.hint):
                if b is None:
                    # No hint for this dimension, it's not part of a tiled block.
                    range_size = min(tree.var_ranges[var] for var in vars)
                    block = self.BlockedRange(tree, self, range_size, sympy_product(size), vars)
                    self.ranges.append(block)
                    for var in vars:
                        var_to_block[var] = block
                    size.append(range_size)
                    continue
                if b in processed_label:
                    # This hint has been seen, group with the previous block.
                    assert b == last, f"block hint must be consecutive labels {hint}; non-consecutive {b}"
                    block = self.ranges[-1]
                    block.var_list.extend(vars)
                    for var in vars:
                        var_to_block[var] = block
                    continue
                # A new block group is found.
                numel = sympy.Symbol(f"{tree.prefix}numbl{block_suffix[b]}", integer=True, positive=True)
                block = self.BlockedRange(tree, self, numel, sympy_product(size), vars, block_suffix[b])
                self.ranges.append(block)
                for var in vars:
                    var_to_block[var] = block
                size.append(numel)
                processed_label.add(b)
            self.var_to_block = lambda v: var_to_block[v]
            self.size = tuple(size)

        # 5. Create helper mappings for easy access during codegen.
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

    def insert_range(self, i, *, numel, stride, var_list, suffix=''):
        self.ranges.insert(i, self.BlockedRange(self.range_tree, self,
                                                numel=numel, stride=stride, var_list=var_list, suffix=suffix))


    def triton_tensor_ndim(self) -> int:
        """number of blocked dimensions"""
        if self.range_tree.tensor_dim is None:
            return 0
        return sum(1 for r in self.ranges if r.numel != 1 and r.block != 1)

    def dense_size_list(self) -> list[str]:
        return [f"{r.block}" for r in self.ranges if r.block != 1]


@monkey.patch(TritonKernel)
def finalize_indexing(self: TritonKernel, indices: Sequence[sympy.Expr]):
    """
    Hook called right before codegen with every index that will be
    used in the fused kernel.
    """
    for tree in self.range_trees:
        setattr(tree, 'block_meta', RangeTreeExt(tree))
    fusion_log.debug("-"*18)

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
#     # sizes[-1] = "None"  # assuming one r tree
#     # return f"{value}[{', '.join(sizes)}]"


@monkey.patch(TritonKernel)
def dense_size_list(self: TritonKernel) -> list[str]:
    """dense size/block size list
    + multiple dense size for a tree
    + suffix for dense size symbol
    """
    sizes = []
    for tree in self.range_trees:
        if tree.tensor_dim is None:
            continue
        if treex:= getattr(tree, 'block_meta', None):
            # treex: RangeTreeExt
            sizes.extend(treex.dense_size_list())
            continue
        # fallback to original implementation
        if tree.prefix != 'r' or self.inside_reduction:
            prefix, suffix = tree.name.split('index')
            assert tree.prefix == prefix, f"{tree.name=} {tree.prefix=}"
            sizes.append(f"{tree.prefix.upper()}BLOCK{suffix}")
    return sizes


#####
# Codegen
#####


@monkey.patch(TritonKernel)
def codegen_range_tree(self: TritonKernel) -> None:
    f"""Override by patch in kernel fusion
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
        block_meta: RangeTreeExt = getattr(entry, 'block_meta')
        if len(block_meta.ranges) <= 1:  # all indexing share same pid; TODO @bozhiyou this part seems covered by the general 'else' branch, remove this
            if block_meta.ranges[0].block == 1:  # no blocking
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

                if ran9e.block == 1:
                    for var in ran9e.var_list:
                        code.writeline(f"{var} = {self.kexpr(ModularIndexing(pid, ran9e.stride, entry.var_ranges[var]))}")
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
        # const range
        if one_shot_ranges:= getattr(entry, 'one_shot', {}):
            for var_size, var in one_shot_ranges.items():
                size = ''  # TODO @bozhiyou
                convert = f".to({self.index_dtype})" if self.index_dtype != "tl.int32" else ""
                code.writeline(f"{var} = tl.arange(0, {var_size}){size}{convert}")

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

class IndexingVarOrder:
    """
    order of vars represents the memory layout.

    # var name prefix (f"{prefix}{n}")
    - i: Loop IR pointwise indexing vars. The prefix is defined as `SymT.INDEX` in `torch.utils._sympy.symbol`.
        `Loop` or subclass IR calls `self.inner_fn_args` -> `self._index` that constructs the vars.
    - r: Loop IR reduction indexing vars. Similar but `SymT.RINDEX`.
    - d: `extract_read_writes` from `torch._inductor.dependencies` uses "d" as prefix for IR node execution.
    - z: `prefix="z"` when calling `dependencies.index_vars_no_squeeze` from `ComputedBuffer.simplify_and_reorder`.
        It has nothing to do with the z grid dimension here. Rather, it's used for retrace the loop body with simplification and reordering applied.
    - q: `prefix="q"` when calling `dependencies.index_vars_squeeze` from `ComputedBuffer.get_default_sizes_body`.
    """
    _index_order = {}

    @classmethod
    def add(cls, expr: sympy.Expr, ordered_vars: Sequence):
        vars = OrderedSet()
        for v in ordered_vars:
            if v.is_Integer:
                continue
            if not expr.has(v):
                continue
            if not v.is_Symbol:
                if isinstance(v, ModularIndexing):
                    v = v.args[0]
                elif len(v.free_symbols) == 1:
                    v = next(iter(v.free_symbols))
                else:
                    # expression from induction variable elimination
                    assert isinstance(v, sympy.Add), v
                    coef = v.as_coefficients_dict()
                    for var, coeff in sorted(coef.items(), key=lambda item: item[1]):
                      vars.add(var)
                    continue
            vars.add(v)
        cls._index_order[expr] = tuple(vars)

    @classmethod
    def update(cls, expr, new_expr, replacements={}):
        if expr not in cls._index_order:
            return False
        replaced_vars = [(replacements[v] if v in replacements else v) for v in cls._index_order[expr]]
        cls.add(new_expr, replaced_vars)
        return True

    @classmethod
    def get(cls, expr):
        return cls._index_order.get(expr)



@monkey.patch(ir.FixedLayout)
def make_indexer(self: ir.FixedLayout):
    """
    + keep original index order
    """

    def indexer(index):
        assert len(index) == len(self.stride)
        assert len(index) == len(self.size)
        result = self.offset
        for idx, stride, sz in zip(index, self.stride, self.size):
            if sz != 1:
                result = result + idx * stride
        IndexingVarOrder.add(result, index)
        return result

    return indexer


@monkey.patch(LoopBody)
def indexing_from_args(self: LoopBody, indices):
    """
    + keep original index order
    """
    index = [*itertools.chain.from_iterable(indices)]
    vars = [v for v in itertools.chain.from_iterable(self.vars) if v.is_Symbol]
    assert len(index) == len(vars), (index, self.vars)
    assert all(
        v not in vars for v in index
    ), f"same var does not need to replace: {self.var_ranges=}, {indices=}"
    replacements = dict(zip(vars, index))
    indexing = {}
    for name, expr in self.indexing_exprs.items():
        sub = sympy_subs(expr, replacements)
        indexing[name] = sub
        IndexingVarOrder.update(expr, sub, replacements)
    return indexing

# @monkey.patch(LoopBody)
def add_index_expr(self: LoopBody, expr: sympy.Expr, *args, **kwargs):
    simplified = V.graph.sizevars.combine_modular_indexing_pairs(expr, self.var_ranges)
    return monkey.fallback(self, simplified, *args, **kwargs)

@monkey.patch(SimplifyIndexing)
def __init__(self, inner, var_ranges: VarRanges) -> None:
    monkey.fallback(self, inner, var_ranges)
    _simplify = self._simplify
    def _simplify_with_order_trace(index):
        nonlocal _simplify
        result = _simplify(index)
        if result != index:
            IndexingVarOrder.update(index, result)
        return result
    self._simplify = _simplify_with_order_trace


def _is_one_shot(var: sympy.Symbol, kernel: TritonKernel) -> bool:
    return kernel.range_tree_nodes[var].expr is var
    # return var not in kernel.range_tree_nodes

def get_variable_to_block_dim_map(self: TritonKernel, index: sympy.Expr) -> dict[sympy.Symbol, int]:
    """
    Maps index variables to their logical block dimension for N-dimensional tiling.

    In Inductor, a multi-dimensional iteration space is often flattened into a
    single loop. This function reverse-engineers the multi-dimensional structure
    from a flattened `index` expression. It analyzes how each symbolic variable
    (e.g., `x0`, `x1`, `r0`) contributes to the final index and assigns it to a
    logical "block dimension" (0, 1, 2, ...).

    This mapping is crucial for generating correctly-shaped, N-dimensional masks
    and indexing expressions in Triton. It allows pointwise operations on tensors
    of different ranks to be fused by correctly broadcasting their masks and indices
    to a common, multi-dimensional tile shape.

    For example, for a 2D access pattern with an index like `x0*128 + x1`, this
    function would produce a mapping like `{x0: 1, x1: 0}`. This indicates a
    logical 2D layout that can be used to generate broadcastable masks, such as:
    `mask_x0 = (x0 < M)[:, None]` and `mask_x1 = (x1 < N)[None, :]`.

    Args:
        self: The TritonKernel instance.
        index: The flattened 1D indexing expression to analyze.

    Returns:
        A dictionary mapping each symbolic variable in the index to its
        corresponding logical block dimension index.
    """
    # index vars in order of dimension
    index_vars: tuple[sympy.Symbol]
    if isinstance(index, sympy.Symbol) or not index.args:  # singleton
        index_vars = (index,)
    elif vars:= IndexingVarOrder.get(index):
        index_vars = tuple(reversed(vars))
        # add simplified index as key
        simplified = self.simplify_indexing(index)
        if index != simplified:
            IndexingVarOrder.add(simplified, vars)
    else:
        # TODO REMOVE: this ordering is not stable
        assert all(len(arg.free_symbols) <= 1 for arg in index.args), f"{[arg.free_symbols for arg in index.args]}"
        index_vars = tuple(var for arg in index.args for var in arg.free_symbols)  # `args` is (kind of) ordered; `free_symbols` is not

    dim = itertools.count()
    _blocked_range_order = dict[RangeTreeExt.BlockedRange, int]()
    var_to_blocked_dim = dict[sympy.Symbol, int]()

    for i, var in enumerate(index_vars):
        if var in var_to_blocked_dim:  # Already processed
            continue
        if _is_one_shot(var, self):  # A "one-shot" range is a dimension that is not iterated over in a loop but is instead processed "all at once" within a single block (e.g., using tl.arange) (conceptually, when block == dense size of the dimension).
            var_to_blocked_dim[var] = next(dim)  # Assign it a unique block dimension.
            continue

        # Get the extended range tree representation
        tree = self.range_tree_nodes[var].root
        treex: RangeTreeExt = getattr(tree, 'block_meta', None)

        if not treex:
            # single block size by default
            # for var in tree.var_list:
            var_to_blocked_dim[var] = next(dim)
        else:
            ran9e: RangeTreeExt.BlockedRange = treex.var_to_block(var)
            if ran9e.block == 1:
                # This dimension is not tiled (block size is 1). The corresponding loop
                # variable (e.g., `x2`) will be a scalar within the kernel, representing a
                # single element for the entire thread block.
                #
                # Since it's not tiled, it doesn't get a block dimension index and won't be
                # part of the N-dimensional block-level mask. Instead, its boundary check
                # will be a simple scalar guard condition (e.g., `(x2 < B)`).
                #
                # For example, an indexing expression `x0[None, :] + x1[:, None] + x2` might
                # have a combined mask `(x0 < M)[None, :] & (x1 < N)[:, None]` for the tiled
                # dimensions `x0` and `x1`, while the check for `x2` remains a separate scalar guard.
                # The final masking will be `(x0 < M)[None, :] & (x1 < N)[:, None] & (x2 < B)`.
                continue
            # The bookkeeping `_blocked_range_order` is for the case when a `BlockedRange` has more than one loop vars (which is often the case for pre-fusion scheduler nodes where the block hint is not processed yet).
            # For example, if a blocked range has var list (x0, x1), presumably they are indexing continuously, so the indexing is `(x0 * s0)[mask] + (x1 * s1)[mask]` with same mask.
            if ran9e not in _blocked_range_order:
                _blocked_range_order[ran9e] = next(dim)
            var_to_blocked_dim[var] = _blocked_range_order[ran9e]
    return var_to_blocked_dim


def _simplify_subexpression(self: TritonKernel, index: sympy.Expr):
    expr = self.simplify_indexing(index)  # V.graph.sizevars.simplify_with_ranges(index, self.var_ranges())
    if not expr.has(FloorDiv) and not expr.has(ModularIndexing):
        return expr

    var_ranges = {k: v for k, v in self.var_ranges().items() if isinstance(k, sympy.Add)}
    if not var_ranges:
        return expr

    def remove_zero_terms(base, divisor):
        for v in var_ranges:
            if base.has(v):
                # var smaller than divisor can be removed
                # if the rest is guaranteed to be multiple of divisor
                rest = sympy.Wild("_rest", exclude=[v])
                m = base.match(v + rest)
                if m and not m[rest].has(v):
                    gcd = sympy.gcd(m[rest], divisor)
                    if gcd == divisor:
                        if var_ranges[v] <= divisor:
                            base = m[rest]
        return base

    def visit_indexing_div(base, divisor):
        return FloorDiv(remove_zero_terms(base, divisor), divisor)

    def visit_modular_indexing(base, divisor, modulus):
        base = remove_zero_terms(base, divisor)
        return ModularIndexing(base, divisor, modulus)

    if expr.has(ModularIndexing):
        expr = expr.replace(
            ModularIndexing(
                sympy.Wild("base", integer=True),
                sympy.Wild("divisor", integer=True),
                sympy.Wild("modulus", integer=True),
            ),
            visit_modular_indexing,
        )

    if expr.has(FloorDiv):
        expr = expr.replace(
            FloorDiv(
                sympy.Wild("base", integer=True),
                sympy.Wild("divisor", integer=True),
            ),
            visit_indexing_div,
        )

    return expr

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
    Add size expansion/broadcasting to mask as `override_mask`.
    The original implementation generated simple, 1D masks. This patch enables the creation of N-dimensional masks that can be broadcast correctly across a tiled kernel, which is essential when fusing operations with different but compatible iteration spaces.

    It first determines which logical "blocked dimension" each index variable belongs to.
    Construct N-Dimensional Masks:
        If the indexing expression only involves variables from a single blocked dimension (or is a simple 1D case), it generates a standard 1D mask.
        If the indexing involves variables from multiple blocked dimensions, it constructs a separate mask for each variable. Crucially, it expands the shape of each mask with None (equivalent to unsqueeze) to match the total number of blocked dimensions. For example, a variable in the second of three blocked dimensions would get a mask shaped like [None, :, None].
    Combine and Override: The individual N-dimensional masks are combined with a logical AND (&). The resulting string is then passed as the override_mask to the original indexing implementation, which proceeds with the rest of the code generation.
    """
    if override_mask is None:
        var_to_blocked_dim = get_variable_to_block_dim_map(self, index)
        ndim = len(set(var_to_blocked_dim.values()))

        index = _simplify_subexpression(self, index)
        var_ranges = self.var_ranges()
        if ndim <= 1:
            override_mask = ' & '.join(f"({var} < {var_ranges[var]})" for arg in index.args for var in arg.free_symbols)
        else:
            masks = []
            for arg in (index.args or (index,)):  # `args` is ordered, `free_symbols` is not
                if not arg.free_symbols:  # constant offset
                    continue
                assert len(arg.free_symbols) == 1
                var = next(iter(arg.free_symbols))
                if var not in var_ranges:
                    continue  # one-shot var
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
    """Extends `index_to_str` to handle N-dimensional tiling for advanced fusions.

    TODO @bozhiyou: This logic might be better placed in `TritonKernel.indexing`
    to handle indexing semantics more centrally. This patch acts as a
    post-processing step on the generated index string.

    The original Inductor `index_to_str` would produce indexing variables that
    were pre-broadcasted (e.g., `x0 = tl.arange(0, BLOCK)[:, None]`). This patch,
    in conjunction with a modified `codegen_iteration_ranges_entry`, decouples
    the variable's definition from its broadcasting. The base variable is defined
    as a 1D vector (e.g., `x0 = tl.arange(0, BLOCK)`), and this function adds the
    necessary broadcasting semantics (e.g., `[:, None]`) during string generation.

    This is achieved by:
    1.  Analyzing the `index` expression to determine which logical "blocked
        dimension" each variable belongs to, using `get_variable_to_block_dim_map`.
    2.  If the index spans multiple logical dimensions, it reconstructs the
        Triton code to explicitly broadcast each component of the index to the
        correct shape. For example, an expression like `x0*128 + x1` (representing
        a 2D access) is converted into `(x0*128)[:, None] + (x1)[None, :]`.

    This enables pointwise operations on tensors of different ranks to be fused
    by ensuring their indices are broadcastable to a common multi-dimensional tile shape.
    If the index only involves a single dimension, it requires no broadcasting and
    the original behavior is preserved.
    """
    index_str: str = monkey.fallback(self, index)
    if isinstance(index, list):
        return index_str

    var_to_blocked_dim = get_variable_to_block_dim_map(self, index)
    ndim = len(set(var_to_blocked_dim.values()))

    if ndim <= 1:
        return index_str
    args = []
    for arg in (index.args or (index,)):
        if not arg.free_symbols:  # constant offset
            args.append(f"{arg}")
            continue
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
def codegen_iteration_ranges_entry(self: TritonKernel, entry: IterationRangesEntry):
    """Entries are used as bases."""
    line = f"{entry.name} = {self.kexpr(self.rename_indexing(entry.expr))}"
    if entry.root.is_loop:
        if buffer:= getattr(entry.root, 'code_buffer', None):
            buffer.indexing_code.writeline(line)
            return
        self.indexing_code.writeline(line)
        return
    if entry.root.prefix == 'r' and isinstance(entry.expr, sympy.Symbol):  # non-loop symbol -> one shot
        size = ''  # TODO @bozhiyou
        convert = f".to({self.index_dtype})" if self.index_dtype != "tl.int32" else ""
        code = self.body
        if buffer_stack:= getattr(self, '_buffer_stack', []):
            code = buffer_stack[0].body
        code.writeline(f"{entry.name} = tl.arange(0, {entry.length}){size}{convert}")
        return
    # non-reduction indexing lifted outside loop
    self.body.writeline(f"# {line}")
    return


@monkey.patch(TritonKernel)
def combine_contiguous_dims(self, index: sympy.Expr, tree: IterationRangesRoot):
    """Disable combining (flattening) because ND tiling is implemented; keep the broadcasted shape"""
    return index


@monkey.patch(TritonKernel)
def codegen_static_numels(self, code) -> None:
    """Insert constexpr at the start of kernel."""
    monkey.fallback(self, code)
    for key, val in getattr(self, 'constexpr', {}).items():
        code.writeline(f"{key}: tl.constexpr = {val}")


@monkey.patch(TritonKernel)
def _get_heuristic(self: TritonKernel):
    if getattr(self, 'block_hint', None) or any(getattr(tree, 'block_hint', None) for tree in self.range_trees):
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
        code.splice("import monkeypatch.fusion.triton_heuristics")

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

    inductor_meta['block_args'] = {}
    for tree in self.range_trees:
        if tree.prefix == "r" and (self.persistent_reduction or not self.inside_reduction):
            # RBLOCK for persistent_reduction is defined in codegen_static_numels
            continue
        # if tree.tensor_dim is None:
        #     continue
        prefix, suffix = tree.name.split('index')
        assert prefix == tree.prefix
        numel = f"{tree.prefix}numel{suffix}"
        inductor_meta['block_args'][numel] = tree.numel
        if heuristics == 'blockreduction':
            if block_meta:= getattr(tree, 'block_meta', None):
                for ran9e in block_meta.ranges:
                    if not ran9e.block.is_symbol:
                        inductor_meta['block_args'][numel] = ceildiv(inductor_meta['block_args'][numel], ran9e.numel)
                        continue
                    argdefs.append(f"{ran9e.block}: tl.constexpr")
                    if not tree.is_loop:
                        argdefs.append(f"{ran9e.numel}: tl.constexpr")
                        inductor_meta['block_args'][f"{ran9e.numel}"] = sympy_product(tree.var_ranges[var] for var in ran9e.var_list)
        # NOTE @bozhiyou still keep XBLOCK as grid scalar
        block = f"{tree.prefix.upper()}BLOCK{suffix}: tl.constexpr"
        if block not in argdefs:
            argdefs.append(block)

    # for kernel fusion
    for arg, numel in getattr(self.args, 'constexprs', {}).items():
        inductor_meta['block_args'][arg] = numel
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


from . import triton_heuristics
