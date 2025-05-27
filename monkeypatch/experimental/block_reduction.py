from .. import _monkey as monkey
from typing import Any, Sequence, Callable

import collections
import contextlib
import dataclasses
import functools
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
from torch._inductor.sizevars import SizeVarAllocator, SimplifyIndexing
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
        """
        + instanciate using cls rather than Reduction to allow subclassing
        """
        reduction_numel = V.graph.sizevars.simplify(sympy_product(reduction_ranges))

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
        """
        # + interpreted as block_reduction
        + interpreted as single-lane reduction
        """
        # value = ops.block_reduction(
        value = ops.reductionx(
            self.dtype,
            self.src_dtype,
            self.reduction_type,
            self.inner_fn(vars, reduction_vars),
            contraction=True,
        )
        return ops.store_reduction(output_name, indexer(vars), value)


@monkey.patch(TritonKernelOverrides)
@staticmethod
def dot(a, b):
    return f"tl.dot({a}, {b})"


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
        return ops.dot(
            ops.to_dtype(m1, mat1.dtype, use_compute_types=False),  # fix tl.dot type promotion 
            ops.to_dtype(m2, mat2.dtype, use_compute_types=False),
        )

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


# TODO @bozhiyou this main (debug) codegen loop can be staticmethod
@monkey.patch(TritonScheduling)
def codegen_node_schedule_with_kernel(self: TritonScheduling, node_schedule, kernel: TritonKernel):
    """
    + set_current_node
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
    + relay block_hint from scheduler to kernel # TODO @bozhiyou this might not be necessary; irnode->snode->tree
    + reset tensor_dim
    """
    with monkey.fallback(self, node):
        if self.inside_reduction:  # default: multi-lane reduction
            # tensor_dim = itertools.count()
            for i, tree in enumerate(self.range_trees):
                assert tree.tensor_dim == i, "broken assumption"
                if tree.tensor_dim != i:
                    tree.tensor_dim = i

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
#####

@monkey.patch(IterationRangesRoot)
def index_sym(self: IterationRangesRoot):
    if any(tree != self and tree.prefix == self.prefix for tree in V.kernel.range_trees):
        return sympy_index_symbol(f"{self.prefix}index{self.index}")
    return sympy_index_symbol(f"{self.prefix}index")


# def lookup(self: IterationRangesRoot, divisor, length, parent):
#     """Not patching the original method because of incompatible signature.
#     + parent setting to reflect tree structure
#     """
#     if V.graph.sizevars.statically_known_equals(divisor * length, self.numel):
#         expr = FloorDiv(self.index_sym(), divisor)
#     else:
#         expr = ModularIndexing(self.index_sym(), divisor, length)

#     if expr not in self.nodes:
#         node = IterationRangesEntry(
#             f"{self.prefix}{next(V.kernel.iter_vars_count)}",
#             divisor,
#             length,
#             expr,
#             parent,
#         )
#         V.kernel.range_tree_nodes[node.symbol()] = node
#         self.var_list.append(node.symbol())
#         self.var_ranges[node.symbol()] = length
#         self.nodes[expr] = node
#     return self.nodes[expr]


# @monkey.patch(IterationRangesRoot)
# def construct_entries(self: IterationRangesRoot, lengths: list[sympy.Expr]):
#     """
#     + maintain (reversed) parentship
#     """
#     divisor = sympy.Integer(1)
#     itervars = []
#     for length in reversed(lengths):
#         itervars.append(lookup(self, divisor, length, itervars[-1] if itervars else self))
#         divisor = divisor * length
#     return list(reversed(itervars))
#     # """
#     # + construct entries in original order to maintain parentship
#     # """
#     # reversed_divisors = [sympy.Integer(1)]
#     # for length in reversed(lengths[1:]):
#     #     reversed_divisors.append(reversed_divisors[-1] * length)
#     # itervars = []
#     # for length, divisor in zip(lengths, reversed(reversed_divisors)):
#     #     assert V.graph.sizevars.statically_known_multiple_of(self.numel, divisor), f"{self} cannot be split by {divisor}"
#     #     itervars.append(lookup(self, divisor, length, itervars[-1] if itervars else self))
#     # return itervars


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


def sort_var_list_by_stride(var_list: list[sympy.Symbol]) -> dict[int, list[sympy.Symbol]]:
    stride_to_vars = collections.defaultdict[int, list[sympy.Symbol]](list)
    for var in sorted(var_list, key=lambda var: V.kernel.range_tree_nodes[var].divisor):
        stride_to_vars[V.kernel.range_tree_nodes[var].divisor].append(var)
    return stride_to_vars


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
            number of elements at this level
            could be multiple vars where other vars covers across multiple ranges
            """
            stride_to_vars = sort_var_list_by_stride(self.var_list)
            return sympy_product([min(self.range_tree.var_ranges[var] for var in vars) for vars in stride_to_vars.values()])


    def __init__(self, tree: IterationRangesRoot):
        self.range_tree = tree
        self.size: tuple[sympy.Symbol, ...]

        # if not _is_var_list_ordered(tree.var_list):
        #     fusion_log.debug("\033[033m @bozhiyou should not reach here but in case the invariance is violated \033[0m")
        #     var_list = [entry.symbol() for entry in _get_range_hierarchy(tree)]
        #     assert set(var_list) == set(tree.var_list), f"{tree.var_list} {var_list}"
        #     tree.var_list = var_list
        var_list = tree.var_list
        stride_to_vars = sort_var_list_by_stride(var_list)

        hint = getattr(tree, 'block_hint', [])
        if 0 < len(hint) < len(stride_to_vars):
            # raise NotImplementedError(f"insufficient hints: len({hint}) < {len(stride_to_vars)}")
            hint = hint[:1] * (len(stride_to_vars) - len(hint)) + hint  # ad hoc; TODO @bozhiyou trace range split
        self.hint = tuple(reversed(hint[-len(stride_to_vars):]))

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
            self.ranges.append(block)
            self.var_to_block = lambda _: block
        else:
            var_to_block = {}
            size = []
            last = self.hint[0]
            for (stride, vars), b in zip(stride_to_vars.items(), self.hint):
                if b is None:
                    range_size = min(tree.var_ranges[var] for var in vars)
                    block = self.BlockedRange(tree, self, range_size, sympy_product(size), vars)
                    self.ranges.append(block)
                    for var in vars:
                        var_to_block[var] = block
                    size.append(range_size)
                    continue
                if b in processed_label:
                    assert b == last, f"block hint must be consecutive labels {hint}; non-consecutive {b}"
                    block = self.ranges[-1]
                    block.var_list.extend(vars)
                    for var in vars:
                        var_to_block[var] = block
                    continue
                numel = sympy.Symbol(f"{tree.prefix}numbl{block_suffix[b]}", integer=True, positive=True)
                block = self.BlockedRange(tree, self, numel, sympy_product(size), vars, block_suffix[b])
                self.ranges.append(block)
                for var in vars:
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

    def insert_range(self, i, *, numel, stride, var_list, suffix=''):
        self.ranges.insert(i, self.BlockedRange(self.range_tree, self,
                                                numel=numel, stride=stride, var_list=var_list, suffix=suffix))
    
    
    def triton_tensor_ndim(self) -> int:
        """number of blocked dimensions"""
        if self.range_tree.tensor_dim is None:
            return 0
        return sum(1 for r in self.ranges if r.block != 1)
    
    def dense_size_list(self) -> list[str]:
        return [f"{r.block}" for r in self.ranges if r.block != 1]


@monkey.patch(TritonKernel)
def finalize_indexing(self: TritonKernel, indices: Sequence[sympy.Expr]):
    """
    Hook called right before codegen with every index that will be
    used in the fused kernel.
    """
    for tree in self.range_trees:
        setattr(tree, 'block_meta', BlockMeta(tree))
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
            # treex: BlockMeta
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
                else:
                    assert len(v.free_symbols) == 1, f"{v}, {v.free_symbols}"
                    v = next(iter(v.free_symbols))
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
    assert len(index) == len(self.var_ranges), (index, self.var_ranges)
    assert all(
        v not in self.var_ranges for v in index
    ), f"{self.var_ranges=}, {indices=}"
    replacements = dict(zip(self.var_ranges.keys(), index))
    indexing = {}
    for name, expr in self.indexing_exprs.items():
        sub = sympy_subs(expr, replacements)
        indexing[name] = sub
        IndexingVarOrder.update(expr, sub, replacements)
    return indexing

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


def index_var_to_blocked_dim(self: TritonKernel, index: Expr) -> dict[sympy.Symbol, int]:
    index_vars: tuple[sympy.Symbol]
    if isinstance(index, sympy.Symbol) or not index.args:  # singleton
        index_vars = (index,)
    elif vars:= IndexingVarOrder.get(index):
        index_vars = tuple(reversed(vars))
    else:
        # TODO REMOVE: this ordering is not stable
        assert all(len(arg.free_symbols) <= 1 for arg in index.args), f"{[arg.free_symbols for arg in index.args]}"
        index_vars = tuple(var for arg in index.args for var in arg.free_symbols)  # `args` is (kind of) ordered; `free_symbols` is not

    dim = itertools.count()
    block_size_to_dim = dict[sympy.Symbol, int]()
    var_to_blocked_dim = dict[sympy.Symbol, int]()
    # if self.range_trees[-1].tensor_dim is not None:
    #     # reserve reduction dimension
    #     if treex:= getattr(self.range_trees[-1], 'block_meta', None):
    #         blocked_range = treex.ranges[-1]
    #         if blocked_range.block != 1:
    #             next(dim)  # 0
    #             for var in blocked_range.var_list:
    #                 var_to_blocked_dim[var] = 0
    #     else:
    #         # single block size by default
    #         next(dim)  # 0
    #         for var in self.range_trees[-1].var_list:
    #             var_to_blocked_dim[var] = 0

    for i, var in enumerate(index_vars):
        if var in var_to_blocked_dim:
            continue
        if var not in self.range_tree_nodes:  # one-shot range
            var_to_blocked_dim[var] = next(dim)
            continue
        tree = self.range_tree_nodes[var].root
        treex = getattr(tree, 'block_meta', None)
        if not treex:
            # single block size by default
            # for var in tree.var_list:
            var_to_blocked_dim[var] = next(dim)
        else:
            ran9e = treex.var_to_block(var)
            if ran9e.block != 1:
                if ran9e.block not in block_size_to_dim:
                    block_size_to_dim[ran9e.block] = next(dim)
                var_to_blocked_dim[var] = block_size_to_dim[ran9e.block]
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
        if buffer:= getattr(entry.root, 'code_buffer', None):
            buffer.indexing_code.writeline(line)
            return
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
        code.splice("from monkeypatch.experimental import block_reduction")

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

    import operator
    def grid_fn(meta):
        x_grid = xnumel
        xblock = meta.get("XBLOCK", 1)
        if isinstance(xblock, tuple):  # {number of elements: number of blocks}
            for ne, nb in xblock:
                assert x_grid % ne == 0, f"{x_grid} is not multiple of {ne}! {xnumel=} {xblock=}"
                x_grid //= ne
                x_grid *= nb
        else:
            x_grid = get_grid_dim(xnumel, xblock)
        # x_grid = get_grid_dim(xnumel, meta.get("XBLOCK", 1))
        # for k, v in meta.items():
        #     if k.startswith('xnumbl'):
        #         x_grid *= v
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


def _blockreduction_configs(
    *,
    size_hints: dict[str, int],
    inductor_meta={},
):
    f"""
    Config space from {torch._inductor.kernel.flex_attention} and {torch._inductor.kernel.mm_common}.
    """
    from torch._inductor.kernel.flex_attention import _get_default_config_fwd
    # (BLOCK_M, BLOCK_N, num_warps, num_stages)
    configs: list[tuple[int, int, int, int]] = [(128, 32, 4, 3)]  # TODO @bozhiyou default to max or 1?
    # configs.append(_get_default_config_fwd(query))
    if torch._inductor.config.max_autotune:
        configs += [
            (128, 64, 4, 3),
            (128, 128, 4, 3),
            (128, 128, 8, 2),
            (64, 128, 4, 3),
            (64, 64, 4, 3),
        ]

    if 'block_args' in inductor_meta:
        block_hints = inductor_meta['block_args']
        # 'block_args':
        #   # _numel: number of grouped elements, _numel = prod(_numbl_)
        #   'xnumel': 16384,
        #
        #   # _numbl_: nnn means _numbl_ = ceildiv(nnn, _BLOCK_)
        #   'xnumbl0': 16384,
        #   'RBLOCK1': None, 'RBLOCK': None
        def block_config(xblock, rblock):
            block = {'x': xblock, 'r': rblock}
            c = {"XBLOCK": ()}
            for arg, v in block_hints.items():
                if 'numbl' in arg:
                    prefix, suffix = arg.split('numbl')
                    c[arg] = ceildiv(v, block[prefix])
                    c[f"{prefix.upper()}BLOCK{suffix}"] = block[prefix]
                    c[f"{prefix.upper()}BLOCK"] += ((v, c[arg]),)
                if 'BLOCK' in arg:
                    if v <= 2048:
                        c[arg] = v
                    else:
                        c[arg] = block[arg[0].lower()]
            return c

        return [
            triton.Config({  # keys must be kernel args
                **block_config(XBLOCK, RBLOCK),
                # "XBLOCK": inductor_meta['block_args']["xnumel"],  # force XBLOCK to 1 to recalculate number of blocks
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
    size_hints: dict[str, int],
    reduction_hint=False,
    triton_meta={},
    filename=None,
    inductor_meta={},
):
    inductor_meta["reduction_hint"] = reduction_hint
    if inductor_meta.get("no_x_dim"):
        size_hints["x"] = 1

    configs = _blockreduction_configs(size_hints=size_hints, inductor_meta=inductor_meta)

    return cached_autotune(
        size_hints,
        configs,
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

