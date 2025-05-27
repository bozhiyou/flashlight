from . import _monkey as monkey

from ._fusion_common import TRITON_MAX_RBLOCK

import collections
import itertools
import contextlib
import copy
import functools
from typing import Any, Union, Tuple, Callable

import torch
import torch._inductor.config
from torch import fx
from torch._inductor import ir, scheduler
from torch._inductor.virtualized import V, ops, OpsValue
from torch.utils._ordered_set import OrderedSet

from torch._inductor.scheduler import BaseSchedulerNode, SchedulerNode, FusedSchedulerNode, OutputNode, SchedulerBuffer
from torch._inductor.ops_handler import ReductionType, KernelFormatterHandler
from torch._inductor.codegen.common import CSEVariable
from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling
from torch._inductor.codegen.triton import TritonScheduling, TritonKernelOverrides, TritonKernel, TritonCSEVariable
from torch._inductor.codegen.triton import triton_acc_type
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.dependencies import Dep, MemoryDep, WeakDep
from torch._inductor.utils import reduction_num_outputs, is_welford_reduction, IndentedBuffer, sympy_product

import sympy


# add 'fusion' to comma-separated TORCH_LOG to enable
fusion_log = torch._logging.getArtifactLogger('torch._inductor', "fusion")


aten = torch.ops.aten

from torch._inductor.fx_passes import joint_graph
def disable_scaled_softmax_pattern():
    """
    This pattern replacement aims to improve numerical stability for scaling but prevent from fusion.
    NOTE @bozhiyou Can't scaling be always fused with softmax? why do we need this replacement
    """
    p = joint_graph._partial_softmax_pattern(aten.mul.Tensor)
    for (op, target), entries in joint_graph.pass_patterns[1].patterns.items():
        for entry in entries:
            if repr(p) == repr(entry.pattern):  # pattern comparison not defined
                entries.clear()
                return

disable_scaled_softmax_pattern()

from torch._inductor.pattern_matcher import CallFunction, Arg, KeywordArg, MULTIPLE, Match, register_graph_pattern
class div_bmm:
    @staticmethod
    def pattern(linear_func, reverse=False, to_dtype=False):
        scaled = CallFunction(
            linear_func, KeywordArg("a"), KeywordArg("denom"), _users=MULTIPLE
        )

        return CallFunction(aten.bmm, scaled, KeywordArg("b"))

    @staticmethod
    def extra_check(match: Match):
        return isinstance(match.kwargs["denom"], fx.Node) and (
            match.kwargs["denom"].target == aten.sum.dim_IntList
        )

    @staticmethod
    def handler(match: Match, *, a, b, denom):
        fusion_log.debug("div-bmm pattern matched")
        def repl(a, b, denom):
            return torch.bmm(a, b) / denom
        match.replace_by_example(repl, [a, b, denom])
    
    @classmethod
    def enable(cls):
        register_graph_pattern(
            cls.pattern(aten.div),
            pass_dict=joint_graph.pass_patterns[1],
            extra_check=cls.extra_check,
        )(cls.handler)

div_bmm.enable()

class div_expand_view_bmm:
    @staticmethod
    def pattern(linear_func, reverse=False, to_dtype=False):
        scaled = CallFunction(
            linear_func, KeywordArg("a"), KeywordArg("denom"), _users=MULTIPLE
        )
        expand = CallFunction(
            aten.expand, scaled, KeywordArg("expand_sizes"),
        )
        view = CallFunction(
            aten.view, expand, KeywordArg("view_shape"),
        )
        return CallFunction(aten.bmm, view, KeywordArg("b"))

    @staticmethod
    def extra_check(match: Match):
        return isinstance(match.kwargs["denom"], fx.Node) and (
            match.kwargs["denom"].target == aten.sum.dim_IntList
        )

    @staticmethod
    def handler(match: Match, *, a, b, denom, expand_sizes, view_shape):
        fusion_log.debug("div-expand-view-bmm pattern matched")
        def repl(a, b, denom, expand_sizes, view_shape):
            return torch.bmm(a.expand(*expand_sizes).view(*view_shape), b) / denom.view(*view_shape[:-2], *denom.shape[-2:])
        match.replace_by_example(repl, [a, b, denom, expand_sizes, view_shape])

    @classmethod
    def enable(cls):
        register_graph_pattern(
            cls.pattern(aten.div),
            pass_dict=joint_graph.pass_patterns[1],
            extra_check=cls.extra_check,
        )(cls.handler)

div_expand_view_bmm.enable()


#####
# Extended IR
#####

torch._inductor.config.triton.codegen_upcast_to_fp32 = False

import torch._inductor.codegen.triton
_triton_compute_type = torch._inductor.codegen.triton.triton_compute_type
def triton_compute_type(dtype: torch.dtype):
    triton_type_name = str(dtype).split(".")[-1]
    if triton_type_name in ("float16", "bfloat16"):
        return "tl.float32"  # tl.dot output will be promoted to tl.float32  # TODO @bozhiyou only for tl.dot; for sum etc. still fp16/bf16?
    return _triton_compute_type(dtype)
torch._inductor.codegen.triton.triton_compute_type = triton_compute_type


@monkey.patch(TritonKernelOverrides)
@staticmethod
def max(x, dim):
    return f"triton_helpers.max2({x}, {dim})"

@monkey.patch(TritonKernel)
def reductionx(
    self: TritonKernel,
    dtype: torch.dtype,
    src_dtype: torch.dtype,
    reduction_type: ReductionType,
    value: Union[CSEVariable, Tuple[CSEVariable, ...]],
    # extended args
    multilane=True,     # i.e. RBLOCK > 1
    contraction=False,      # no r dim in result if True
    writeback_later=False,
    modification: int|None = None,
) -> Union[CSEVariable, Tuple[CSEVariable, ...]]:
    assert self == V.kernel
    assert isinstance(self, TritonKernel)

    assert self.inside_reduction
    if multilane and not contraction:
        masks = OrderedSet(f"{tree.prefix}mask" for tree in self.range_trees)
        self.filter_masks(masks)
        masks = sorted(masks)
        if self._load_mask:
            masks.append(self._load_mask)
    else:
        masks = []
    reduction_range_prefix = self.range_trees[-1].prefix

    if multilane and not contraction:
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

    dim: int
    root_op: str

    def final_reduction(value):
        use_helper = reduction_type in {"any", "max", "min", "prod"}
        module = "triton_helpers" if use_helper else "tl"
        if reduction_type in {"max", "min"}:
            return self.reduction_resize(
                f"{module}.{reduction_type}2({value}, {dim})"
            )
        return self.reduction_resize(f"{module}.{reduction_type}({value}, {dim})")

    def final_argreduce(buffer, result_var, value, index):
        buffer.splice(
            f"""\
            _, {result_var}_tmp = triton_helpers.{root_op}_with_index({value}, {index}, {dim})
            {result_var} = {self.reduction_resize(f'{result_var}_tmp')}
            """
        )

    cache_key = (src_dtype, reduction_type, value)
    if cache_key in self.cse.reduction_cache:
        return self.cse.reduction_cache[cache_key]

    dim = self.triton_tensor_ndim() - 1
    acc_type = triton_acc_type(src_dtype)
    result_var: Any = self.cse.newvar()
    result_var.mask_vars = OrderedSet(var for var in masks if var[0] != "r")
    cond = " & ".join(masks)

    def where_cond(tval, fval):
        if not cond:
            return tval
        return TritonKernelOverrides.where(cond, tval, fval)

    if self.persistent_reduction:
        default = ir.Reduction.default_value(reduction_type, src_dtype)
        default = self._map_tuple_or_scalar(constant_repr, default)

        def _mask_value(value, default):
            return self.cse.generate(self.compute, where_cond(value, default))

        if isinstance(value, tuple):
            masked_value = [_mask_value(v, d) for v, d in zip(value, default)]
        else:
            masked_value = _mask_value(value, default)

        if reduction_type in {"argmax", "argmin"}:
            accumulator_index = str(
                self.cse.generate(
                    self.compute,
                    f"tl.broadcast_to({reduction_range_prefix}index, {masked_value}.shape)",
                )
            )
            root_op = {"argmax": "max", "argmin": "min"}[reduction_type]
            final_argreduce(
                self.compute, result_var, masked_value, accumulator_index
            )
        elif reduction_type == "welford_reduce":
            # For persistent reductions, don't bother with
            # welford's algorithm since it uses more registers, and
            # taking two reductions doesn't increase memory usage.
            result_var = self.welford_reduce_fallback(dtype, value)
        elif reduction_type == "welford_combine":
            mean, m2, weight = masked_value
            welford = f"triton_helpers.welford({mean}, {m2}, {weight}, {dim})"
            mean, m2, weight = (self.cse.newvar() for _ in range(3))
            self.compute.writeline(f"{mean}, {m2}, {weight} = {welford}")

            result_var = tuple(
                self.cse.generate(self.compute, self.reduction_resize(var_name))
                for var_name in (mean, m2, weight)
            )
        else:
            result_var = self.cse.generate(
                self.compute, final_reduction(masked_value)
            )
    else:
        accumulator = f"_{result_var}" if multilane and not contraction else result_var
        default = ir.Reduction.default_accumulator(reduction_type, src_dtype)
        default = self._map_tuple_or_scalar(constant_repr, default)
        if not isinstance(default, tuple):
            if multilane and not contraction:
                self.body.writeline(
                    f"{accumulator} = tl.full({self.dense_size_str()}, {default}, {acc_type})"
                )
            else:
                if self.triton_tensor_ndim() == 1:
                    self.body.writeline(
                        f"{accumulator} = triton_helpers.promote_to_tensor({default})"
                    )
                else:
                    self.body.writeline(
                        f"{accumulator} = tl.full([{', '.join(self.dense_size_list()[:-1] + (['1'] if not contraction else []))}], {default}, {acc_type})"
                    )

        if reduction_type in {"argmax", "argmin"}:
            accumulator_index = f"_{result_var}_index"
            long_max = torch.iinfo(torch.int64).max
            self.body.writeline(
                f"{accumulator_index} = tl.full({self.dense_size_str()}, {long_max}, tl.int64)"
            )
            root_op = {"argmax": "max", "argmin": "min"}[reduction_type]

            self.compute.splice(
                f"""\
            {accumulator}_next, {accumulator_index}_next = triton_helpers.{root_op}imum_with_index(
                {accumulator}, {accumulator_index}, {value}, {reduction_range_prefix}index
            )
            {accumulator} = {where_cond(f'{accumulator}_next', accumulator)}
            {accumulator_index} = {where_cond(f'{accumulator_index}_next', accumulator_index)}
            """
            )
            final_argreduce(self.suffix, result_var, accumulator, accumulator_index)
        elif is_welford_reduction(reduction_type):
            accumulator = f"{result_var}_mean"
            accumulator_m2 = f"{result_var}_m2"
            accumulator_weight = f"{result_var}_weight"
            self.body.writeline(
                f"{accumulator} = tl.zeros({self.dense_size_str()}, {acc_type})"
            )
            self.body.writeline(
                f"{accumulator_m2} = tl.zeros({self.dense_size_str()}, {acc_type})"
            )
            self.body.writeline(
                f"{accumulator_weight} = tl.zeros({self.dense_size_str()}, {acc_type})"
            )

            if reduction_type == "welford_combine":
                mean, m2, weight = value
                self.compute.splice(
                    f"""\
                {accumulator}_next, {accumulator_m2}_next, {accumulator_weight}_next = triton_helpers.welford_combine(
                    {accumulator}, {accumulator_m2}, {accumulator_weight},
                    {mean}, {m2}, {weight}
                )
                """
                )
            else:
                assert reduction_type == "welford_reduce"
                self.compute.splice(
                    f"""\
                {accumulator}_next, {accumulator_m2}_next, {accumulator_weight}_next = triton_helpers.welford_reduce(
                    {value}, {accumulator}, {accumulator_m2}, {accumulator_weight}, roffset == 0
                )
                """
                )

            self.compute.splice(
                f"""\
            {accumulator} = {where_cond(f'{accumulator}_next', accumulator)}
            {accumulator_m2} = {where_cond(f'{accumulator_m2}_next', accumulator_m2)}
            {accumulator_weight} = {where_cond(f'{accumulator_weight}_next', accumulator_weight)}
            """
            )

            result_mean = result_var
            result_m2 = self.cse.newvar()
            result_weight = self.cse.newvar()
            self.suffix.splice(
                f"""\
            {result_mean}_tmp, {result_m2}_tmp, {result_weight}_tmp = triton_helpers.welford(
                {accumulator}, {accumulator_m2}, {accumulator_weight}, {dim}
            )
            {result_mean} = {self.reduction_resize(f'{result_mean}_tmp')}
            {result_m2} = {self.reduction_resize(f'{result_m2}_tmp')}
            {result_weight} = {self.reduction_resize(f'{result_weight}_tmp')}
            """
            )
            result_var = result_mean, result_m2, result_weight
        else:
            if modification is not None:
                updated_accumulator = self.overrides.modification(modification[0], accumulator, **modification[1])
                self.compute.writeline(
                        f"{accumulator} = {updated_accumulator}"
                    )

            if not multilane and reduction_type in {"max", "min"}:
                # triton_helpers.max2
                value = self.reduction_resize(getattr(ops, reduction_type)(value, dim))
            combine_fn = ir.get_reduction_combine_fn(reduction_type, src_dtype)
            updated = combine_fn(accumulator, value)

            if not writeback_later:
                if multilane and not contraction:
                    self.compute.writeline(
                        f"{accumulator} = {where_cond(updated, accumulator)}"
                    )
                else:
                    self.compute.writeline(
                        f"{accumulator} = {updated}"
                    )
            else:
                self.cse.reduction_cache[ops._unwrap(updated)] = result_var
                self.stores.writeline(
                    f"{accumulator} = {where_cond(updated, accumulator)}"
                )
            if multilane and not contraction:
                if src_dtype == torch.bool:
                    # This is only really used for aten.any. It changes the
                    # final reduction of a non-persistent reduction from
                    #     tmp5 = triton_helpers.max(_tmp5, 1)[:, None]
                    # to
                    #     tmp5 = triton_helpers.max(_tmp5.to(tl.int8), 1)[:, None].to(tl.int1)
                    # which is needed because tl.reduce doesn't support tl.int1
                    accumulator = f"{accumulator}.to(tl.int8)"
                    result_type = triton_compute_type(dtype)
                    self.suffix.writeline(
                        f"{result_var} = {final_reduction(accumulator)}.to({result_type})"
                    )
                else:
                    self.suffix.writeline(
                        f"{result_var} = {final_reduction(accumulator)}"
                    )

    self.cse.reduction_cache[cache_key] = result_var

    if isinstance(result_var, tuple):
        assert all(isinstance(x, TritonCSEVariable) for x in result_var)
        self.outside_loop_vars |= OrderedSet(result_var)
    else:
        assert isinstance(result_var, TritonCSEVariable)
        self.outside_loop_vars.add(result_var)

    return result_var if not writeback_later else ops._unwrap(updated)


@monkey.patch(SchedulerNode)
def codegen(self, index_vars) -> None:
    """
    + extend interpretation of reduction
    """
    @monkey.patch(V.ops)
    def reductionx(
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[CSEVariable, Tuple[CSEVariable, ...]],
        **kwargs
    ) -> Union[CSEVariable, Tuple[CSEVariable, ...]]:
        V.kernel.num_reduction += 1
        return V.kernel.reductionx(dtype, src_dtype, reduction_type, value, **kwargs)
    return monkey.fallback(self, index_vars)




from torch._inductor.loop_body import InterpreterShim
# from torch.fx.node import Argument, Target
# class DependencyGraphInterpreter(InterpreterShim):
#     def placeholder(self, target : 'Target', args : Tuple['Argument', ...], kwargs : dict[str, Any]) -> Any:
#         return super().placeholder(target, args, kwargs)

@contextlib.contextmanager
def set_subgraph_body(self, body_name: str):
    old_body, old_indexing_code, old_loads, old_compute, old_stores, old_suffix = self.body, self.indexing_code, self.loads, self.compute, self.stores, self.suffix
    yield
    self.body, self.indexing_code, self.loads, self.compute, self.stores, self.suffix = old_body, old_indexing_code, old_loads, old_compute, old_stores, old_suffix


@monkey.patch(TritonKernelOverrides)
# @classmethod
def modification(
    subgraph_id: int, accumulator, output_name: str = '', **producer_updated
) -> str:
    """
    This function is adapted from TritonTemplateKernel::modification in torch/_inductor/select_algorithm.py.

    This creates a modification function for a subgraph.
    """
    self: TritonKernel = V.kernel

    for key in producer_updated:
        if key not in self.cse.store_cache:
            return accumulator

    subgraph = getattr(TritonKernelOverrides, 'subgraphs')[subgraph_id]
    assert isinstance(subgraph, fx.Graph), f"{type(subgraph)} {repr(subgraph)}"
    # assert (
    #     self.body.getvalue() == ""
    # ), "Body should be clear before adding a modification"

    class OpsHandlerOverride(V.WrapperHandler):  # type: ignore[name-defined]
        """
        def recursively_copy_from_dependency_graph(node: fx.Node):
            if node.op == 'placeholder' and node.target == 'ops':
                dependency_node_remapping[node] = 'suffix_ops'
            if node.target == 'stale_partial_reduction':
                dependency_node_remapping[node] = decomposed['localbuf']
            if node.target == 'prev_ancestor_partial_reduction':
                dependency_node_remapping[node] = decomposed_ancestors[
                    node.args[0]]['localbuf']
            if node.op == 'call_method' and node.target == 'load':
                dependency_node_remapping[node] = decomposed_ancestors[
                    name_of_load(node.meta['origin'])]['finalreduce']
        """
        # self.name = 'PlaceholderSubstitution_dependency'

        def load(self, name: str):
            # if not name:
            #     return accumulator
            if name.startswith('stale_'):
                assert name[6:] in producer_updated, f"{name[6:]} not in {producer_updated}"
                name = name[6:]
                updated = producer_updated[name]
                assert updated in V.kernel.cse.reduction_cache, f"{updated} not in {V.kernel.cse.reduction_cache}"
                return V.kernel.cse.reduction_cache[updated]
            assert name in producer_updated, f"{name} not in {producer_updated}"
            return producer_updated[name]

    with V.set_ops_handler(OpsHandlerOverride(V.ops)):
        out = InterpreterShim(subgraph, submodules={}).run(V.ops, accumulator)

    return out




# helper
def find_unique_node(graph: fx.Graph, *, op, target) -> fx.Node:
    candidates = graph.find_nodes(op=op, target=target)
    assert len(candidates) == 1, f"{op, target} with multiple matches {candidates}"
    return next(iter(candidates))

REDUCTION_TARGET = {'reduction', 'reductionx'}

def find_unique_reduction(graph: fx.Graph) -> fx.Node:
    """Default to
        find_unique_node(graph, op='call_method', target='reduction')
    Allow custom reduction registered in `REDUCTION_TARGET`.
    """
    for rtarget in REDUCTION_TARGET:
        candidates = graph.find_nodes(op='call_method', target=rtarget)
        if not candidates:
            continue
        assert len(candidates) == 1, f"multiple {rtarget}: {candidates}"
        return next(iter(candidates))
    raise ValueError(f"no reduction node found ({REDUCTION_TARGET})")

# node helpers: instruction parsers
def get_method_name(node: fx.Node):
    assert node.op == 'call_method', f"{node.op=} {node.target=}"
    if node.target in REDUCTION_TARGET:
        return type_of_reduction(node)
    return node.graph._target_to_str(node.target)

def flatten_args(node: fx.Node):
    return tuple(itertools.chain.from_iterable(flatten_args(a) if a in node.all_input_nodes else (a,) for a in node.args))

def name_of_index(get_index: fx.Node) -> str:
    # assert get_index.op == 'call_module' and get_index.target =='get_index'
    return get_index.args[0]  # def get_index(self, name): ...

def name_of_load(load: fx.Node) -> str:
    # assert load.op == 'call_method' and load.target =='load'
    return load.args[1]  # def load(self, name: str, index: sympy.Expr): ...

def name_of_store_reduction(store_reduction: fx.Node) -> str:
    # assert store_reduction.op == 'call_method' and store_reduction.target == 'store_reduction'
    return store_reduction.args[1]  # def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable): ...

def reduction_to_store(store_reduction: fx.Node) -> fx.Node:
    # assert store_reduction.op == 'call_method' and store_reduction.target == 'store_reduction'
    return store_reduction.args[3]  # def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable): ...

from torch._inductor.ops_handler import ReductionType
def type_of_reduction(reduction: fx.Node) -> ReductionType:
    """
    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
    --> reduction_type: ReductionType,
        value: Union[CSEVariable, Tuple[CSEVariable, ...]],
    )
    """
    # assert reduction.op == 'call_method' and reduction.target == 'reduction'
    return reduction.args[3]


# helper
def trace_dag_with_sink(sink: fx.Node, frontier: OrderedSet[fx.Node]) -> OrderedSet[fx.Node]:
    """
    If there is a DAG with `frontier` as sources and `sink` as the only sink node, return the nodes in the dag.
    Otherwise, return empty set.
    """
    if not frontier:
        return frontier
    extension = OrderedSet()
    for src in frontier:
        if src is sink:
            continue
        if not src.users:  # sink in non-sink
            return OrderedSet()
        res = trace_dag_with_sink(sink, OrderedSet(src.users))
        if not res:
            return res
        extension |= res
    return frontier | extension


########
# Homomorphic transformations
########

def homomorphic_transform(domain_node: fx.Node, hom_node: fx.Node, codomain_funcname: str):
    """
    If `hom_node` is a homomorphism from some previous operation (`domain_operation`)
    to one of the `codomain_hints` operations (`codomain_operation`), then
        hom ∘ domain_operation = codomain_operation ∘ hom
    so we can transform `hom` to `codomain_operation` as the last/outer-most op.

    op: domain operation
    coop: codomain operation
    hom: homomorphism
    """
    # domop-hom -> hom-codomop
    # 1. broadcast hom for all inputs of domain operation (-hom -domop-hom-)
    # 2. insert codomain node (-hom-codomop -domop-hom-)
    # 3. replace all uses of original hom with codomain operation (-hom-codomop-)
    codomain_node_args = [domain_node.args[0]]  # keep `ops` handler
    with domain_node.graph.inserting_before(domain_node):
        for domain_arg in domain_node.args[1:]:
            new_hom_node = hom_node.graph.node_copy(hom_node, arg_transform=
                lambda hom_arg: hom_arg if hom_arg in domain_node.args else domain_arg)
            codomain_node_args.append(new_hom_node)
    assert not domain_node.kwargs, NotImplementedError(f"TODO {domain_node.kwargs=}")
    codomain_node_kwargs = domain_node.kwargs
    with hom_node.graph.inserting_after(hom_node):
        codomain_node = domain_node.graph.create_node(
            op=hom_node.op,  # 'call_method'
            target=codomain_funcname,
            args=tuple(codomain_node_args),
            kwargs=codomain_node_kwargs,
            type_expr=hom_node.type
        )
        codomain_node.meta = copy.copy(domain_node.meta)
    hom_node.replace_all_uses_with(codomain_node)
    hom_node.graph.erase_node(hom_node)
    assert not domain_node.users, f"domain operation has dependants {tuple(domain_node.users.keys())}"
    domain_node.graph.erase_node(domain_node)
    return codomain_node


def inverse_homomorphic_transform(hom_node: fx.Node, codomain_node: fx.Node, domain_reduction_type: str):
    """
    op: domain operation
    coop: codomain operation
    hom: homomorphism
    """
    # hom-codomop -> domop-hom
    # 1. Create a domain operation node (-domop -hom-codomop-)
    # 2. insert hom node (-domop-hom -hom-codomp-)
    # 3. Replace all uses of codomain_node with the new hom node (-domop-hom-)

    # Extract arguments from codomain_node, skipping the ops handler (first arg)
    domain_node_args = []
    for codomain_arg in codomain_node.args:
        to_dtype = None
        if getattr(codomain_arg, 'target', '') == 'to_dtype':  # ad hoc fix
            to_dtype = codomain_arg
            codomain_arg = to_dtype.args[1]
        if codomain_arg == get_method_name(codomain_node):
            domain_node_args.append(domain_reduction_type)
            continue
        if codomain_arg is hom_node:
            if to_dtype:
                to_dtype.replace_input_with(hom_node, hom_node.args[1])
                domain_node_args.append(to_dtype)
                continue
            domain_node_args.append(hom_node.args[1]) # skip `ops` handler TODO this is specific to dividend (vs divisor)
            continue
        domain_node_args.append(codomain_arg)
    # Create a new domain node
    with hom_node.graph.inserting_before(codomain_node):
        domain_node = codomain_node.graph.create_node(
            op=codomain_node.op,  # 'call_method'
            target=codomain_node.target,  # 'reduction'
            args=tuple(domain_node_args),
            kwargs=codomain_node.kwargs,
            type_expr=hom_node.type
        )
        domain_node.meta = copy.copy(codomain_node.meta)

    # Create a new hom node applied to the domain node's output
    with domain_node.graph.inserting_after(domain_node):
        new_hom_node = codomain_node.graph.node_copy(
            hom_node,
            arg_transform=lambda x: domain_node if x in [
                    arg.args[1] if getattr(arg, 'target', '') == 'to_dtype' else arg for arg in domain_node.args[1:]
                ] else x
        )

    # Replace all uses of codomain_node with new_hom_node
    codomain_node.replace_all_uses_with(new_hom_node)
    codomain_node.graph.erase_node(codomain_node)
    assert not hom_node.users, f"homomorphism has dependants {tuple(hom_node.users.keys())}"
    hom_node.graph.erase_node(hom_node)
    return domain_node


"""
mul (R, add) (R, add)   # distributive property
exp (R, add) (R+, mul)
log (R+, mul) (R, add)
"""
HOMOMORPHISM_OPTIONS = {
    # sub = add ∘ neg
    # div = mul ∘ recip
    ('add', 'add'): ['neg', 'mul', 'truediv'],  # distributive property
    ('mul', 'mul'): ['recip', 'pow'],  # power funciton properties
    ('add', 'mul'): ['exp'],  # exp ∘ add = mul ∘ exp
    ('mul', 'add'): ['log'],  # log ∘ mul = add ∘ log
    ('neg', 'recip'): ['exp'], # exp ∘ neg = recip ∘ exp
    ('recip', 'neg'): ['log'], # log ∘ recip = neg ∘ log
    ('sub', 'truediv'): ['exp'],  # exp ∘ sub = exp ∘ add ∘ neg = mul ∘ exp ∘ neg = mul ∘ recip ∘ exp = div ∘ exp
    ('truediv', 'sub'): ['log'],  # log ∘ div = log ∘ mul ∘ recip = add ∘ log ∘ recip = add ∘ neg ∘ log = sub ∘ log
}

# {h: {f: g}} such that h(f(...)) = g(h(...))
HOMOMORPHIC_TRANSFORMATIONS: dict[tuple[str, str], tuple[str, str, Callable]] = {
    ('exp', 'add'): ('mul', 'exp', homomorphic_transform),
    ('mul', 'exp'): ('exp', 'add', inverse_homomorphic_transform),
    ('exp', 'sum'): ('prod', 'exp', homomorphic_transform),
    ('prod', 'exp'): ('exp', 'sum', inverse_homomorphic_transform),
    ('exp', 'neg'): ('recip', 'exp', homomorphic_transform),
    ('recip', 'exp'): ('exp', 'neg', inverse_homomorphic_transform),
    ('exp', 'sub'): ('truediv', 'exp', homomorphic_transform),
    ('truediv', 'exp'): ('exp', 'sub', inverse_homomorphic_transform),
    ('log', 'mul'): ('add', 'log', homomorphic_transform),
    ('add', 'log'): ('log', 'mul', inverse_homomorphic_transform),
    ('log', 'prod'): ('sum', 'log', homomorphic_transform),
    ('sum', 'log'): ('log', 'prod', inverse_homomorphic_transform),
    ('log', 'recip'): ('neg', 'log', homomorphic_transform),
    ('neg', 'log'): ('log', 'recip', inverse_homomorphic_transform),
    ('log', 'truediv'): ('sub', 'log', homomorphic_transform),
    ('sub', 'log'): ('log', 'truediv', inverse_homomorphic_transform),
    ('neg', 'add'): ('add', 'neg', homomorphic_transform),
    ('add', 'neg'): ('neg', 'add', inverse_homomorphic_transform),
    ('mul', 'add'): ('add', 'mul', homomorphic_transform),
    ('add', 'mul'): ('mul', 'add', inverse_homomorphic_transform),
    ('truediv', 'add'): ('add', 'truediv', homomorphic_transform),
    ('add', 'truediv'): ('truediv', 'add', inverse_homomorphic_transform),
    ('truediv', 'sum'): ('sum', 'truediv', homomorphic_transform),
    ('sum', 'truediv'): ('truediv', 'sum', inverse_homomorphic_transform),
    ('truediv', 'dot'): ('dot', 'truediv', homomorphic_transform),
    ('dot', 'truediv'): ('truediv', 'dot', inverse_homomorphic_transform),
    ('recip', 'mul'): ('mul', 'recip', homomorphic_transform),
    ('mul', 'recip'): ('recip', 'mul', inverse_homomorphic_transform),
    ('pow', 'mul'): ('mul', 'pow', homomorphic_transform),
    ('mul', 'pow'): ('pow', 'mul', inverse_homomorphic_transform),
}

# {g: {h: f}} such that g(h(...)) = h(f(...))
HOMOMORPHISM_TO: dict[str|ReductionType, dict[str, str|ReductionType]] = {
    'sum': {
        'log': 'prod',  # sum(log(x)...) = log(prod(x...))
        'mul': 'sum', # sum((x * c)...) = sum(x...) * c
        'truediv': 'sum', # sum((x / c)...) = sum(x...)) / c
    },
    'prod': {
        'exp': 'sum',  # prod(exp(x)...) = exp(sum(x...))
        'pow': 'prod',  # prod((x^c)...) = prod(x...)^c
    },
    'min': {  # TODO monotonic functions
        'add': 'min', # min(x + c...) = min(x...) + c
        'exp': 'min', # min(exp(x)...) = exp(min(x...))
        'neg': 'max', # min(-x...) = -max(x...)
        # TODO for c >= 0:
        # 'mul': 'min', # min(c * x...) = c * min(x...)
        # 'log': 'min', # min(log(x)...) = log(min(x...))
        # 'recip': 'max', # min(1/x...) = 1 / max(x...)
    },
    'max': {  # TODO monotonic functions
        'add': 'max', # max(x + c...) = max(x...) + c
        'exp': 'max', # max(exp(x)...) = exp(max(x...))
        'neg': 'min', # max(-x...) = -min(x...)
        # TODO for c >= 0:
        # 'mul': 'max', # max(c * x...) = c * max(x...)
        # 'log': 'max', # max(log(x)...) = log(max(x...))
        # 'recip': 'min', # max(1/x...) = 1 / min(x...)
    },
}

# `truediv` is the name used in ops_handler
INVERSE = {
    'add': 'sub',
    'sub': 'add',   # when subtrahen is constant; TODO 'sub': 'sub' when minuend is constant
    'mul': 'truediv',
    'truediv': 'mul',   # when divisor is constant; TODO 'truediv': 'truediv' when dividend is constant
}



def try_hoist(node: fx.Node):
    """
    f:A -> B is a homomorphism b/w (A,∗) and (B,∘) such that f(a_0 ∗ a_1) = f(a_0) ∘ f(a_1)

    Let `g` be the binary function of reduction, `f: R -> R` be a homomorphism b/w (R, g)
    and (R, h) where `h` has same associative property with `g`. By definition, for scalars
        f(g(a, b)) = h(f(a), f(b))
    Generally, for an input sequence `x`,
        f(reduce(g, x)) = reduce(h, map(f, x))
    The goal of this transformation is to hoist reductions (from rhs to lhs).
    -> given h and f, find g
    """
    # only handles a chains of ops
    pred_calls = [n for n in node.all_input_nodes if n.op == 'call_method' and n.target != 'load']
    if len(pred_calls) != 1:
        return False
    pred_call = pred_calls[0]
    if len(pred_call.users) != 1:
        return False
    arg_index = node.args.index(pred_call)

    # ad hoc fix
    if pred_call.target == 'to_dtype':
        pred_call = pred_call.args[1]

    m2, m1 = get_method_name(node), get_method_name(pred_call)
    while (m2, m1) not in HOMOMORPHIC_TRANSFORMATIONS:
        if try_hoist(pred_call):
            new_pred_call = node.args[arg_index]
            assert len(new_pred_call.users) == 1 and (node in new_pred_call.users), f"{new_pred_call.op=} {new_pred_call.target=} {new_pred_call.users=}"
            pred_call = new_pred_call
            # ad hoc fix
            if pred_call.target == 'to_dtype':
                pred_call = pred_call.args[1]
            m1 = get_method_name(pred_call)
            continue
        return False

    n2, n1, transform = HOMOMORPHIC_TRANSFORMATIONS[m2, m1]
    return transform(pred_call, node, n2)



def find_dep_store_load(anc_graph: fx.Graph, graph: fx.Graph, buf_map = {}):
    """
                                    non-dep-load
                                        |
    <anc_graph> dep-store -> dep-load <graph>
      |
      non-dep-store
    """
    dep_load = dict[str, fx.Node]()
    non_dep_loads = dict[str, tuple[fx.Node]]()
    dep_store = dict[str, fx.Node]()
    non_dep_store = dict[str, fx.Node]()

    # all stores of anc_graph
    anc_reduction_stores: dict[str, fx.Node] = {
        name_of_store_reduction(store_reduction): store_reduction
        for store_reduction in anc_graph.find_nodes(op='call_method', target='store_reduction', sort=True)
    }

    # all loads of graph
    all_loads = collections.defaultdict[str, OrderedSet[fx.Node]](OrderedSet[fx.Node])
    for ld in graph.find_nodes(op='call_method', target='load', sort=True):
        all_loads[name_of_load(ld)].add(ld)
    all_loads = {k: tuple(v) for k, v in all_loads.items()}

    # triage
    for st_name in anc_reduction_stores:
        ld_names = {st_name} if st_name not in buf_map else buf_map[st_name]
        for ld_name in ld_names:
            if ld_name in all_loads:
                assert len(all_loads[ld_name]) == 1, f"multiple loads of {ld_name} {all_loads[ld_name]}"
                dep_load[ld_name] = next(iter(all_loads[ld_name]))
                dep_store[st_name] = anc_reduction_stores[st_name]
                continue
            non_dep_store[st_name] = anc_reduction_stores[st_name]

    for ld_name in all_loads:
        if ld_name not in dep_load:
            non_dep_loads[ld_name] = all_loads[ld_name]
    
    return dep_store, non_dep_store, dep_load, non_dep_loads

class Sympifier:
    def __init__(self, inner=V.MockHandler()):
        # super().__init__(inner)
        self.bufs = dict()
        self.args = dict()

    def __getattr__(self, name):
        def op(*args, **kwargs):
            mock_fn = getattr(V.MockHandler(), name)
            mock_out = mock_fn(*args, **kwargs)
            res = sympy.sympify(mock_out, locals={**self.bufs, **self.args})
            return res
        return op

    def exp(self, x):
        return sympy.E ** x

    def load(self, name: str, *_):
        s = sympy.Symbol(name)
        if name.startswith('buf'):
            self.bufs[name] = s
        if name.startswith('arg'):
            self.args[name] = s
        return s

class NodeRemapping(collections.UserDict[fx.Node, fx.Node]):
    def __setitem__(self, src: fx.Node, dst: fx.Node):
        dst.meta['origin'] = src.meta.get('origin', src)
        super().__setitem__(src, dst)


def can_eliminate_reduction_dependency(graph: fx.Graph, anc_graph: fx.Graph, buf_map={}, shared_reads=OrderedSet()):
    """ancestor reduction --> dependent reduction"""
    dependent_store, non_dependent_store, dependency_loads, non_dependency_loads = find_dep_store_load(anc_graph, graph, buf_map)
    assert dependency_loads, "dependency not found"
    reduction: fx.Node = find_unique_reduction(graph)
    # 1. build dependency graph
    dependency_graph: fx.Graph = fx.Graph()
    dag_node_remapping = NodeRemapping()
    # copy args
    def arg_recursive_copy(arg: fx.Node):
        if arg not in dag_node_remapping:
            dag_node_remapping[arg] = dependency_graph.node_copy(arg, arg_transform=arg_recursive_copy)
        return dag_node_remapping[arg]
    def arg_as_placeholder(arg: fx.Node):
        if arg not in dag_node_remapping:
            with dependency_graph.inserting_after(next(iter(dependency_graph.nodes))):
                dag_node_remapping[arg] = dependency_graph.placeholder('stale_partial_reduction', type_expr=arg.type)
        return dag_node_remapping[arg]
    for name, lds in non_dependency_loads.items():
        # if name.startswith('buf'):
        #     continue
        # assert name.startswith('arg')
        for ld in lds:
            dag_node_remapping[ld] = dependency_graph.node_copy(ld, arg_transform=arg_recursive_copy)
    # trace dependencies
    for name, ld in dependency_loads.items():
        dag_nodes = trace_dag_with_sink(reduction, OrderedSet([ld]))
        if not dag_nodes:
            continue
        # copy load and their args
        if ld.args[0] not in dag_node_remapping:
            dag_node_remapping[ld.args[0]] = dependency_graph.node_copy(ld.args[0])  # ops handler
        dag_node_remapping[ld] = dependency_graph.create_node(
            op=ld.op, target=ld.target, args=(  # updated_ancestor_partial_reduction
                dag_node_remapping[ld.args[0]],
                # 'updated_' + 
                name_of_load(ld),
        ), type_expr=ld.type)
        # copy other nodes and their args
        for node in dag_nodes:
            if node in dag_node_remapping:
                continue
            dag_node_remapping[node] = dependency_graph.node_copy(
                node,
                arg_transform=arg_as_placeholder
            )
    assert dag_node_remapping, f"no dependency path to {reduction} found"
    assert reduction in dag_node_remapping, f"{reduction} not in {dag_node_remapping}"
    # save a handle to the last node before transformations
    output_handle = dependency_graph.output(dag_node_remapping[reduction])
    # dependency graph built

    # 2. try hoist reduction to eliminate dependency
    reduction_ = dag_node_remapping[reduction]
    while reduction_:= try_hoist(reduction_):
        dag_nodes = trace_dag_with_sink(reduction_, OrderedSet(dag_node_remapping[ld] for ld in dependency_loads.values()))
        if not dag_nodes:
            break
    if not reduction_:
        return False  # dependency cannot be eliminated

    # 3. trace dependency graph to formulate update function
    dag_nodes_from_ld = trace_dag_with_sink(output_handle, OrderedSet(dag_node_remapping[ld] for ld in dependency_loads.values()))
    dag_nodes_from_red = trace_dag_with_sink(output_handle, OrderedSet({reduction_}))
    dag_merge_to_output = dag_nodes_from_ld & dag_nodes_from_red
    merge_node = next(iter(dag_merge_to_output))
    
    arg_index, dep_index = 0, 0
    for i, arg in enumerate(merge_node.args):
        if arg in dag_nodes_from_red:
            arg_index = i
        if arg in dag_nodes_from_ld:
            dep_index = i
    assert arg_index and dep_index, f"{arg_index=} {dep_index=}"

    with dependency_graph.inserting_before(merge_node):
        def recursively_copy_pre_merge_ops(arg_node: fx.Node):
            assert arg_node.graph is dependency_graph
            if arg_node.op == 'placeholder':
                return arg_node
            if arg_node.meta['origin'] in dependency_loads.values():
                name = name_of_load(arg_node.meta['origin'])
                return dependency_graph.create_node(
                    op='call_method', target='load', args=(  # prev_ancestor_partial_reduction
                        arg_node.args[0],
                        'stale_' + name,
                    ), type_expr=arg_node.type)
            return dependency_graph.node_copy(arg_node, arg_transform=recursively_copy_pre_merge_ops)

        restored = merge_node.args[arg_index]
        for to_inverse in reversed(list[fx.Node](dag_merge_to_output)):
            if to_inverse == output_handle:
                continue
            target_to_inverse = get_method_name(to_inverse)
            if target_to_inverse not in INVERSE:
                return False  # not invertible
            inverse_target = INVERSE[target_to_inverse]
            restored = dependency_graph.create_node(
                op=to_inverse.op, target=inverse_target,
                args=tuple(restored if x in dag_nodes_from_red else (
                        recursively_copy_pre_merge_ops(x) if isinstance(x, fx.Node) else x
                    ) for x in to_inverse.args),
                kwargs=to_inverse.kwargs
            )
        merge_node.replace_input_with(merge_node.args[arg_index], restored)
    with dependency_graph.inserting_before(reduction_):
        reduction_.replace_all_uses_with(dependency_graph.create_node(
            # op='call_method', target='load', args=(reduction_.args[0], 'localbuf'),  # consumer accumulator
            op='placeholder', target='', args=(reduction_.args[0],),  # consumer accumulator as input
        ))
    for node in reversed(dependency_graph.nodes):
        if node.op != "output" and len(node.users) == 0:
            dependency_graph.erase_node(node)
    # update function graph built

    # simplify with sympy
    with V.set_ops_handler(Sympifier()):
        out = InterpreterShim(dependency_graph, submodules={}).run(V.get_ops_handler(), sympy.Symbol("localbuf"))
    simplified = sympy.simplify(out)
    if simplified != out:
        dependency_graph = sympy_expr_to_fx_graph(simplified)

    return dependency_graph

def sympy_expr_to_fx_graph(expr: sympy.Expr) -> fx.Graph:
    """
    Converts a Sympy expression into a torch.fx.Graph.

    This function recursively traverses the Sympy expression tree and constructs
    an equivalent fx.Graph. It's a foundational step for JIT compilation
    or symbolic-to-executable code generation.

    Features:
    - Sympy Symbols are converted to `placeholder` nodes.
    - Sympy numeric literals become literal arguments in `call_function` nodes.
    - Sympy operations (Add, Mul, Pow, sin, etc.) are converted to
      `call_function` nodes targeting equivalent callables.
    - Division (e.g., `x/y`) is correctly translated from `Mul(x, Pow(y, -1))`.
    - Variadic Sympy ops (e.g., `a+b+c`) are chained into binary calls.

    Args:
        expr: The Sympy expression to convert.

    Returns:
        A `torch.fx.Graph` representing the computation.

    Raises:
        NotImplementedError: If the expression contains an unsupported Sympy function.
        ValueError: If the expression simplifies to a constant, as it cannot be the
                    sole output of a graph without inputs.
    """
    # Create an empty FX graph
    graph = fx.Graph()
    ops_node = graph.placeholder('ops')

    from functools import cache
    from torch.fx.node import Argument

    @cache
    def _build_recursive(sub_expr: sympy.Expr) -> Argument:
        """
        Recursively traverses the expression, builds nodes, and returns the
        resulting fx.Node or literal value for the given sub-expression.
        """
        # symbols
        if sub_expr.is_Symbol:
            if sub_expr.name == 'localbuf':
                node = graph.placeholder(sub_expr.name)
                return node
            node = graph.call_method('load', (ops_node, sub_expr.name))
            return node

        # numeric literal
        if sub_expr.is_number:
            val = float(sub_expr) if getattr(sub_expr, 'is_Float', False) else int(sub_expr)
            return val
        
        # handle sub and (true)div
        if sub_expr.is_Add or sub_expr.is_MatAdd:
            minuend_terms = []
            subtrahend_terms = []

            for arg in sub_expr.args:
                if (arg.is_Mul or arg.is_MatMul) and sympy.Integer(-1) in arg.args:
                    args = list(arg.args)
                    args.remove(sympy.Integer(-1))
                    if len(args) == 1:
                        subtrahend_terms.append(args[0])
                    else:
                        subtrahend_terms.append(sympy.Mul(*args))
                else:
                    minuend_terms.append(arg)

            if subtrahend_terms:
                if not minuend_terms:
                    minuend_node = 0.0  # -Add(...)
                elif len(minuend_terms) == 1:
                    minuend_node = _build_recursive(minuend_terms[0])
                else:
                    minuend_node = _build_recursive(sympy.Mul(*minuend_terms))

                if len(subtrahend_terms) == 1:
                    subtrahend_node = _build_recursive(subtrahend_terms[0])
                else:
                    subtrahend_node = _build_recursive(sympy.Mul(*subtrahend_terms))

                node = graph.call_method('sub', (ops_node, minuend_node, subtrahend_node))
                return node
        # TODO @bozhiyou use sympy.fraction(sub_expr) (see sympy.count_op)
        elif sub_expr.is_Mul or sub_expr.is_MatMul:
            numer_terms = []
            denom_terms = []

            for arg in sub_expr.args:
                if isinstance(arg, sympy.Pow) and arg.args[1].is_negative:
                    base, exp = arg.args
                    if exp == -1:
                        denom_terms.append(base)
                    else:
                        denom_terms.append(sympy.Pow(base, -exp))
                else:
                    numer_terms.append(arg)

            if denom_terms: # construct a division operation
                # Build the numerator node
                if not numer_terms:
                    numer_node = 1.0  # 1/Mul(...)
                elif len(numer_terms) == 1:
                    numer_node = _build_recursive(numer_terms[0])
                else:
                    numer_node = _build_recursive(sympy.Mul(*numer_terms))

                # Build the denominator node
                if len(denom_terms) == 1:
                    denom_node = _build_recursive(denom_terms[0])
                else:
                    denom_node = _build_recursive(sympy.Mul(*denom_terms))

                # Create the final division node
                node = graph.call_method('truediv', (ops_node, numer_node, denom_node))
                return node

        op_func = sub_expr.func
        arg_nodes = tuple(_build_recursive(arg) for arg in sub_expr.args)

        if not arg_nodes:
            if hasattr(op_func, 'identity'):
                return op_func.identity
            raise NotImplementedError(f"{type(sub_expr)=} {sub_expr}")
        
        target_name = op_func.__name__.lower()  # TODO @bozhiyou handle special names

        if len(arg_nodes) > 2 and issubclass(op_func, sympy.core.operations.AssocOp):
            # Chain the operations: e.g., a+b+c -> add(add(a, b), c)
            node = arg_nodes[0]
            for i in range(1, len(arg_nodes)):
                node = graph.call_method(target_name, (ops_node, node, arg_nodes[i]))
            return node

        # default
        node = graph.call_method(target_name, (ops_node,) + arg_nodes)
        return node

    result_node = _build_recursive(expr)

    assert isinstance(result_node, fx.Node), f"graph output must be a fx.Node {type(result_node)=} {result_node}"
    graph.output(result_node)

    graph.lint()

    return graph


def prepare_fusion(dependency_graph: fx.Graph, graph: fx.Graph, anc_graph: fx.Graph) -> None:
    dependent_store, non_dependent_store, dependency_loads, non_dependency_loads = find_dep_store_load(anc_graph, graph)

    anc_reduction = find_unique_reduction(anc_graph)
    anc_reduction.target = 'reductionx'
    if 'writeback_later' not in anc_reduction.kwargs:
        anc_reduction.kwargs = dict(**anc_reduction.kwargs, writeback_later=True, multilane=False)

    subgraphs = getattr(TritonKernelOverrides, 'subgraphs', {})
    dependency_graph_id = len(subgraphs)
    subgraphs[dependency_graph_id] = dependency_graph
    setattr(TritonKernelOverrides, 'subgraphs', subgraphs)

    reduction: fx.Node = find_unique_reduction(graph)
    reduction.target = 'reductionx'
    assert 'modification' not in reduction.kwargs, f"existing reduction dependency: {reduction.kwargs['modification']}"
    reduction.kwargs = dict(**reduction.kwargs, modification=(dependency_graph_id, dependency_loads))


######
# Enable fusion: pattern extensions
######

def mark_one_shot_range(self: SchedulerNode, prefix: sympy.Integer, range_size: sympy.Integer):
    """
    Node level one shot range
    """
    one_shot_ranges: dict[sympy.Integer, sympy.Integer] = getattr(self, 'one_shot', {})
    one_shot_ranges[prefix] = range_size
    setattr(self, 'one_shot', one_shot_ranges)


@monkey.patch(SchedulerNode)
def get_ranges(self: SchedulerNode, one_shot_ranges: dict[sympy.Integer, sympy.Integer] = {}):
    """
    - filter out one-shot ranges
    """
    ranges, rranges = monkey.fallback(self)
    one_shot_ranges.update(getattr(self, 'one_shot', {}))
    if not one_shot_ranges:
        return ranges, rranges
    active_one_shot_ranges = {}
    prefix = sympy.Integer(1)
    non_trivial_ranges = []
    for ran9e in ranges:
        if prefix in one_shot_ranges and ran9e == one_shot_ranges[prefix]:
            active_one_shot_ranges[prefix] = ran9e
            prefix *= ran9e
            continue
        prefix *= ran9e
        non_trivial_ranges.append(ran9e)
    setattr(self, 'one_shot', active_one_shot_ranges)
    return non_trivial_ranges, rranges


class ReductionDependency:
    reduction_glue_functions = dict[tuple[SchedulerNode, SchedulerNode], Callable]()

    @staticmethod
    def get_reduction_reads(node: BaseSchedulerNode, buf_map: dict[str, set[str]] = {}) -> OrderedSet[str]:
        """
        Recursively finds the ultimate Reduction nodes that a given node reads from.
        If a node's dependency chain only consists of Pointwise nodes, return those
        initial Pointwise dependencies.
        """
        reads = OrderedSet()
        for r in node.read_writes.reads:
            if not r.name.startswith('buf'):
                continue
            op = node.scheduler.name_to_buf[r.name].defining_op
            if op.is_reduction():
                reads.add(op.get_name())
                buf_map[r.name] = set([r.name])
                continue
            recursive_buf_map: dict[str, set[str]] = {}
            recursive_reads = ReductionDependency.get_reduction_reads(op, recursive_buf_map)
            if recursive_reads:
                reads |= recursive_reads
                for wr in recursive_buf_map:
                    if wr in buf_map:
                        buf_map[wr].add(r.name)
                        continue
                    buf_map[wr] = set([r.name])
            else:
                reads.add(op.get_name())
                buf_map[r.name] = set([r.name])
        return reads

    @staticmethod
    def capture(node1: BaseSchedulerNode, node2: SchedulerNode):
        buf_map = {}
        dep_ops = ReductionDependency.get_reduction_reads(node2, buf_map)
        anc_reds = OrderedSet(
                n.get_name() for n in node1.get_nodes() if n.is_reduction()
            )
        anc_names = anc_reds & dep_ops
        # assert anc_names == dep_ops
        anc_nodes = [node1.scheduler.name_to_node[name] for name in anc_names]
        for anc_node in anc_nodes:
            assert isinstance(anc_node, SchedulerNode), anc_node
            if (anc_node, node2) in ReductionDependency.reduction_glue_functions:
                continue
            # if any(r.name not in (rw.name for rw in anc_node.read_writes.reads_and_writes()) for r in node2.read_writes.reads if r.name.startswith('buf')):
            if any(r not in ReductionDependency.get_reduction_reads(anc_node) for r in dep_ops if r != anc_node.get_name()):
                continue  # node2 reads that are not dep of node1
            # ReductionDependency.try_eliminate(anc_node, node2)
            if dependency_graph:= can_eliminate_reduction_dependency(
                node2._body.root_block.graph,
                anc_node._body.root_block.graph,
                buf_map = buf_map,
                shared_reads=node1.read_writes.reads & node2.read_writes.reads,
            ):
                assert (node1, node2) not in ReductionDependency.reduction_glue_functions, "multiple dependencies"
                ReductionDependency.reduction_glue_functions[node1, node2] = lambda: prepare_fusion(
                    dependency_graph,
                    node2._body.root_block.graph,
                    anc_node._body.root_block.graph,
                )
                fusion_log.debug(f"\033[33mfuse\033[0m {node1.get_name()} {node1.group[1]} with {node2.get_name()} {node2.group[1]}: dependent reduction")

    @staticmethod
    def try_eliminate(node1: BaseSchedulerNode, node2: SchedulerNode):
        n1 = node1.node.get_origin_node()
        n2 = node2.node.get_origin_node()
        graph = n1.graph
        assert n2.graph is graph

        # find dag
        dag_nodes = OrderedSet[fx.Node]()
        def _dfs(src, path):
            nonlocal dag_nodes, n2
            if src is n2:
                dag_nodes |= path
                return
            for u in src.users:
                _dfs(u, OrderedSet([*path, src]))
        _dfs(n1, OrderedSet())
        dag_nodes.add(n2)

        # copy graph
        dependency_graph: fx.Graph = fx.Graph()
        dag_node_remapping = {}
        dummy_inputs = []
        def arg_as_placeholder(arg: fx.Node):
            if arg not in dag_node_remapping:
                with dependency_graph.inserting_after():
                    dag_node_remapping[arg] = dependency_graph.placeholder(arg.name, type_expr=arg.type)
                dummy_inputs.append(f'input_{arg.name}_{len(dummy_inputs)}')
            return dag_node_remapping[arg]
        for node in dag_nodes:
            if node not in dag_node_remapping:
                dag_node_remapping[node] = dependency_graph.node_copy(
                    node, arg_transform=arg_as_placeholder
                )

        # trace dag
        # LoopBodyBlock.__init__
        tracer = fx.Tracer()
        tracer.graph = fx.Graph(tracer_cls=tracer.__class__)
        proxy_ops = tracer.create_proxy("placeholder", "ops", (), {})
        handler = V.WrapperHandler(proxy_ops)
        with V.set_ops_handler(handler):
            # V.graph.run(*(V.graph.example_inputs))
            # InterpreterShim(dependency_graph, {}).run(*dummy_inputs)
            ops.output(n2.get_store_function()(*args))
        graph = tracer.graph
        pass



def _ranges_fit_allow_one_shot(node1: BaseSchedulerNode, node2: SchedulerNode) -> bool:
    """check if node ranges fit when one-shot range is allowed"""
    _, (numel1, rnumel1) = node1.group
    _, (numel2, rnumel2) = node2.group

    ranges2, _ = node2.get_ranges()
    accumulated = sympy.Integer(1)
    one_shot_ranges = []
    for r in ranges2:
        if accumulated == numel1:
            # TODO @bozhiyou this only allows trailing one-shot ranges
            if r > TRITON_MAX_RBLOCK:  # one-shot no-iteration for small dimensions
                return False
            one_shot_ranges.append((accumulated, r))
            continue
        accumulated *= r
        if accumulated > numel1:
            return False
    fusion_log.debug(f"\033[33mfuse\033[0m {node1.get_name()} {node1.group[1]} with {node2.get_name()} {node2.group[1]}: one shot range")
    for accumulated, r in one_shot_ranges:
        mark_one_shot_range(node2, accumulated, r)
    return True


def ranges_fit(node1: BaseSchedulerNode, node2: BaseSchedulerNode):
    _, (numel1, rnumel1) = node1.group
    _, (numel2, rnumel2) = node2.group

    if numel1 == numel2:  # and rnumel1 == rnumel2:
        fusion_log.debug(f"\033[33mfuse\033[0m {node1.get_name()} {node1.group[1]} with {node2.get_name()} {node2.group[1]}: same parallelism")
        return True  # same parallelism: fit into same kernel

    if numel1 > numel2:
        if numel1 == numel2 * rnumel2:  # reduction over previous range
            return True  # TODO @bozhiyou ensure numel1 can be factored into numel2 * rnumel2
        return False  # TODO @bozhiyou split numel1

    assert numel1 < numel2
    if (numel1 * rnumel1 == numel2 and rnumel2 == 1):  # fusable if split second range
        return True
    # if isinstance(node2, FusedSchedulerNode):
    #     return any(_ranges_fit(node1, snode) for snode in node2.snodes)
    if isinstance(node2, SchedulerNode):
        return _ranges_fit_allow_one_shot(node1, node2)
    return False


@functools.lru_cache()
def _can_fuse_vertical(node1: BaseSchedulerNode, node2: BaseSchedulerNode):
    assert node1.get_operation_names() & node2.ancestors, "this is the outer condition to fuse vertically"
    if not ranges_fit(node1, node2):
        return False
    fusion_log.debug(f"\033[33mfuse\033[0m {node1.get_name()} {node1.group[1]} with {node2.get_name()} {node2.group[1]}: {node1.group[1]}, {node2.group[1]}")
    return True


# producer-consumer fusion = Scheduler.can_fuse_vertical() and <backend>.can_fuse_vertical()

@monkey.patch(scheduler.Scheduler)
def can_fuse_vertical(self: scheduler.Scheduler,
    node1: BaseSchedulerNode, node2: BaseSchedulerNode
):
    """
    - relax memory mismatch case
    """
    node1_buf_names = node1.get_buffer_names()
    node1_op_names = node1.get_operation_names()
    computed_deps: OrderedSet[Dep] = OrderedSet()
    from torch._inductor.scheduler import WhyNoFuse
    why = WhyNoFuse(node1, node2)

    for cd in node1.read_writes.writes:
        if not isinstance(cd, MemoryDep):
            continue
        for rd in node2.unmet_dependencies:
            if self.fusable_read_and_write(rd, cd):
                computed_deps.add(rd)

    for dep in node2.unmet_dependencies:
        if isinstance(dep, WeakDep) and self.fusable_weak_dep(dep, node1, node2):
            computed_deps.add(dep)

    remaining_deps = OrderedSet(
        dep.name for dep in node2.unmet_dependencies - computed_deps
    )
    if remaining_deps & node1_buf_names:
        # MemoryDeps didn't match and read different locations of the same buffer.
        # Examples here include:
        #   - MemoryDep("foo", x) != MemoryDep("foo", x + 1)
        #   - MemoryDep("foo", x) != StarDep("foo")
        why("memory deps did not match")
        # return False
    for name in remaining_deps:
        op_name = self.name_to_buf[name].defining_op.get_name()
        if node1_op_names & self.name_to_fused_node[op_name].ancestors:
            why("intermediate nodes between node1 & node2")
            return False
    return _can_fuse_vertical(node1, node2)


@monkey.patch(TritonScheduling)
def can_fuse_vertical(self: TritonScheduling,
    node1: BaseSchedulerNode, node2: BaseSchedulerNode
) -> bool:
    """Add reduction fusion rule."""
    can = monkey.fallback(self, node1, node2) or _can_fuse_vertical(node1, node2)
    if can and node1.is_reduction() and node2.is_reduction():
        for n2 in node2.get_nodes():
            assert isinstance(n2, SchedulerNode), n2
            if n2.is_reduction():
                ReductionDependency.capture(node1, n2)
    return can


@monkey.patch(CUDACombinedScheduling)
def fuse(self: CUDACombinedScheduling,
    node1: BaseSchedulerNode, node2: BaseSchedulerNode
) -> FusedSchedulerNode:
    """
    + rewrite dependent reduction IR graph
    """
    if node1.is_foreach() or node2.is_foreach():
        return monkey.fallback(node1, node2)
    if node1.is_reduction() or node2.is_reduction():
        for m in node1.get_nodes():
            for n in node2.get_nodes():
                if m.is_reduction() and n.is_reduction() and (m, n) in ReductionDependency.reduction_glue_functions:
                    ReductionDependency.reduction_glue_functions[m, n]()
                    fusion_log.debug(f"\033[33mReduction Rewrite\033[0m {node1.get_name()} {node1.group[1]} and {node2.get_name()} {node2.group[1]}")
    return FusedSchedulerNode.fuse(node1, node2)


# MemoryDep hashing

@monkey.patch(MemoryDep)
def __hash__(self: MemoryDep) -> int:
    """
    + as long as vars used in index are of the same size/range, the hash should be the same
    """
    cardinality = sympy_product(s for v, s in zip(self.var_names, self.size) if v in self.index.free_symbols)
    lower_bound = self.index.subs((v, 0) for v in self.var_names)
    upper_bound = self.index.subs((v, s-1) for v, s in zip(self.var_names, self.size))
    return hash((self.name, cardinality, lower_bound, upper_bound, self.mode))

@monkey.patch(MemoryDep)
def __eq__(self: MemoryDep, other: MemoryDep) -> bool:
    """
    + as long as vars used in index are of the same size/range, the hash should be the same
    """
    return hash(self) == hash(other)


##
# debug breakpoint
##

@monkey.patch(ir.ExpandView)
def __post_init__(self):
    super(ir.ExpandView, self).__post_init__()
    pass

@monkey.patch(ir.ComputedBuffer)
def __post_init__(self):
    super(ir.ComputedBuffer, self).__post_init__()
    pass

# @monkey.patch(ir.Reduction)
# def store_reduction(self: ir.Reduction, output_name, indexer, vars, reduction_vars):
#     """
#     reduction opsvalue flow
#     """
#     # load
#     value = self.inner_fn(vars, reduction_vars)
#     # reduction
#     value = ops.reduction(
#         self.dtype,
#         self.src_dtype,
#         self.reduction_type,
#         value
#     )
#     # store
#     value = ops.store_reduction(output_name, indexer(vars), value)
#     return value

torch._inductor.config.realize_opcount_threshold = 27
