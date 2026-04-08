"""
This file implements reduction fusion within a kernel, e.g. softmax.

Homomorphic Transformations: a system to analyze the data dependency between two reductions. If the operations between them are homomorphic (like exp between add and mul), you may rewrite the graph to eliminate the dependency. The `can_eliminate_reduction_dependency` function generates a new FX graph for the "update" function.

Putting It All Together: The `ReductionDependency.capture` method identifies potential dependent reduction fusion opportunities, and if the dependency can be eliminated, it prepares a "glue function" that rewrites the IR during the fusion process in the scheduler.

Softmax Fusion: a key target/usecase for this is softmax. By fusing the max and sum reductions, a single, highly efficient online softmax kernel is generated for performance.
"""

from .. import _monkey as monkey

from ._common import TRITON_MAX_RBLOCK

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

@monkey.patch(TritonKernelOverrides)
@staticmethod
def max(x, dim):
    # return f"triton_helpers.max2({x}, {dim})"
    return f"tl.max({x}, {dim})"

from . import _reduction

from torch._inductor.loop_body import InterpreterShim
# from torch.fx.node import Argument, Target
# class DependencyGraphInterpreter(InterpreterShim):
#     def placeholder(self, target : 'Target', args : Tuple['Argument', ...], kwargs : dict[str, Any]) -> Any:
#         return super().placeholder(target, args, kwargs)

@contextlib.contextmanager
def _set_subgraph_body(self, body_name: str):
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
        # def recursively_copy_from_dependency_graph(node: fx.Node):
        #     if node.op == 'placeholder' and node.target == 'ops':
        #         dependency_node_remapping[node] = 'suffix_ops'
        #     if node.target == 'stale_partial_reduction':
        #         dependency_node_remapping[node] = decomposed['localbuf']
        #     if node.target == 'prev_ancestor_partial_reduction':
        #         dependency_node_remapping[node] = decomposed_ancestors[
        #             node.args[0]]['localbuf']
        #     if node.op == 'call_method' and node.target == 'load':
        #         dependency_node_remapping[node] = decomposed_ancestors[
        #             name_of_load(node.meta['origin'])]['finalreduce']

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


# graph helpers
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
    """
    Analyzes if the dependency between an ancestor reduction (`anc_graph`) and a
    dependent reduction (`graph`) can be eliminated through homomorphic transformations.

    This is the core logic that enables fusing operations like softmax, where a `sum`
    reduction depends on the result of a `max` reduction. By proving that the
    intervening operations (e.g., `exp`) are homomorphic, we can rewrite the
    computation to update the `max` accumulator directly into the `sum`
    accumulator within a single loop.

    The process involves several steps:
    1.  **Build Dependency Subgraph**: It first identifies the specific data flow from the
        ancestor's output (a `store_reduction` op) to the dependent's input (a `load`
        op). It then constructs a new FX graph (`dependency_graph`) that captures
        only the part of the dependent kernel's computation that is on the path
        from this load to its final reduction.

    2.  **Homomorphic Hoisting**: It attempts to "hoist" the reduction operation in the
        dependency subgraph upwards by applying homomorphic rewrites. For example,
        if the pattern is exp-of-sum (`exp(load(max_val) + ...)`), it can be transformed
        into prod-of-exp `exp(load(max_val)) * exp(...)`. This rewrite moves the dependency
        on `max_val` outside the reduction.

    3.  **Formulate Update Function**: If the hoisting is successful, the function then
        inverts the intermediate operations to create a new FX graph. This "update
        function" describes how to modify the ancestor's accumulator to produce the
        dependent's result, effectively eliminating the need for a separate kernel.

    4.  **Simplification**: The resulting graph for the update function is simplified
        using Sympy to make it as efficient as possible.

    Args:
        graph (fx.Graph): The FX graph of the dependent reduction kernel's loop body.
        anc_graph (fx.Graph): The FX graph of the ancestor (producer) reduction kernel's loop body.
        buf_map (dict): A map to resolve buffer name differences between the two graphs.
        shared_reads (OrderedSet): A set of buffers that are read by both kernels.

    Returns:
        Union[fx.Graph, bool]: If the dependency can be eliminated, it returns a new `fx.Graph`
        representing the update function, which gets stored in TritonKernelOverrides.subgraphs and is later inlined by reductionx. Otherwise, it returns `False`.
    """
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


######
# Enable fusion: pattern extensions
######

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
    # also set multilane=False here because it's in fusion, so presumably not to use multilane would save registers
    reduction.kwargs = dict(**reduction.kwargs, multilane=False, modification=(dependency_graph_id, dependency_loads))



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
        fusion_log.debug(
            "  capture(%s, %s): dep_ops=%s anc_reds=%s anc_names=%s",
            node1.get_name(), node2.get_name(), dep_ops, anc_reds, anc_names,
        )
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



@monkey.patch(SchedulerNode)
def get_ranges(self: SchedulerNode, one_shot_ranges: dict[sympy.Integer, sympy.Integer] | None = None):
    """
    Retrieves the iteration/reduction ranges for a :class:`SchedulerNode`.

    This patched method is fully backward-compatible. By default (when `one_shot_ranges` 
    is `None`), it preserves the original behavior (getter of `self._sizes`) and simply returns the unmodified 
    iteration ranges `(ranges, reduction_ranges)` for the node.

    Additional Functionality:
    When `one_shot_ranges` is provided, the method extends the default behavior by filtering 
    out specific "one-shot" ranges from the returned standard iteration ranges. These 
    one-shot ranges represent dimensions that are processed "all at once" instead of 
    being explicitly looped over. The identified one-shot ranges are then stored separately 
    in the node's `one_shot` attribute for use during kernel generation.

    Args:
        one_shot_ranges (dict[sympy.Integer, sympy.Integer] | None): A dictionary specifying 
            the ranges to be treated as one-shot.
            - Key (prefix): The accumulated size (product) of all dimensions preceding 
              the one-shot dimension.
            - Value (range_size): The size of the one-shot dimension itself.
            Defaults to None.

    Returns:
        tuple: A tuple `(ranges, reduction_ranges)` where `ranges` contains the standard 
        iteration ranges (with any active one-shot ranges filtered out), and 
        `reduction_ranges` remains unmodified.
    """
    ranges, rranges = monkey.fallback(self)
    if one_shot_ranges is None:
        return ranges, rranges
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


class _RangeMethods:
    @staticmethod
    def _mark_one_shot_range(self: SchedulerNode, prefix: sympy.Integer, range_size: sympy.Integer):
        """Annotates a dimension of a SchedulerNode as a "one-shot" range to enable fusion.

        A "one-shot" range is a dimension that exists in one operation's iteration
        space but not in another's, which we want to fuse. In the fused kernel,
        this dimension is not iterated over with an explicit outer loop. Instead,
        the entire range is processed "all at once" within a single iteration of
        the other loops, typically through vectorization or block-level operations
        (e.g., `tl.arange` in Triton).

        This is useful for fusing a smaller operation into a larger one. For example,
        fusing an operation on a `(128,)` tensor with an operation on a `(128, 64)`
        tensor. The dimension of size `64` becomes the one-shot range.

        Example of conceptual Triton kernel for the above fusion:
        ```python
        # pid iterates over the 128 dimension
        m_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)

        # The one-shot range of size 64 is handled in a single block.
        n_offsets = tl.arange(0, 64)

        # Load a 1D block from the first tensor
        a = tl.load(ptr_a + m_offsets)

        # Load a 2D block from the second tensor, materializing the one-shot range
        b = tl.load(ptr_b + m_offsets[:, None] * stride_b_m + n_offsets[None, :] * stride_b_n)

        # Fused computation where 'a' is broadcasted
        result = pointwise_op(a[:, None], b)
        ```
        
        This implementation of annotation modifies the node by setting an `one_shot`
        attribute, a dictionary mapping the accumulated size of preceding dimensions
        (`prefix`) to the size of the one-shot dimension (`range_size`).
        """
        one_shot_ranges: dict[sympy.Integer, sympy.Integer] = getattr(self, 'one_shot', {})
        one_shot_ranges[prefix] = range_size
        setattr(self, 'one_shot', one_shot_ranges)


    @staticmethod
    def _ranges_fit_allow_one_shot(node1: BaseSchedulerNode, node2: SchedulerNode) -> bool:
        """Check if two nodes can be fused by treating some dimensions of the larger node
        as "one-shot" tiles. This is a specialized fusion check that allows fusing a smaller
        operation (`node1`) into a larger one (`node2`) if the iteration space of `node1`
        is a prefix of `node2`'s iteration space an the remaining dimensions of `node2` can be considered "one-shot", meaning they are
        not iterated over by loops in the fused kernel but are instead handled by broadcasting or
        specialized codegen.

        For example, fusing a reduction of shape `(128,)` into a pointwise op of shape
        `(128, 64)` is possible. The `64` dimension of the pointwise op becomes a
        one-shot range.
        """
        _, (numel1, rnumel1) = node1.group
        _, (numel2, rnumel2) = node2.group

        ranges2, _ = node2.get_ranges()
        accumulated = sympy.Integer(1)
        one_shot_ranges = []
        for i, r in enumerate(ranges2):
            if accumulated == numel1:
                # TODO(bozhiyou): This currently only supports one-shot ranges that are trailing
                # dimensions. It could be extended to handle one-shot dimensions in the middle.
                if r > TRITON_MAX_RBLOCK:  # one-shot no-iteration for small dimensions
                    return False
                one_shot_ranges.append((accumulated, r))
                continue
            accumulated *= r
            if accumulated > numel1:
                return False
        fusion_log.debug(f"\033[33mfuse\033[0m {node1.get_name()} {node1.group[1]} with {node2.get_name()} {node2.group[1]}: one shot range")
        for accumulated, r in one_shot_ranges:
            _RangeMethods._mark_one_shot_range(node2, accumulated, r)
        # NOTE @bozhiyou actual handling (`_sizes`, `var_ranges`, `iter_vars`, etc) of the one-shot dimension is
        # deferred to the codegen phase, which has a more complete view of the kernel.

        return True


    @staticmethod
    def ranges_fit(node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        """Allow fusing nodes even if their iteration spaces don't
        perfectly match.

        This function extends the default fusion logic by allowing fusion even when
        iteration spaces (`numel`) and reduction spaces (`rnumel`) don't perfectly
        match. It enables more advanced fusion patterns by checking for specific
        compatible relationships between the nodes' iteration spaces.

        The main scenarios handled are:
        1.  **Same Parallelism**: Both nodes have the same iteration space (`numel1 == numel2`).
            This is the simplest case for fusion.
        2.  **Reduction into Pointwise**: A reduction (`node1`) is fused into a pointwise
            operation (`node2`). This is possible if the total number of elements in the
            reduction's output (`numel1 * rnumel1`) matches the number of elements in the
            pointwise operation (`numel2`).
        3.  **Pointwise into Reduction**: A pointwise operation (`node1`) is fused into a
            reduction (`node2`). This is possible if the pointwise op's iteration space (`numel1`)
            can be mapped to the reduction's iteration and reduction spaces (`numel2 * rnumel2`).
        4.  **One-Shot Tiling**: A special case where a larger pointwise operation (`node2`)
            is fused with a smaller one (`node1`) by treating the extra dimensions
            of `node2` as "one-shot" tiles that are not iterated over in the fused kernel.
        """
        _, (numel1, rnumel1) = node1.group
        _, (numel2, rnumel2) = node2.group

        if numel1 == numel2:  # and rnumel1 == rnumel2: # Original check was stricter
            # Nodes have the same iteration space, so they can be fused.
            # This allows fusing a pointwise op with a reduction if their `numel` matches,
            # ignoring the reduction dimension (`rnumel`).
            fusion_log.debug(f"\033[33mfuse\033[0m {node1.get_name()} {node1.group[1]} with {node2.get_name()} {node2.group[1]}: same parallelism")
            return True  # same parallelism: fit into same kernel

        if numel1 > numel2:
            # This case handles fusing a pointwise operation (node1) into a reduction (node2).
            if numel1 == numel2 * rnumel2:  # Pointwise op's elements match total elements of the reduction.
                # TODO @bozhiyou For correctness, ensure that the iteration space of `node1`
                # can be cleanly factored into the iteration and reduction spaces of `node2`.
                # This is currently assumed but not explicitly guarded.
                return True
            # TODO @bozhiyou Implement loop splitting for `node1`. If `numel1` is a multiple of
            # `numel2` but doesn't fit the pattern above, we could potentially split
            # the loop of `node1` to enable fusion.
            return False

        assert numel1 < numel2
        if (numel1 * rnumel1 == numel2 and rnumel2 == 1):  # Fuse a reduction into a pointwise op by splitting the pointwise range (parallelism reduced).
            # This pattern occurs when `node1` is a reduction and `node2` is a pointwise
            # operation whose number of elements (`numel2`) equals the total number of
            # elements processed by the reduction (`numel1 * rnumel1`).
            return True
        # NOTE @bozhiyou unnecessary to recursively check fusion with a FusedSchedulerNode.
        # The fusion logic in `Scheduler.fuse_nodes_once`
        # iterates and handles fused nodes by looking up the `name_to_fused_node` map.

        # The final case handles "one-shot" tiling, where a larger pointwise op (`node2`)
        # is fused with a smaller one (`node1`). The extra dimensions in `node2` are
        # treated as tiles that are not iterated over with an explicit loop in the fused kernel.
        if isinstance(node2, SchedulerNode):
            return _RangeMethods._ranges_fit_allow_one_shot(node1, node2)
        return False


@functools.lru_cache()
def _have_compatible_ranges(node1: BaseSchedulerNode, node2: BaseSchedulerNode):
    assert node1.get_operation_names() & node2.ancestors, "this is the outer condition to fuse vertically"
    if not _RangeMethods.ranges_fit(node1, node2):
        return False
    fusion_log.debug(f"\033[33mfuse\033[0m {node1.get_name()} {node1.group[1]} with {node2.get_name()} {node2.group[1]}: {node1.group[1]}, {node2.group[1]}")
    return True


# Inductor's fusion logic for vertical fusion (producer-consumer)
# involves a two-part check:
# 1. `Scheduler.can_fuse_vertical()`: This performs backend-agnostic checks on the
#    dependency graph, ensuring that all dependencies of the consumer node can be
#    satisfied by the producer or are already available.
# 2. `<backend>.can_fuse_vertical()`: This performs backend-specific checks, such
#    as tiling compatibility or hardware constraints for the target backend (e.g., Triton).

@monkey.patch(scheduler.Scheduler)
def can_fuse_vertical(self: scheduler.Scheduler,
    node1: BaseSchedulerNode, node2: BaseSchedulerNode
):
    """
    - Relax the case where `MemoryDeps` didn't match and read different locations of the same buffer.
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
            # aggressive fusion
            elif isinstance(rd, MemoryDep):
                def _drop_unused_symbols(md: MemoryDep):
                    md_ranges = [(v, s) for v, s in zip(md.var_names, md.size) if v in md.index.free_symbols]
                    if len(md_ranges) != len(md.var_names):
                        return MemoryDep(md.name, md.index, tuple(x[0] for x in md_ranges), tuple(x[1] for x in md_ranges), md.mode)
                    return md
                _rd = _drop_unused_symbols(rd)
                _cd = _drop_unused_symbols(cd)
                if _rd is not rd or _cd is not cd and self.fusable_read_and_write(
                    _rd, _cd
                ):
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
    return True


@monkey.patch(TritonScheduling)
def can_fuse_vertical(self: TritonScheduling,
    node1: BaseSchedulerNode, node2: BaseSchedulerNode
) -> bool:
    """Identifying and enabling our new fusion for dependent reductions."""
    can = monkey.fallback(self, node1, node2) or _have_compatible_ranges(node1, node2)
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


# Raise Inductor materialization thresholds to prevent early realization of
# intermediates in attention patterns (e.g. exp(score - max) in causal+ALiBi
# has num_reads=5 due to mask + local_pos buffer reads).
torch._inductor.config.realize_opcount_threshold = 27
torch._inductor.config.realize_reads_threshold = 8
