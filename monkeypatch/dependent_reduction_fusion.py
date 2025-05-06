from . import _monkey as monkey

import collections
import itertools
import contextlib
import copy
from typing import Any, Union, Tuple

import torch
from torch import fx
from torch._inductor import ir
from torch._inductor.virtualized import V, OpsValue
from torch.utils._ordered_set import OrderedSet



from torch._inductor.ops_handler import ReductionType
from torch._inductor.codegen.common import CSEVariable
from torch._inductor.utils import is_welford_reduction, IndentedBuffer
from torch._inductor.codegen.triton import TritonKernelOverrides, TritonKernel, TritonCSEVariable
from torch._inductor.codegen.triton import triton_acc_type, triton_compute_type
from torch._inductor.codegen.simd import constant_repr


@monkey.patch(TritonKernelOverrides)
@staticmethod
def reduction_localbuf(
    _self: TritonKernelOverrides,
    src_dtype: torch.dtype,
    reduction_type: ReductionType,
):
    assert isinstance(V.kernel, TritonKernel), f"unknown kernel type: {type(V.kernel)}"
    self: TritonKernel = V.kernel
    def _reduction_localbuf(
        self,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
    ):
        assert self.inside_reduction
        assert not self.persistent_reduction, "persistent reduction does not use accumulator"
        masks = OrderedSet(f"{tree.prefix}mask" for tree in self.range_trees)
        self.filter_masks(masks)
        masks = sorted(masks)
        if self._load_mask:
            masks.append(self._load_mask)
        default = ir.Reduction.default_accumulator(reduction_type, src_dtype)
        if reduction_type == "welford_reduce":
            default = default[0]  # which is 0 since default of welford_reduce is (0, 0, 0)
        default = self._map_tuple_or_scalar(constant_repr, default)

        result_var = self.cse.newvar()
        result_var.mask_vars = OrderedSet(var for var in masks if var[0] != "r")
        accumulator = result_var
        acc_type = triton_acc_type(src_dtype)
        if not isinstance(default, tuple):
            self.body.writeline(
                f"{accumulator} = tl.full({self.dense_size_str()}, {default}, {acc_type})"
            )
        return accumulator
    return _reduction_localbuf(self, src_dtype, reduction_type)


@monkey.patch(TritonKernelOverrides)
@staticmethod
def reduction_combine(
    _self: TritonKernelOverrides,
    src_dtype: torch.dtype,
    reduction_type: ReductionType,
    value: Union[CSEVariable, Tuple[CSEVariable, ...]],
    accumulator: CSEVariable,
    previous: CSEVariable,
):
    assert isinstance(V.kernel, TritonKernel), f"unknown kernel type: {type(V.kernel)}"
    self: TritonKernel = V.kernel
    def _reduction_combine(
        self,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[CSEVariable, Tuple[CSEVariable, ...]],
        accumulator: CSEVariable,
        previous: CSEVariable,
    ):
        assert self.inside_reduction
        masks = OrderedSet(f"{tree.prefix}mask" for tree in self.range_trees)
        self.filter_masks(masks)
        masks = sorted(masks)
        if self._load_mask:
            masks.append(self._load_mask)

        cond = " & ".join(masks)
        def where_cond(tval, fval) -> str:
            if not cond:
                return tval
            return TritonKernelOverrides.where(cond, tval, fval)

        if reduction_type in {"argmax", "argmin"} or is_welford_reduction(reduction_type):
            raise NotImplementedError
        else:
            combine_fn = ir.get_reduction_combine_fn(reduction_type, src_dtype)
            updated = combine_fn(previous, value)

            self.stores.writeline(
                f"{accumulator} = {where_cond(updated, accumulator)}")
        return updated
    return _reduction_combine(self, src_dtype, reduction_type, value, accumulator, previous)


@monkey.patch(TritonKernelOverrides)
@staticmethod
def reduction_update(
    _self: TritonKernelOverrides,
    accumulator: CSEVariable,
    updated: CSEVariable,
):
    assert isinstance(V.kernel, TritonKernel), f"unknown kernel type: {type(V.kernel)}"
    self: TritonKernel = V.kernel
    def _reduction_update(
        self,
        accumulator: CSEVariable,
        updated: CSEVariable,
    ):
        assert self.inside_reduction
        masks = OrderedSet(f"{tree.prefix}mask" for tree in self.range_trees)
        self.filter_masks(masks)
        masks = sorted(masks)
        if self._load_mask:
            masks.append(self._load_mask)

        cond = " & ".join(masks)
        def where_cond(tval, fval) -> str:
            if not cond:
                return tval
            return TritonKernelOverrides.where(cond, tval, fval)
    

        
        self.stores.writeline(f"{accumulator} = {where_cond(updated, accumulator)}")
        return accumulator
    return _reduction_update(self, accumulator, updated)


@monkey.patch(TritonKernelOverrides)
@staticmethod
def reduction_finalreduce(
    _self: TritonKernelOverrides,
    dtype: torch.dtype,
    src_dtype: torch.dtype,
    reduction_type: ReductionType,
    accumulator: CSEVariable,
):
    assert isinstance(V.kernel, TritonKernel), f"unknown kernel type: {type(V.kernel)}"
    self: TritonKernel = V.kernel
    def _reduction_finalreduce(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        accumulator: CSEVariable,
    ):
        assert self.inside_reduction

        dim: int = self.triton_tensor_ndim() - 1

        def final_reduction(value):
            use_helper = reduction_type in {"any", "max", "min", "prod"}
            module = "triton_helpers" if use_helper else "tl"
            if reduction_type in {"max", "min"}:
                return self.reduction_resize(
                    f"{module}.{reduction_type}2({value}, {dim})"
                )
            return self.reduction_resize(f"{module}.{reduction_type}({value}, {dim})")
        
        result_var: Any = self.cse.newvar()
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
        return result_var
    return _reduction_finalreduce(self, dtype, src_dtype, reduction_type, accumulator)




from torch._inductor.loop_body import InterpreterShim
# from torch.fx.node import Argument, Target
class DependencyGraphInterpreter(InterpreterShim):
    pass
    # def placeholder(self, target : 'Target', args : Tuple['Argument', ...], kwargs : dict[str, Any]) -> Any:
    #     return super().placeholder(target, args, kwargs)

@contextlib.contextmanager
def set_subgraph_body(self, body_name: str):
    old_body, old_indexing_code, old_loads, old_compute, old_stores, old_suffix = self.body, self.indexing_code, self.loads, self.compute, self.stores, self.suffix
    yield
    self.body, self.indexing_code, self.loads, self.compute, self.stores, self.suffix = old_body, old_indexing_code, old_loads, old_compute, old_stores, old_suffix


@monkey.patch(TritonKernelOverrides)
def modification(
        _self: TritonKernelOverrides, subgraph: fx.Graph, *subgraph_args, output_name: str = '', **fixed_inputs
    ) -> str:
    """
    This function is adapted from TritonTemplateKernel::modification in torch/_inductor/select_algorithm.py.

    This creates a modification function for a subgraph.
    """
    self: TritonKernel = V.kernel


    # num = 0
    # while f"mod_{subgraph_number}_{num}" in self.subgraph_bodies:
    #     num += 1
    # with self.create_subgraph_body(f"mod_{subgraph_number}_{num}"):
    if True:
        assert isinstance(subgraph, fx.Graph)
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
                return fixed_inputs[name]


        if 'ops' in fixed_inputs:
            # CSEProxy and CSE are hardcoded for compute scope
            _V_kernel_compute, _V_kernel_compute_cache = V.kernel.compute, V.kernel.cse.cache
            V.kernel.compute, V.kernel.cse.cache = V.kernel.suffix, {}

        with V.set_ops_handler(OpsHandlerOverride(V.ops)):
            out = DependencyGraphInterpreter(subgraph, submodules={}).run(V.ops, *subgraph_args)

        if 'ops' in fixed_inputs:
            V.kernel.compute, V.kernel.cse.cache = _V_kernel_compute, _V_kernel_compute_cache

        return out

        self.codegen_body()
        self.body.writeline(f"{output_name} = {out.value}")

        body_val = self.body.getvalue()
        self.cse.invalidate(set())  # type: ignore[arg-type]
        return body_val


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

@monkey.patch(ir.Reduction)
def store_reduction(self: ir.Reduction, output_name, indexer, vars, reduction_vars):
    """
    reduction opsvalue flow
    """
    # load
    value = self.inner_fn(vars, reduction_vars)
    # reduction
    value = ir.ops.reduction(
        self.dtype,
        self.src_dtype,
        self.reduction_type,
        value
    )
    # store
    value = ir.ops.store_reduction(output_name, indexer(vars), value)
    return value


from torch._inductor.scheduler import SchedulerNode


# helper
def find_unique_node(graph: fx.Graph, *, op, target) -> fx.Node:
    candidates = graph.find_nodes(op=op, target=target)
    assert len(candidates) == 1, f"{op, target} with multiple matches {candidates}"
    return next(iter(candidates))

# helpers
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


"""
mul (R, add) (R, add)   # distributive property
exp (R, add) (R+, mul)
log (R+, mul) (R, add)
"""
HOMOMORPHISM = {
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

# reduction(map_f(a_{n})) = f(g(a_{n}))
# {reduction: {f: g}}
REDUCTION_HOMOMORPHISM: dict[ReductionType, dict] = {
    'sum': {
        'log': 'product',  # sum(log(x)...) = log(prod(x...))
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


def homomorphic_transform(domain_node: fx.Node, hom_node: fx.Node, codomain_funcname: str):
    """
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
                lambda x: x if x in domain_node.args else domain_arg)
            codomain_node_args.append(new_hom_node)
    assert not domain_node.kwargs, NotImplementedError("TODO check kwargs as well")
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

def try_homomorphic_transform(hom_node: fx.Node, codomain_hints: dict[str, str]):
    """
    If `hom_node` is a homomorphism from some previous operation (`domain_operation`)
    to one of the `codomain_hints` operations (`codomain_operation`), then
        hom ∘ domain_operation = codomain_operation ∘ hom
    so we can transform `hom` to `codomain_operation` as the last/outer-most op.
    """
    domain_calls = [n for n in hom_node.all_input_nodes if n.op == 'call_method']
    if len(domain_calls) != 1:
        return False
    domain_node = domain_calls[0]
    if len(domain_node.users) != 1:
        return False

    for codomain_funcname in codomain_hints:
        if not isinstance(domain_node.target, str):
            raise NotImplementedError(f"TODO transform target into string")
        if (domain_node.target, codomain_funcname) in HOMOMORPHISM and (
            hom_node.target in HOMOMORPHISM[(domain_node.target, codomain_funcname)]):
                homomorphic_transform(domain_node=domain_node, hom_node=hom_node, codomain_funcname=codomain_funcname)
                return True
        # TODO further try to transform `domain_node` recursively
    return False

def hoist_reduction(hom_node: fx.Node, codomain_node: fx.Node, domain_reduction_type: str):
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
        if codomain_arg == type_of_reduction(codomain_node):
            domain_node_args.append(domain_reduction_type)
            continue
        if codomain_arg is hom_node:
            domain_node_args.append(hom_node.args[1]) # skip `ops` handler
            continue
        domain_node_args.append(codomain_arg)
    assert not codomain_node.kwargs, NotImplementedError("TODO check kwargs as well")
    domain_node_kwargs = codomain_node.kwargs
    # Create a new domain node
    with hom_node.graph.inserting_before(hom_node):
        domain_node = codomain_node.graph.create_node(
            op=codomain_node.op,  # 'call_method'
            target=codomain_node.target,  # 'reduction'
            args=tuple(domain_node_args),
            kwargs=domain_node_kwargs,
            type_expr=hom_node.type
        )
        domain_node.meta = copy.copy(codomain_node.meta)

    # Create a new hom node applied to the domain node's output
    with domain_node.graph.inserting_after(domain_node):
        new_hom_node = codomain_node.graph.node_copy(
            hom_node,
            arg_transform=lambda x: domain_node if x in domain_node.args[1:] else x
        )

    # Replace all uses of codomain_node with new_hom_node
    codomain_node.replace_all_uses_with(new_hom_node)
    codomain_node.graph.erase_node(codomain_node)
    assert not hom_node.users, f"homomorphism has dependants {tuple(hom_node.users.keys())}"
    hom_node.graph.erase_node(hom_node)

def try_hoist_reduction(reduction: fx.Node):
    """
    f:A -> B is a homomorphism b/w (A,∗) and (B,∘) such that f(a_0 ∗ a_1) = f(a_0) ∘ f(a_1)
    Generally, if `g` and `h` has same associative property, map_f is the map version of f,
    f:X -> Y is a homomorphism b/w (X, g) and (Y, h) such that f(g(a_{n})) = h(map_f(a_{n}))

    for a reduction h, here is a dict of (f: g).

    The goal of the transformation is to hoist reductions.
    """
    # only handles a chains of ops
    calls = [n for n in reduction.all_input_nodes if n.op == 'call_method']
    if len(calls) != 1:
        return False
    pred_call = calls[0]
    if len(pred_call.users) != 1:
        return False

    homoms = REDUCTION_HOMOMORPHISM[type_of_reduction(reduction)]
    if pred_call.target not in homoms:
        if not try_homomorphic_transform(pred_call, homoms):
            return False
        return try_hoist_reduction(reduction)  # try harder
    domain_funcname: str = homoms[pred_call.target]
    hoist_reduction(hom_node=pred_call, codomain_node=reduction, domain_reduction_type=domain_funcname)
    while try_hoist_reduction(reduction): pass  # greedy iterations
    return True

def suppress_linting(node: fx.Node):
    # suppress linting node by adding it to output args.
    output: fx.Node = find_unique_node(node.graph, op='output', target='output')
    output.args += (node,)

def decompose_reduction(reduction: fx.Node, user_reduction: fx.Node | None = None):
    assert reduction.op == 'call_method' and reduction.target == 'reduction'
    ops_handler, dtype, src_dtype, reduction_type, value= reduction.args
    with reduction.graph.inserting_before(reduction):
        localbuf = reduction.graph.create_node(
            op='call_method', target='reduction_localbuf',
            args=(ops_handler, src_dtype, reduction_type))
        combine = reduction.graph.create_node(
            op='call_method', target='reduction_combine',
            args=(ops_handler, src_dtype, reduction_type, value) + (localbuf, localbuf))
        # with reduction.graph.inserting_before(user_reduction.next
        #     ) if user_reduction is not None else contextlib.nullcontext():
        #     update = reduction.graph.create_node(
        #         op='call_method', target='reduction_update',
        #         args=(ops_handler,) + (localbuf, combine),
        #     )
        suppress_linting(combine)
        finalreduce = reduction.graph.create_node(
            op='call_method', target='reduction_finalreduce',
            args=(ops_handler, dtype, src_dtype, reduction_type) + (localbuf,),
            type_expr=reduction.type
        )
    reduction.replace_all_uses_with(finalreduce)
    reduction.graph.erase_node(reduction)
    return {
        'localbuf': localbuf,
        'combine': combine,
        # 'update': update,
        'finalreduce': finalreduce,
    }
        

def find_loads(graph: fx.Graph, names: set[str] = set()):
    load_nodes = graph.find_nodes(op='call_method', target='load')
    if not names:
        return set(load_nodes)
    load_points: set[fx.Node] = set(ld
        for ld in load_nodes
            for out in names
                if name_of_load(ld) == out
    )
    assert len(load_points), f"no load found for {names}"
    return load_points

class NodeRemapping(collections.UserDict[fx.Node, fx.Node]):
    def __setitem__(self, src: fx.Node, dst: fx.Node):
        dst.meta['origin'] = src
        super().__setitem__(src, dst)

def eliminate_reduction_dependency(graph: fx.Graph, anc_graph: fx.Graph, anc_out_to_keep:set[str] = set()):
    """ancestor reduction --> dependent reduction"""
    anc_output: fx.Node = find_unique_node(anc_graph, op='output', target='output')
    anc_reduction_stores = {name_of_store_reduction(store_reduction): store_reduction for store_reduction in anc_output.all_input_nodes}

    reduction: fx.Node = find_unique_node(graph, op='call_method', target='reduction')
    # 0. trace dependency path
    all_loads = collections.defaultdict(OrderedSet)
    for ld in graph.find_nodes(op='call_method', target='load'):
        all_loads[name_of_load(ld)].add(ld)

    dependency_loads = {}
    for name in anc_reduction_stores:
        if name not in all_loads:
            return False    # ancestor has external load; cannot fuse
        dependency_loads[name] = all_loads[name]

    # build dependency graph
    dependency_dag_nodes = {name: trace_dag_with_sink(reduction, lds) for name, lds in dependency_loads.items()}
    dependency_graph: fx.Graph = fx.Graph()
    dag_node_remapping = NodeRemapping()
    out_dag_input_remapping: dict[fx.Node, fx.Node] = {}
    for name, dag in dependency_dag_nodes.items():
        for ld in dependency_loads[name]:
            if ld.args[0] not in dag_node_remapping:  # ops handler
                dag_node_remapping[ld.args[0]] = dependency_graph.node_copy(ld.args[0])
            dag_node_remapping[ld] = dependency_graph.create_node(
                op=ld.op, target=ld.target, args=(  # updated_ancestor_partial_reduction
                    dag_node_remapping[ld.args[0]],
                    'updated_' + name_of_load(ld),
            ), type_expr=ld.type)
        out_dag_args = set()
        def copy_from_dag(arg: fx.Node):
            if arg in dag_node_remapping:
                return dag_node_remapping[arg]
            out_dag_args.add(arg)
            if arg not in out_dag_input_remapping:
                out_dag_input_remapping[arg] = dependency_graph.create_node(
                    op='placeholder', target='stale_partial_reduction', type_expr=arg.type)
            return out_dag_input_remapping[arg]
        for node in dag:
            if node in dag_node_remapping:
                continue
            dag_node_remapping[node] = dependency_graph.node_copy(
                node, arg_transform=copy_from_dag)
        if len(out_dag_args) > 1:
            return False  # cannot factor out more than one non-dependency input
        pre_merge_node = next(iter(out_dag_args))
        assert len(pre_merge_node.users) == 1
        original_merge_node = next(iter(pre_merge_node.users))
        # save a handle to the last node before transformations
        output_handle = dependency_graph.create_node(
                op='output', target='output', args=(dag_node_remapping[reduction],))
        try_hoist_reduction(dag_node_remapping[reduction])

        pre_merge_node = out_dag_input_remapping[pre_merge_node]
        while True:
            assert len(pre_merge_node.users) == 1
            merge_node = next(iter(pre_merge_node.users))
            if merge_node.meta['origin'] is original_merge_node:
                break
            merge_node.replace_all_uses_with(pre_merge_node)
            merge_node.graph.erase_node(merge_node)
        arg_index = merge_node.args.index(pre_merge_node)
        assert arg_index in (1, 2)
        dep_index = arg_index % 2 + 1

        with dependency_graph.inserting_before(merge_node):
            restored = pre_merge_node
            to_inverse = list(output_handle.all_input_nodes)[-1]
            while True:
                inverse_target = INVERSE[dependency_graph._target_to_str(to_inverse.target)]
                if to_inverse is merge_node:
                    inverse_merge_args = tuple(restored if i == arg_index else arg for i, arg in enumerate(to_inverse.args))
                    restored = dependency_graph.create_node(
                        op=to_inverse.op, target=inverse_target,
                        args=inverse_merge_args, kwargs=to_inverse.kwargs
                    )
                    def recursively_copy_pre_merge_ops(arg_node: fx.Node):
                        assert arg_node.graph is dependency_graph
                        if arg_node.op == 'placeholder':
                            return arg_node
                        if arg_node.meta['origin'] in dependency_loads[name]:
                            return dependency_graph.create_node(
                                op='call_method', target='load', args=(  # prev_ancestor_partial_reduction
                                    arg_node.args[0],
                                    'stale_' + name,
                                ), type_expr=arg_node.type)
                        return dependency_graph.node_copy(arg_node, arg_transform=recursively_copy_pre_merge_ops)
                    with dependency_graph.inserting_before(restored):
                        new_dep = dependency_graph.node_copy(
                            to_inverse.args[dep_index],
                            arg_transform=recursively_copy_pre_merge_ops)
                        restored.replace_input_with(restored.args[dep_index], new_dep)
                    merge_node.replace_input_with(pre_merge_node, restored)
                    break
                old_restored = restored
                restored = dependency_graph.create_node(
                    op=to_inverse.op, target=inverse_target,
                    args=(to_inverse.args[0], restored), kwargs=to_inverse.kwargs
                )
                merge_node.replace_input_with(old_restored, restored)



    # 1. pull and decompose ancestor reductions, eliminate dependency
    node_remapping: dict[fx.Node, fx.Node] = {}  # dependent reduction graph node to merged graph node
    index_to_getter = {name_of_index(gi): gi for gi in graph.find_nodes(
        op='call_module', target='get_index')}
    arg_to_loader = {flatten_args(ld): ld for ld in graph.find_nodes(
        op='call_method', target='load')}
    def recursively_copy_from_ancestor_graph(arg_node: fx.Node):
        if arg_node in node_remapping:
            return node_remapping[arg_node]
        if arg_node.op == 'placeholder':
            return find_unique_node(graph, op=arg_node.op, target=arg_node.target)
        if arg_node.op == 'call_module' and arg_node.target =='get_index':
            key = name_of_index(arg_node)
            if key not in index_to_getter:
                index_to_getter[key] = graph.node_copy(
                    arg_node, arg_transform=recursively_copy_from_ancestor_graph)
            return index_to_getter[key]
        if arg_node.op == 'call_method' and arg_node.target =='load':
            key = flatten_args(arg_node)
            if key not in arg_to_loader:
                arg_to_loader[key] = graph.node_copy(
                    arg_node, arg_transform=recursively_copy_from_ancestor_graph)
            return arg_to_loader[key]
        node_ = graph.node_copy(arg_node, arg_transform=recursively_copy_from_ancestor_graph)
        node_remapping[arg_node] = node_
        return node_

    def recursively_erase_node(node: fx.Node):
        inputs = [x for x in node.all_input_nodes]
        node.graph.erase_node(node)
        for input_node in inputs:
            if input_node.op not in {"placeholder", "output"} and len(input_node.users) == 0:
                recursively_erase_node(input_node)

    decomposed_ancestors = {}
    output: fx.Node = find_unique_node(graph, op='output', target='output')
    for name in dependency_loads:
        anc_store_reduction = anc_reduction_stores[name]
        anc_reduction = reduction_to_store(anc_store_reduction)
        with graph.inserting_before(next(iter(dependency_loads[name]))):
            anc_reduction_ = graph.node_copy(
                anc_reduction, arg_transform=recursively_copy_from_ancestor_graph
            )
            decomposed = decompose_reduction(anc_reduction_, user_reduction=reduction)
            decomposed_ancestors[name] = decomposed
            for ld in dependency_loads[name]:
                ld.replace_all_uses_with(decomposed['combine'])
                recursively_erase_node(ld)
            node_remapping[anc_reduction] = decomposed['finalreduce']

        if name not in anc_out_to_keep:
            continue

        with graph.inserting_before(output):
            node_remapping[anc_store_reduction] = graph.node_copy(
                anc_store_reduction,
                arg_transform=recursively_copy_from_ancestor_graph
            )
            output.args = (
                output.args[0] + node_remapping[anc_store_reduction]
                if isinstance(output.args[0], tuple) else
                (output.args[0], node_remapping[anc_store_reduction])
            ) + output.args[1:]

    # 2. add replay function
    decomposed = decompose_reduction(reduction)
    dependency_node_remapping: dict[fx.Node, fx.Node] = {}
    def recursively_copy_from_dependency_graph(node: fx.Node):
        if node.op == 'placeholder' and node.target == 'ops':
            dependency_node_remapping[node] = find_unique_node(graph,
                op=node.op, target=node.target)
        if node.target == 'stale_partial_reduction':
            dependency_node_remapping[node] = decomposed['localbuf']
        if node.target == 'prev_ancestor_partial_reduction':
            dependency_node_remapping[node] = decomposed_ancestors[
                node.args[0]]['localbuf']
        if node.op == 'call_method' and node.target == 'load':
            dependency_node_remapping[node] = decomposed_ancestors[
                name_of_load(node.meta['origin'])]['combine']
        if node not in dependency_node_remapping:
            dependency_node_remapping[node] = graph.node_copy(
                node,
                arg_transform=recursively_copy_from_dependency_graph
            )
        return dependency_node_remapping[node]
    with graph.inserting_before(decomposed['combine']):
        # updated_partial = graph.node_copy(
        #     output_handle.args[0], arg_transform=recursively_copy_from_dependency_graph
        # )
        updated_partial = graph.create_node(
            op='call_method', target='modification', args=(
                decomposed['combine'].args[0],  # ops handler
                dependency_graph,
                decomposed['localbuf']
            ), kwargs={
                'localbuf': decomposed['localbuf'],
                **{f'stale_{name}': decomposed_ancestors[name]['localbuf'] for name in decomposed_ancestors},
                **{f'updated_{name}': decomposed_ancestors[name]['combine'] for name in decomposed_ancestors},
            }
        )
    decomposed['combine'].args = decomposed['combine'].args[:-1] + (updated_partial,)  # replace last arg of combine

    dependency_node_remapping: dict[fx.Node, fx.Node] = {}
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
        if node not in dependency_node_remapping:
            dependency_node_remapping[node] = graph.node_copy(
                node,
                arg_transform=recursively_copy_from_dependency_graph
            )
        return dependency_node_remapping[node]
    with graph.inserting_before(decomposed['finalreduce']):
        updated_partial = graph.create_node(
            op='call_method', target='modification', args=(
                decomposed['finalreduce'].args[0],  # ops handler
                dependency_graph,
                decomposed['localbuf']
            ), kwargs={
                'ops': 'OpsHandlerOverride',  # subgraph ops handler
                'localbuf': decomposed['localbuf'],
                **{f'stale_{name}': decomposed_ancestors[name]['localbuf'] for name in decomposed_ancestors},
                **{f'updated_{name}': decomposed_ancestors[name]['finalreduce'] for name in decomposed_ancestors},
            }
        )
    decomposed['finalreduce'].args = decomposed['finalreduce'].args[:-1] + (updated_partial,)  # replace last arg of combine



    # 3. clean up
    # graph.eliminate_dead_code() requires owning module to have nn.Module properties
    for node in reversed(graph.nodes):
        if node.op not in {"placeholder", "output"} and len(node.users) == 0:
            graph.erase_node(node)
    return True


# This happens before fusion because this optimization rewrites IR graph
def pre_fusion_custom_pass(snodes: list[SchedulerNode]) -> list[SchedulerNode]:
    fuse_reductions(snodes)
    V.graph.scheduler.dead_node_elimination()
    return V.graph.scheduler.nodes

import torch._inductor.config
torch._inductor.config._pre_fusion_custom_pass = pre_fusion_custom_pass

def reductions_fit(node: SchedulerNode, other: SchedulerNode):
    """Adapted from SIMDScheduling.generate_node_schedule.<locals>.fits_in_main_body"""
    _, (numel, rnumel) = other.group
    _, (node_numel, node_rnumel) = node.group
    return (node_numel == numel and node_rnumel == rnumel) or (
        node_numel == numel * rnumel and node_rnumel == 1
    )

def check_ancestors(node: SchedulerNode):
    if not node.ancestors:
        return False
    for anc_name in node.ancestors:
        anc = node.scheduler.name_to_node[anc_name]
        if not anc.is_reduction() or not reductions_fit(node, anc):
            return False
    return True

def fuse_reductions(snodes: list[SchedulerNode]) -> list[SchedulerNode]:
    if len(snodes) < 2:
        return snodes

    for node in snodes:
        if not node.is_reduction() or not check_ancestors(node):
            continue
        # node: reduction node with ancestors which are all fit reductions
        to_discard: set[str] = set()
        for anc_name in node.ancestors:
            anc: SchedulerNode = node.scheduler.name_to_node[anc_name]
            output_names_to_keep = {x.get_name() for x in anc.get_outputs()
                if all(u.node != node for u in x.users) or len(x.users) > 1}
            if eliminate_reduction_dependency(
                node._body.root_block.graph,
                anc._body.root_block.graph,
                anc_out_to_keep=output_names_to_keep
            ):
                buffers_to_keep = [anc.outputs_by_name.pop(name) for name in output_names_to_keep]
                node.outputs = buffers_to_keep + node.outputs
                for name, buf in zip(output_names_to_keep, buffers_to_keep):
                    assert name not in node.outputs_by_name, f" {node} sharing same output buffer {name} with ancestor {anc}"
                    node.outputs_by_name[name] = buf
                    anc.outputs.remove(buf)
                to_discard.add(anc_name)

        if len(node.ancestors) == len(to_discard):  # only if all dependencies fit
            for anc_name in to_discard:
                anc = node.scheduler.name_to_node[anc_name]
                for obuf in anc.get_outputs():
                    obuf.users = [u for u in obuf.users if u.get_name() != node.get_name()]

            node.ancestors.clear()
    snodes = [snode for snode in snodes if snode.get_outputs()]
    return snodes