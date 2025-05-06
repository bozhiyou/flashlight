from . import _monkey as monkey

import functools
from collections import deque
from typing import List, Set, Dict

import torch
import torch.fx
import torch._inductor.lowering  # avoid circular import
import torch._inductor.ir
from torch._inductor.fx_passes.post_grad import pass_patterns
from torch._inductor.pattern_matcher import PatternMatcherPass, register_graph_pattern
from torch._inductor.pattern_matcher import Match, CallFunction, Arg
from torch._inductor.ir import Subgraph, FixedLayout
from torch._dynamo.utils import counters
from torch._inductor.select_algorithm import TritonTemplate, TritonTemplateCaller, autotune_select_algorithm
from torch._inductor.kernel.mm_common import acc_type

import torch._inductor.config
torch._inductor.config.pattern_matcher = True
torch._inductor.config.max_autotune_gemm = True
torch._inductor.config.max_autotune_gemm_backends.replace("ATEN",'')  # disable external aten kernel for test purposes


BMM_FUSION_PASS = PatternMatcherPass(pass_name="bmm_fusion_pass")
pass_patterns.append(BMM_FUSION_PASS)

@register_graph_pattern(
    CallFunction(torch.ops.aten.bmm, Arg(), Arg()),
    pass_dict=BMM_FUSION_PASS,
)
def bmm_fusion_matcher(match: Match, mat1: torch.fx.Node, mat2: torch.fx.Node) -> None:
    r"""
    Pattern:

      [external inputs]
                    X  (no external inputs)
     [inner_mm]-[pointwise]-*[outer_mm]
           \*
           [pointwise]* (inner_mm may have epilogues)

    `match` matches `inner_mm`. This function matches the pattern above.

    Let `inner_mm` represent `A @ B`, `outer_mm` represent `f(A @ B) @ C`.
    `f_node` is the node representing `f(A @ B)`, which is on the path
    from `inner_mm` to `outer_mm` and is the immediate predecessor of
    `outer_mm`.
    `f` can be a chain of pointwise nodes.
    when the pattern is simply (A @ (B @ C)), f_node is just inner_mm
    """
    # match.args: list[torch.fx.Node]
    inner_mm: torch.fx.Node = match.nodes[-1]
    outer_mm: torch.fx.Node | None = None
    subgraph_node_set: Set[torch.fx.Node] = set([inner_mm])
    external_inputs: Set[torch.fx.Node] = set()

    def is_pointwise_node(node: torch.fx.Node) -> bool:
        return (
            node.op == "call_function"
            and isinstance(node.target, torch._ops.OpOverload)
            and (torch.Tag.pointwise in node.target.tags)
        )

    def is_bmm(node: torch.fx.Node) -> bool:
        return node.target == torch.ops.aten.bmm.default
    
    # BFS for outer MM.
    queue = deque(inner_mm.users.keys())
    while queue:
        node = queue.popleft()
        if node in subgraph_node_set:
            continue
        # At most one outer MM
        if is_bmm(node) and outer_mm in (None, node):
            outer_mm = node
            continue
        # No non-pointwise node (except for outer MM)
        if not is_pointwise_node(node):
            return
        # external input tracking
        for inp in node.all_input_nodes:
            if inp not in subgraph_node_set:
                external_inputs.add(inp)
        if node in external_inputs:
            external_inputs.remove(node)
        subgraph_node_set.add(node)
        queue.extend(node.users)

    # At least one outer MM and no external inputs
    if outer_mm is None or external_inputs:
        return

    for f_node in outer_mm.all_input_nodes:
        if f_node in subgraph_node_set:
            break  # @bozhiyou: two f_node case?

    # check inner_mm's inputs and f_node's outputs
    # if not (len(inner_mm.all_input_nodes) == 2 and len(f_node.users) == 1):
    #     return
    _rewrite(match, subgraph_node_set, inner_mm, f_node, outer_mm)

def _rewrite(match: Match, subgraph_node_set: Set[torch.fx.Node], inner_mm: torch.fx.Node, f_node: torch.fx.Node, outer_mm: torch.fx.Node):
    # at this point, the nodes between inner_mm and f_node (both included)
    # are all used internally inside (A @ subgraph(B @ C))
    # i.e. they neither have external users nor have external inputs

    # original graph and module
    graph, module = match.graph, match.graph.owning_module

    # construct the new (sub)graph
    subgraph_node_list: List[torch.fx.Node] = []  # ordered list of nodes used for node removal later
    new_graph: torch.fx.Graph = torch.fx.Graph()
    node_remapping: Dict[torch.fx.Node, torch.fx.Node] = {}
    for node in graph.nodes:  # preserve the order of nodes
        if node not in subgraph_node_set:
            continue
        subgraph_node_list.append(node)
        node_remapping[node] = new_graph.node_copy(
            node, lambda x: node_remapping.get(x, x)
        )

    # add the output node
    new_output_anchor: torch.fx.Node = node_remapping[f_node]
    new_output_node: torch.fx.Node = new_graph.output(new_output_anchor)
    new_output_node.meta.update(new_output_anchor.meta)

    # update the input node
    new_input_anchor: torch.fx.Node = node_remapping[inner_mm]
    with new_graph.inserting_before(new_input_anchor):
        new_input_node: torch.fx.Node = new_graph.placeholder(name="subgraph_input")
        new_input_node.meta.update(new_input_anchor.meta)
        new_input_anchor.replace_all_uses_with(new_input_node)
    new_graph.erase_node(new_input_anchor)

    new_graph.lint()

    # construct the subgraph
    subgraph = Subgraph(
        name="subgraph", graph_module=torch.fx.GraphModule(module, new_graph)
    )

    # two cases
    # (1) (subgraph(A @ B) @ C), called "left_assoc"
    # (2) (A @ subgraph(B @ C)), called "right_assoc"
    is_left_assoc = outer_mm.args[0] is f_node

    # find the nodes A, B, C and check the sizes
    A: torch.fx.Node
    B: torch.fx.Node
    C: torch.fx.Node
    if is_left_assoc:
        A = inner_mm.args[0]  # type: ignore[assignment]
        B = inner_mm.args[1]  # type: ignore[assignment]
        C = outer_mm.args[1]  # type: ignore[assignment]
    else:
        A = outer_mm.args[0]  # type: ignore[assignment]
        B = inner_mm.args[0]  # type: ignore[assignment]
        C = inner_mm.args[1]  # type: ignore[assignment]
    # TODO is_good_on(is_left_assoc, A, B, C)

    # finally update the original graph
    counters["inductor"]["bmm_fusion"] += 1
    with graph.inserting_before(outer_mm):
        function = functools.partial(fused_caller_gen, is_left_assoc=is_left_assoc, subgraph=subgraph)
        function.__name__ = fused_caller_gen.__name__  # type: ignore[attr-defined]
        function._inductor_lowering_function = True  # type: ignore[attr-defined]
        replacement: torch.fx.Node = graph.call_function(
            function,
            (A, B, C),
            match.kwargs,
        )
        replacement.meta.update(outer_mm.meta)
        outer_mm.replace_all_uses_with(replacement)
    # erase unnecessary nodes
    graph.erase_node(outer_mm)
    for node in reversed(subgraph_node_list):
        graph.erase_node(node)
    graph.lint()

from torch._inductor.fx_passes.b2b_gemm import  build_subgraph_buffer, create_placeholder

def fused_caller_gen(
    A: torch._inductor.ir.TensorBox,
    B: torch._inductor.ir.TensorBox,
    C: torch._inductor.ir.TensorBox,
    is_left_assoc: bool,
    subgraph: Subgraph,
    *,
    layout=None,
) -> torch._inductor.ir.TensorBox:
    # call .realize() to get rid of Pointwise
    A.realize()
    B.realize()
    C.realize()
    layout = FixedLayout(A.get_device(), A.get_dtype(), [A.shape[0], A.shape[-2], C.shape[-1]])
    subgraph_buffer = build_subgraph_buffer(
        [create_placeholder("inner_mm", A.get_dtype(), A.get_device())],
        subgraph,
    )
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32 and (
        not torch._inductor.config.force_same_precision or (
        (A.shape[-2] % 16) == 0 and (C.shape[-1] % 16) == 0 and (B.shape[-2] % 8) == 0 and (B.shape[-1] % 8) == 0)
    )
    choices: list[TritonTemplateCaller] = []
    for config in fused_bmm_configs:
        if is_left_assoc:
            fused_bmm_template.maybe_append_choice(
                choices,
                input_nodes=(A, B, C),
                layout=layout,
                subgraphs=[subgraph_buffer],
                ALLOW_TF32=allow_tf32,
                ACC_TYPE=acc_type(layout.dtype),
                **config,
            )
        # TODO right assoc
    # TODO add the unoptimized choice to mitigate performance degradation
    # from torch._inductor.fx_passes.b2b_gemm import unoptimized_choice
    # choices.append(
    #     unoptimized_choice.bind(
    #         (A, B, C), layout, is_left_assoc=is_left_assoc, subgraph=subgraph
    #     )
    # )
    # autotune
    return autotune_select_algorithm("b2b_bmm", choices, [A, B, C], layout)


fused_bmm_configs = [
    {
        "BLOCK_M": 128,
        "BLOCK_N": 16,
        "BLOCK_O": 16,
        "BLOCK_P": 16,
        "num_stages": 4,
        "num_warps": 8,
    },
    {
        "BLOCK_M": 128,
        "BLOCK_N": 32,
        "BLOCK_O": 32,
        "BLOCK_P": 32,
        "num_stages": 2,
        "num_warps": 4,
    },
    {
        "BLOCK_M": 128,
        "BLOCK_N": 64,
        "BLOCK_O": 64,
        "BLOCK_P": 64,
        "num_stages": 2,
        "num_warps": 4,
    },
    {
        "BLOCK_M": 128,
        "BLOCK_N": 16,
        "BLOCK_O": 128,
        "BLOCK_P": 16,
        "num_stages": 4,
        "num_warps": 8,
    },
    {
        "BLOCK_M": 128,
        "BLOCK_N": 32,
        "BLOCK_O": 128,
        "BLOCK_P": 32,
        "num_stages": 2,
        "num_warps": 4,
    },
    {
        "BLOCK_M": 128,
        "BLOCK_N": 64,
        "BLOCK_O": 128,
        "BLOCK_P": 64,
        "num_stages": 2,
        "num_warps": 4,
    },
    {
        "BLOCK_M": 16,
        "BLOCK_N": 16,
        "BLOCK_O": 16,
        "BLOCK_P": 128,
        "num_stages": 4,
        "num_warps": 8,
    },
    {
        "BLOCK_M": 32,
        "BLOCK_N": 32,
        "BLOCK_O": 32,
        "BLOCK_P": 128,
        "num_stages": 2,
        "num_warps": 4,
    },
    {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_O": 64,
        "BLOCK_P": 128,
        "num_stages": 2,
        "num_warps": 4,
    },
    {
        "BLOCK_M": 16,
        "BLOCK_N": 128,
        "BLOCK_O": 16,
        "BLOCK_P": 128,
        "num_stages": 4,
        "num_warps": 8,
    },
    {
        "BLOCK_M": 32,
        "BLOCK_N": 128,
        "BLOCK_O": 32,
        "BLOCK_P": 128,
        "num_stages": 2,
        "num_warps": 4,
    },
    {
        "BLOCK_M": 64,
        "BLOCK_N": 128,
        "BLOCK_O": 64,
        "BLOCK_P": 128,
        "num_stages": 2,
        "num_warps": 4,
    },
]

from torch._inductor.utils import ceildiv as cdiv
def fused_bmm_grid(b, m, p, meta):
    return (cdiv(m, meta["BLOCK_M"]) * cdiv(p, meta["BLOCK_P"]), b, 1)

# fused kernel template
fused_bmm_template = TritonTemplate(
    name="fused_bmm",
    grid=fused_bmm_grid,
    source=r"""
{{def_kernel("A", "B", "C")}}
    M = {{size("A", -2)}}
    N = {{size("A", -1)}}
    O = {{size("C", -2)}}
    P = {{size("C", -1)}}

    stride_aq = {{stride("A", 0)}}
    stride_am = {{stride("A", -2)}}
    stride_an = {{stride("A", -1)}}

    stride_bq = {{stride("B", 0)}}
    stride_bn = {{stride("B", -2)}}
    stride_bo = {{stride("B", -1)}}

    stride_cq = {{stride("C", 0)}}
    stride_co = {{stride("C", -2)}}
    stride_cp = {{stride("C", -1)}}

    pid = tl.program_id(0)
    # output block counts
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_p = tl.cdiv(P, BLOCK_P)
    # internal block counts
    num_n_block = tl.cdiv(N, BLOCK_N)
    num_o_block = tl.cdiv(O, BLOCK_O)

    # output block ids
    pid_m = pid // grid_p
    pid_p = pid % grid_p

    # in-block indices
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rp = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    if (stride_am == 1 and stride_an == M) or (stride_am == N and stride_an == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_co == 1 and stride_cp == O) or (stride_co == P and stride_cp == 1):
        rcp = tl.max_contiguous(tl.multiple_of(rp % P, BLOCK_P), BLOCK_P)
    else:
        rcp = rp % P
    

    # (subgraph(A @ B) @ C)
    rn = tl.arange(0, BLOCK_N)
    ro = tl.arange(0, BLOCK_O)
    idx_q = tl.program_id(1)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rn[None, :] * stride_an + idx_q * stride_aq)
    B = B + (rn[:, None] * stride_bn + ro[None, :] * stride_bo + idx_q * stride_aq)
    C = C + (ro[:, None] * stride_co + rcp[None, :] * stride_cp + idx_q * stride_cq)

    Abase = A  # backup
    acc = tl.zeros((BLOCK_M, BLOCK_P), dtype=ACC_TYPE)
    for o in range(O, 0, -BLOCK_O):
        Bbase = B  # backup
        acc_ab = tl.zeros((BLOCK_M, BLOCK_O), dtype=ACC_TYPE)
        for n in range(N, 0, -BLOCK_N):
            a = tl.load(A, mask=rn[None, :] < n, other=0.)  # .to(tl.float32)  # BLOCK_M * BLOCK_N
            b = tl.load(B, mask=(rn[:, None] < n) & (ro[None, :] < o), other=0.)  #.to(tl.float32)  # BLOCK_N * BLOCK_O
            acc_ab = tl.dot(a, b, acc_ab, allow_tf32=ALLOW_TF32)
            A += BLOCK_N * stride_an
            B += BLOCK_N * stride_bn
        # apply the subgraph
        {{ modification(
            subgraph_number=0,
            output_name="post_subgraph_acc_ab",
            inner_mm="acc_ab"
        ) | indent_except_first(2) }}

        c = tl.load(C, mask=ro[:, None] < o, other=0.)  #.to(tl.float32)  # BLOCK_O * BLOCK_P
        acc = tl.dot(post_subgraph_acc_ab.to(c.type.element_ty), c, acc, allow_tf32=ALLOW_TF32)
        A = Abase  # reset A
        B = Bbase + BLOCK_O * stride_bo
        C += BLOCK_O * stride_co

    # store preparation
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rp = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    idx_q = tl.program_id(1)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_p = rp[None, :]
    out_mask = (idx_m < M) & (idx_p < P)

    {{store_output(("idx_q", "idx_m", "idx_p"), "acc", "out_mask")}}
""",
debug=True
)
