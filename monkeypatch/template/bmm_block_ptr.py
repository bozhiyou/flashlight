from .. import _monkey as monkey

import itertools
import functools
import collections
from typing import Union, List, Tuple, Optional, Any

import torch
import torch._inductor.select_algorithm  # import before bmm to avoid circular import
import torch._inductor.kernel.bmm
from torch._inductor.virtualized import V
from torch._inductor.codegen.triton import BlockParameters, BlockPtrOptions
from torch._dynamo.utils import counters
from torch._inductor.utils import sympy_index_symbol, IndentedBuffer


import torch._inductor.config
# @bozhiyou: this option does not make big differences so far; use it to control
torch._inductor.config.triton.use_block_ptr = True
torch._inductor.config.max_autotune_gemm = True
torch._inductor.config.max_autotune_gemm_backends.replace("ATEN",'')  # disable external aten kernel for test purposes


## override template
from torch._inductor.kernel.bmm import bmm_grid
from torch._inductor.select_algorithm import TritonTemplate
TritonTemplate.all_templates.pop("bmm")  # hack registration
torch._inductor.kernel.bmm.bmm_template = TritonTemplate(
    name="bmm",
    grid=bmm_grid,
    source=r"""
{{def_kernel("A", "B")}}
    M = {{size("A", -2)}}
    N = {{size("B", -1)}}
    K = {{size("A", -1)}}

    stride_aq = {{stride("A", 0)}}
    stride_am = {{stride("A", 1)}}
    stride_ak = {{stride("A", 2)}}

    stride_bq = {{stride("B", 0)}}
    stride_bk = {{stride("B", 1)}}
    stride_bn = {{stride("B", 2)}}

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    idx_m = pid_m * BLOCK_M
    idx_n = pid_n * BLOCK_N

    idx_q = tl.program_id(1)  # batch dimension for BMM
    # O = tl.make_block_ptr(
    #     base=out_ptr0 + idx_q*M*N,  # stride_oq = M*N
    #     shape=(M, N),
    #     strides=(N, 1),
    #     offsets=(idx_m, idx_n),
    #     block_shape=(BLOCK_M, BLOCK_N),
    #     order=(1, 0),
    # )
    {{make_output_block_ptr("O",
        ("idx_q",),
        offsets=("idx_m", "idx_n"),
        block_shape=("BLOCK_M", "BLOCK_N"),
    ) | indent_except_first(1)}}
    # A = tl.make_block_ptr(
    #     base=A + idx_q*stride_aq,
    #     shape=(M, K),
    #     strides=(stride_am, stride_ak),
    #     offsets=(idx_m, 0),
    #     block_shape=(BLOCK_M, BLOCK_K),
    #     order=(1, 0),
    # )
    {{make_input_block_ptr("A",
        ("idx_q",),
        offsets=("idx_m", 0),
        block_shape=("BLOCK_M", "BLOCK_K"),
    ) | indent_except_first(1)}}
    # B = tl.make_block_ptr(
    #     base=B + idx_q*stride_bq,
    #     shape=(K, N),
    #     strides=(stride_bk, stride_bn),
    #     offsets=(0, idx_n),
    #     block_shape=(BLOCK_K, BLOCK_N),
    #     order=(1, 0),
    # )
    {{make_input_block_ptr("B",
        ("idx_q",),
        offsets=(0, "idx_n"),
        block_shape=("BLOCK_K", "BLOCK_N"),
    ) | indent_except_first(1)}}


    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        a = tl.load(A)
        b = tl.load(B)
        acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32)
        A = tl.advance(A, (0, BLOCK_K))
        B = tl.advance(B, (BLOCK_K, 0))

    {{store_output("O", "acc")}}
""",
debug=True
)


import torch._inductor.codegen.simd
@monkey.patch(torch._inductor.codegen.simd.IterationRangesEntry)
def set_name(self: torch._inductor.codegen.simd.IterationRangesEntry,
    name
):
    V.kernel.range_tree_nodes.pop(self.symbol())  # == self
    var_index = self.var_list.index(self.symbol())
    length = self.var_ranges.pop(self.symbol())
    monkey.fallback(self, name)
    self.var_ranges[self.symbol()] = length
    self.var_list[var_index] = self.symbol()
    V.kernel.range_tree_nodes[self.symbol()] = self


def make_output_block_ptr(self: torch._inductor.select_algorithm.TritonTemplateKernel,
    name: str,
    constant_offset_indexing: tuple[str],
    block_shape: tuple[str | int],
    offsets: tuple[str | int]
):
    counters.setdefault("inductor", collections.Counter())["block_ptr_bmm"] += 1
    assert isinstance(name, str)
    assert torch._inductor.config.triton.use_block_ptr and self.allow_block_ptr, "block_ptr disabled"
    assert not self._load_mask, "additional mask disables block_ptr"

    # initalize output indexing
    output_range_tree = self.range_trees[0]
    assert not output_range_tree.nodes
    indices = constant_offset_indexing + offsets
    lengths = [V.graph.sizevars.simplify(s) for s in self.output_node.get_size()]
    for iname, range_tree_entry in zip(
            indices, output_range_tree.construct_entries(lengths)
        ):
            range_tree_entry: torch._inductor.codegen.simd.IterationRangesEntry
            range_tree_entry.set_name(iname)  # set indexing varname

    output = IndentedBuffer()
    strides = self.output_node.get_stride()
    constant_offset = sum(sympy_index_symbol(x) * strides[i] for i, x in enumerate(constant_offset_indexing))
    constant_offset_var = self.cse.generate(output, str(constant_offset))
    # NOTE @bozhiyou self.indexing unnecessarily checks mask_vars for block_ptr
    indexing = BlockPtrOptions.create(
        params=BlockParameters(
            shape=lengths[-len(offsets):],
            strides=strides[-len(offsets):],
            block_shape=[sympy_index_symbol(str(x)) for x in block_shape],
            offsets=[sympy_index_symbol(str(x)) for x in offsets],
        ),
        constant_offset=sympy_index_symbol(constant_offset_var.name),
        range_trees=self.active_range_trees(reorder=True),  # for reshaping
        mask_vars=None,  # NOTE @bozhiyou this parameter seems redundant
    )
    output.writeline(f"{name} = {indexing.format(
        self.args.output(self.output_node.get_name()))}")  # shortcut of self.codegen_block_ptr
    return output.getvalue()


def make_input_block_ptr(self: torch._inductor.select_algorithm.TritonTemplateKernel,
    name: str,
    constant_offset_indexing: tuple[str],
    block_shape: tuple[str | int],
    offsets: tuple[str | int]
):
    assert isinstance(name, str)
    assert torch._inductor.config.triton.use_block_ptr and self.allow_block_ptr, "block_ptr disabled"
    assert not self._load_mask, "additional mask disables block_ptr"

    assert self.range_trees[0].nodes, "make_output_block_ptr first"

    node = self.named_input_nodes[name]
    output = IndentedBuffer()
    lengths = [V.graph.sizevars.simplify(s) for s in node.get_size()]
    strides = node.get_stride()
    constant_offset = sum(sympy_index_symbol(x) * strides[i] for i, x in enumerate(constant_offset_indexing))
    constant_offset_var = self.cse.generate(output, str(constant_offset))
    # NOTE @bozhiyou self.indexing unnecessarily checks mask_vars for block_ptr
    indexing = BlockPtrOptions.create(
        params=BlockParameters(
            shape=lengths[-len(offsets):],
            strides=strides[-len(offsets):],
            block_shape=[sympy_index_symbol(str(x)) for x in block_shape],
            offsets=[sympy_index_symbol(str(x)) for x in offsets],
        ),
        constant_offset=sympy_index_symbol(constant_offset_var.name),
        range_trees=self.active_range_trees(reorder=True),  # for reshaping
        mask_vars=None,  # NOTE @bozhiyou this parameter seems redundant
    )
    output.writeline(f"{name} = {indexing.format(name)}")  # shortcut of self.codegen_block_ptr
    return output.getvalue()


@monkey.patch(torch._inductor.select_algorithm.TritonTemplateKernel)
def template_env(self: torch._inductor.select_algorithm.TritonTemplateKernel):
    """
    Generate the namespace visible in the template.
    """
    return monkey.fallback(self) | {
        f.__name__: functools.partial(f, self)
        for f in [
            make_input_block_ptr,
            make_output_block_ptr,
        ]
    }


@monkey.patch(torch._inductor.select_algorithm.TritonTemplateKernel, 'store_output')
def store_output(
    self: torch._inductor.select_algorithm.TritonTemplateKernel,
    indices: Union[List[Any], Tuple[Any]],
    val: str,
    mask: Optional[str] = None,
    indent_width: int = 4,
):
    with self.create_subgraph_body("<STORE_OUTPUT>"):
        self.template_mask = mask
        self.template_out = val

        epilogue_args = [val]
        for input_node in itertools.chain(
            self.input_nodes[: self.prefix_args],
            self.input_nodes[len(self.input_nodes) - self.suffix_args :],
        ):
            input_node.freeze_layout()
            epilogue_args.append(input_node.make_loader()(sympy_index_symbol(indices)))

        # store line by kernel
        # V.ops.store(
        #     self.output_node.get_name(),
        #     output_index,
        #     self.epilogue_fn(*epilogue_args),
        # )
        # self.codegen_body()
        self.render_hooks["<STORE_OUTPUT>"] = lambda: f"tl.store({indices}, {self.epilogue_fn(*epilogue_args)}.to({indices}.dtype.element_ty))"
    return "<STORE_OUTPUT>"

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests, TestCase
    from torch.testing import assert_close, make_tensor
    import torch._inductor.config as inductor_config

    class BmmBlockPtrTest(TestCase):
        def _test_bmm_block_ptr(self, dtype):
            def bmm(a, b):
                return torch.bmm(a, b)

            # The original test used a large N_CTX, which is good for benchmarking.
            # For a unit test, smaller values are better for faster execution.
            for batch, m, k, n in [
                (2, 1024, 128, 512),
                (1, 2048, 64, 256),
                (4, 512, 256, 1024),
                (2, 64, 32, 128),  # smaller case
            ]:
                with self.subTest(batch=batch, m=m, k=k, n=n, dtype=dtype):
                    a = make_tensor((batch, m, k), dtype=dtype, device=self.device)
                    b = make_tensor((batch, k, n), dtype=dtype, device=self.device)

                    o0 = bmm(a, b)

                    counters.clear()
                    # The patch enables autotuning and disables ATEN to test the Triton template
                    # It also enables use_block_ptr
                    with inductor_config.patch({
                        "triton.use_block_ptr": True,
                        "max_autotune_gemm": True,
                        "max_autotune_gemm_backends": "TRITON",
                    }):
                        o1 = torch.compile(bmm)(a, b)

                    assert_close(o0, o1)
                    self.assertEqual(counters["inductor"]["block_ptr_bmm"], 1)

        def test_bmm_block_ptr_bf16(self):
            self._test_bmm_block_ptr(torch.bfloat16)

        def test_bmm_block_ptr_fp16(self):
            self._test_bmm_block_ptr(torch.float16)

    run_tests(needs="filelock")
