import torch
from torch._dynamo.utils import counters
from torch._inductor import ir, lowering as L
from torch._inductor.kernel.bmm import tuned_bmm, mm_args

import sympy


# Unregister bmm
# torch.ops.aten.bmm: torch._ops.OpOverloadPacket
for overload in torch.ops.aten.bmm.overloads():
    other_fn = getattr(torch.ops.aten.bmm, overload)
    L.lowerings.pop(other_fn)  # tuned_bmm

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
        return mat1_loader(mat1_index) * mat2_loader(mat2_index)

    return dict(
        input_node=(mat1, mat2),
        device=layout.device,
        dst_dtype=layout.dtype,  # TODO @bozhiyou match torch._inductor.kernel.mm_common.acc_type
        src_dtype=layout.dtype,
        inner_fn=loader,
        ranges=layout.size,
        reduction_ranges=[k],
    )

@L.register_lowering(torch.ops.aten.bmm)
def bmm(mat1, mat2, *, layout=None):
    f"""Adapted from {L.make_reduction}.<locals>.inner."""
    counters["inductor"]["bmm_as_reduction"] += 1
    if all(x.get_device().type == "cpu" for x in [mat1, mat2]):
        return tuned_bmm(mat1, mat2, layout=layout)

    kwargs = _make_bmm_inner(*mm_args(mat1, mat2, layout=layout))
    result = ir.Reduction.create(reduction_type='sum', **kwargs)
    if isinstance(
        result.data.data, ir.Reduction
    ):  # Only realize if reduction isn't unrolled
        result.realize()
    return result

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests, TestCase
    from torch.testing import assert_close, make_tensor
    import torch._inductor.config as inductor_config

    class BmmAsReductionTest(TestCase):
        def _test_bmm_as_reduction(self, dtype):
            def bmm_wrapper(a, b):
                return torch.bmm(a, b)

            # The original test used a large N_CTX, which is good for benchmarking.
            # For a unit test, smaller values are better for faster execution.
            for batch, m, k, n in [
                (2, 1024, 128, 1024),
                (1, 2048, 64, 2048),
                (4, 512, 256, 512),
                (2, 64, 32, 64),  # smaller case
            ]:
                with self.subTest(batch=batch, m=m, k=k, n=n, dtype=dtype):
                    a = make_tensor((batch, m, k), dtype=dtype, device=self.device)
                    b = make_tensor((batch, k, n), dtype=dtype, device=self.device)

                    o0 = bmm_wrapper(a, b)

                    counters.clear()
                    # The test script mentions TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1
                    # which is equivalent to max_autotune=True for reductions.
                    with inductor_config.patch({"max_autotune": True}):
                        o1 = torch.compile(bmm_wrapper)(a, b)

                    assert_close(o0, o1)
                    self.assertEqual(counters["inductor"]["bmm_as_reduction"], 1)

        def test_bmm_as_reduction_bf16(self):
            self._test_bmm_as_reduction(torch.bfloat16)

        def test_bmm_as_reduction_fp16(self):
            self._test_bmm_as_reduction(torch.float16)

    run_tests(needs="filelock")