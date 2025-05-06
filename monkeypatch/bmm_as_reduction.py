from . import _monkey as monkey

import torch
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
    if all(x.get_device().type == "cpu" for x in [mat1, mat2]):
        return tuned_bmm(mat1, mat2, layout=layout)

    kwargs = _make_bmm_inner(*mm_args(mat1, mat2, layout=layout))
    result = ir.Reduction.create(reduction_type='sum', **kwargs)
    if isinstance(
        result.data.data, ir.Reduction
    ):  # Only realize if reduction isn't unrolled
        result.realize()
    return result