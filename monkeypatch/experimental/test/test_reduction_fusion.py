"""
demo envars:
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_CACHE_DIR=torchinductor_cache TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1
TORCH_COMPILE_DEBUG: print debugging information
TORCHINDUCTOR_CACHE_DIR: inductor cache location
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM: enable autotuned Triton backend
"""
import torch
from torch.testing import assert_close, make_tensor

# comment the line to disable the patch
from monkeypatch import dependent_reduction_fusion
from monkeypatch.experimental import block_reduction
from monkeypatch.experimental import reduction_kernel_fusion

DEVICE = torch.device("cuda:0")
BATCH = 2
N_CTX = 16*1024
HEAD_DIM = 128
DTYPE = torch.bfloat16
# torch.set_float32_matmul_precision('high')
import torch.nn.functional as F
import math
def row_reduction(a, b, c):
    # return torch.sum(a, dim=-1)
    # return torch.bmm(a, b)
    # return torch.sum(torch.bmm(a, b), dim=-1)
    return torch.bmm(F.softmax(torch.bmm(a, b) / math.sqrt(a.size(-1)), dim=-1), c)


from torch.nn.attention.flex_attention import flex_attention
def flex_wrapper(q, k, v):
    return flex_attention(q.unsqueeze(1), k.transpose(-1, -2).unsqueeze(1), v.unsqueeze(1), scale=1.0).squeeze(1)


def main():
    q = make_tensor((BATCH, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)
    k = make_tensor((BATCH, HEAD_DIM, N_CTX), dtype=DTYPE, device=DEVICE, requires_grad=False)
    v = make_tensor((BATCH, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)

    o0 = row_reduction(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32))
    o1 = torch.compile(row_reduction)(q, k, v)
    # o0 = torch.compile(row_reduction)(q, k, v)
    # o1 = torch.compile(flex_wrapper)(q, k, v)
    # assert_close(o0.to(torch.float32), o1.to(torch.float32))
    assert_close(o0, o1.to(torch.float32), atol=1e-2, rtol=1e-2)
    assert_close(o0, o1)
    # assert_close(o0[0], o1[1])
    assert_close(o0[:, :N_CTX//2, :HEAD_DIM//2], o1[:, :N_CTX//2, :HEAD_DIM//2])
    assert_close(o0[:, N_CTX//2:, HEAD_DIM//2:], o1[:, N_CTX//2:, HEAD_DIM//2:])

if __name__ == '__main__':
    main()
    print("done")
