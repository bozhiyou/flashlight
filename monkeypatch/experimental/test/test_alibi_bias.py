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
N_CTX = 1024
HEAD_DIM = 128
DTYPE = torch.bfloat16
# torch.set_float32_matmul_precision('high')

from torch.nn.attention.flex_attention import flex_attention
from attn_gym.mods import generate_alibi_bias


def alibi_bias(h, q_len, kv_len):
    return torch.exp2(-((torch.arange(h, dtype=DTYPE, device=DEVICE) + 1) * 8.0 / h))[:, None, None] * (torch.arange(kv_len, dtype=DTYPE, device=DEVICE)[None, :] - torch.arange(q_len, dtype=DTYPE, device=DEVICE)[:, None])



def main():
    q = make_tensor((BATCH, 1, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)
    k = make_tensor((BATCH, 1, HEAD_DIM, N_CTX), dtype=DTYPE, device=DEVICE, requires_grad=False)
    v = make_tensor((BATCH, 1, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)

    # o = torch.compile(generate_alibi_bias(q.size(-3)))(q, 1, torch.ones(1, dtype=DTYPE, device=DEVICE, requires_grad=False), 1, 1)
    # return
    # o0 = flex_attention(q, k.transpose(-1, -2), v, score_mod=generate_alibi_bias(q.size(-3)))
    # o1 = torch.compile(flex_attention)(q, k.transpose(-1, -2), v)
    # assert_close(o0, o1)


    # o0 = row_reduction(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32))
    # o1 = torch.compile(row_reduction)(q, k, v)
    o0 = alibi_bias(4, N_CTX, N_CTX)
    o1 = torch.compile(alibi_bias)(4, N_CTX, N_CTX)
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
