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


def attention_pytorch_alibi(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    return torch.matmul(query, key) + torch.arange(key.size(-1),  dtype=DTYPE, device=DEVICE)


def main():
    q = make_tensor((BATCH, 2, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)
    k = make_tensor((BATCH, 2, HEAD_DIM, N_CTX), dtype=DTYPE, device=DEVICE, requires_grad=False)

    o0 = attention_pytorch_alibi(q, k)
    o1 = torch.compile(attention_pytorch_alibi)(q, k)
    # assert_close(o0.to(torch.float32), o1.to(torch.float32))
    # assert_close(o0, o1.to(torch.float32), atol=1e-2, rtol=1e-2)
    assert_close(o0, o1)
    # assert_close(o0[0], o1[1])
    assert_close(o0[:, :N_CTX//2, :HEAD_DIM//2], o1[:, :N_CTX//2, :HEAD_DIM//2])
    assert_close(o0[:, N_CTX//2:, HEAD_DIM//2:], o1[:, N_CTX//2:, HEAD_DIM//2:])

if __name__ == '__main__':
    main()
    print("done")
