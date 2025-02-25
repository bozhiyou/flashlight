"""
demo envars:
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_CACHE_DIR=torchinductor_cache TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1
TORCH_COMPILE_DEBUG: print debugging information
TORCHINDUCTOR_CACHE_DIR: inductor cache location
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM: enable autotuned Triton backend
"""
import torch
import torch.testing
from torch.testing import assert_close, make_tensor

import torch._inductor.config
torch._inductor.config.max_autotune_gemm = True

# comment the line to disable bmm fusion
from monkeypatch import fuse_bmm

DEVICE = torch.device("cuda:0")
BATCH = 2
N_CTX = 64*1024
HEAD_DIM = 128
DTYPE = torch.bfloat16

def mmm(a, b, c):
    return torch.bmm(torch.bmm(a, b), c)


def main():
    q = make_tensor((BATCH, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)
    k = make_tensor((BATCH, HEAD_DIM, N_CTX), dtype=DTYPE, device=DEVICE, requires_grad=False)
    v = make_tensor((BATCH, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)

    o0 = mmm(q, k, v)
    o1 = torch.compile(mmm)(q, k, v)
    assert_close(o0, o1)

if __name__ == '__main__':
    main()
    print("done")
