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
from monkeypatch import bmm_block_ptr_template

DEVICE = torch.device("cuda:0")
BATCH = 2
N_CTX = 16*1024
HEAD_DIM = 128
DTYPE = torch.bfloat16

def bmm_wrapper(a, b):
    return torch.bmm(a, b)


def main():
    q = make_tensor((BATCH, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)
    k = make_tensor((BATCH, HEAD_DIM, N_CTX), dtype=DTYPE, device=DEVICE, requires_grad=False)

    o0 = bmm_wrapper(q, k)
    o1 = torch.compile(bmm_wrapper)(q, k)
    assert_close(o0, o1)

if __name__ == '__main__':
    main()
    print("done")
