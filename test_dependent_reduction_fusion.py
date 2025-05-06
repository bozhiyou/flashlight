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

DEVICE = torch.device("cuda:0")
BATCH = 2
N_CTX = 2048
DTYPE = torch.float32

import torch.nn.functional as F
def softmax(x, dim=None):
    return F.softmax(x, dim)

def main():
    a = make_tensor((BATCH, N_CTX, N_CTX), dtype=DTYPE, device=DEVICE, requires_grad=False)

    o0 = softmax(a, dim=-1)
    o1 = torch.compile(softmax)(a, dim=-1)
    assert_close(o0, o1)

if __name__ == '__main__':
    main()
    print("done")
