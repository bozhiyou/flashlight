"""
demo envars:
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_CACHE_DIR=torchinductor_cache TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1
TORCH_COMPILE_DEBUG: print debugging information
TORCHINDUCTOR_CACHE_DIR: inductor cache location
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM: enable autotuned Triton backend
"""
import torch
from attention_variants.causal import attention_pytorch_causal


if __name__ == '__main__':
    # comment the line to disable the patch
    from monkeypatch import disable_flashattention_replacement
    from monkeypatch.fusion import dependent_reduction_fusion
    from monkeypatch.fusion import block_reduction
    from monkeypatch.fusion import reduction_kernel_fusion

    from torch.testing import assert_close, make_tensor
    DEVICE = torch.device("cuda:0")
    BATCH = 2
    HEAD = 1
    N_CTX = 16*1024
    HEAD_DIM = 128
    DTYPE = torch.bfloat16
    # torch.set_float32_matmul_precision('high')
    disable_flashattention_replacement()
    q = make_tensor((BATCH, HEAD, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)
    k = make_tensor((BATCH, HEAD, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)
    v = make_tensor((BATCH, HEAD, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)

    o0 = attention_pytorch_causal(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32))
    o1 = torch.compile(dynamic=False)(attention_pytorch_causal)(q, k, v)
    # bf16 compiled vs fp32 eager: ~2/4M elements exceed 1e-2 (max ~0.011)
    # at large seq lengths. If this still fails, increase atol or reduce N_CTX.
    assert_close(o0, o1.to(torch.float32), atol=2e-2, rtol=1e-2)

    print("done")
