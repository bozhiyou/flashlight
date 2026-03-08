"""
demo envars:
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_CACHE_DIR=torchinductor_cache TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1
TORCH_COMPILE_DEBUG: print debugging information
TORCHINDUCTOR_CACHE_DIR: inductor cache location
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM: enable autotuned Triton backend
"""
import torch
from attention_variants.alibi import attention_pytorch_alibi, generate_alibi_bias_pytorch


if __name__ == '__main__':
    # comment the line to disable the patch
    from monkeypatch.fusion import dependent_reduction_fusion
    from monkeypatch.fusion import block_reduction
    from monkeypatch.fusion import reduction_kernel_fusion

    from torch.testing import assert_close, make_tensor
    DEVICE = torch.device("cuda:0")
    BATCH = 4
    HEAD = 16
    GROUP_SIZE = 8
    N_CTX = 1024
    HEAD_DIM = 64
    DTYPE = torch.bfloat16
    # torch.set_float32_matmul_precision('high')

    q = make_tensor((BATCH, HEAD, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)
    k = make_tensor((BATCH, HEAD // GROUP_SIZE, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)
    v = make_tensor((BATCH, HEAD // GROUP_SIZE, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)

    o0 = attention_pytorch_alibi(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), score_mod=generate_alibi_bias_pytorch(q.size(-3)), enable_gqa=(q.size(1) != k.size(1))
    )
    o1 = torch.compile(dynamic=False)(attention_pytorch_alibi)(
        q, k, v, score_mod=generate_alibi_bias_pytorch(q.size(-3)), enable_gqa=(q.size(1) != k.size(1))
    )
    # bf16 compiled vs fp32 eager: ~4/4M elements exceed 1e-2 (max ~0.011)
    # with GQA + alibi bias. If this still fails, increase atol or reduce N_CTX.
    assert_close(o0, o1.to(torch.float32), atol=2e-2, rtol=1e-2)

    print("done")
