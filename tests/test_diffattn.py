"""
debug envvars:
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_CACHE_DIR=torchinductor_cache TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1
TORCH_COMPILE_DEBUG: print debugging information
TORCHINDUCTOR_CACHE_DIR: inductor cache location
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM: enable autotuned Triton backend
"""
import torch
import math


def diffattn(q, k, v, scale=None, lambda_full=0.2, **kwargs):
    """
    q (bsz, 2 * self.num_heads, tgt_len, head_dim)
    k (bsz, 2 * self.num_kv_heads, self.head_dim, src_len)
    v (bsz, self.num_kv_heads, src_len, 2 * self.head_dim)
    """
    scale_factor = 1 / math.sqrt(q.size(-1)) if scale is None else scale

    q0 = q[:, :q.size(1) // 2, :, :]
    q1 = q[:, q.size(1) // 2:, :, :]
    k0 = k[:, :k.size(1) // 2, :, :]
    k1 = k[:, k.size(1) // 2:, :, :]

    attn_weights0 = torch.matmul(q0, k0.transpose(-1, -2)) * scale_factor
    attn_weights1 = torch.matmul(q1, k1.transpose(-1, -2)) * scale_factor

    attn_weights0 = torch.softmax(attn_weights0, dim=-1).type_as(
        attn_weights0
    )
    attn_weights1 = torch.softmax(attn_weights1, dim=-1).type_as(
        attn_weights1
    )

    # attn_weights = attn_weights0 - lambda_full * attn_weights1
    # attn = torch.matmul(attn_weights, v)
    attn = torch.matmul(attn_weights0, v) - lambda_full * torch.matmul(attn_weights1, v)
    return attn


def make_input(config, attention_name='diffattn'):
    batch_size, seqlen, nheads, headdim, causal, dropout_p = config
    # seqlen = seqlen // 8
    Q = torch.randn(batch_size, nheads, seqlen, headdim, dtype=torch.bfloat16, device="cuda", requires_grad=False)
    K = torch.randn(batch_size, nheads, seqlen, headdim, dtype=torch.bfloat16, device="cuda", requires_grad=False)
    V = torch.randn(batch_size, nheads // 2, seqlen, headdim * 2, dtype=torch.bfloat16, device="cuda", requires_grad=False)
    return Q, K, V


if __name__ == '__main__':
    # comment the line to disable the patch
    from monkeypatch import disable_flashattention_replacement
    disable_flashattention_replacement()
    from monkeypatch.fusion import dependent_reduction_fusion
    from monkeypatch.fusion import block_reduction
    from monkeypatch.fusion import reduction_kernel_fusion

    import torch._inductor.config
    torch._inductor.config.aggressive_fusion = True

    from torch.testing import assert_close, make_tensor
    DEVICE = torch.device("cuda:0")
    BATCH = 16
    HEAD = 16
    N_CTX = 1024
    HEAD_DIM = 128
    DTYPE = torch.bfloat16
    # torch.set_float32_matmul_precision('high')

    q = make_tensor((BATCH, HEAD, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)
    k = make_tensor((BATCH, HEAD, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)
    v = make_tensor((BATCH, HEAD // 2, N_CTX, 2 * HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)

    # use float32 result as ref
    o0 = diffattn(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32))
    o1 = torch.compile(diffattn)(q, k, v)
    assert_close(o0, o1.to(torch.float32), atol=1e-2, rtol=1e-2)

    print("done")
