"""
demo envars:
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_CACHE_DIR=torchinductor_cache TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1
TORCH_COMPILE_DEBUG: print debugging information
TORCHINDUCTOR_CACHE_DIR: inductor cache location
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM: enable autotuned Triton backend
"""
import torch
import math


def alibi_bias(h, q_len, kv_len, device='cuda'):
    return torch.exp2(
            -((torch.arange(h, dtype=torch.int32, device=device) + 1) * 8.0 / h)
        )[:, None, None] * (
            torch.arange(kv_len, dtype=torch.int32, device=device)[None, :] - torch.arange(q_len, dtype=torch.int32, device=device)[:, None])

def generate_alibi_bias_pytorch(nheads): return lambda q, k: alibi_bias(nheads, q, k)

def attention_pytorch_alibi(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, score_mod=None,
        attn_mask=None, dropout_p=0.0, is_causal: bool=False, scale=None, enable_gqa=False) -> torch.Tensor:
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    if enable_gqa:
        # Reshape query to align with groups
        # (N, Hq, L, E) -> (N, Hk, num_groups, L, E)
        query = query.view(query.size(0), key.size(1), -1, query.size(-2), query.size(-1))
        # (N, Hk, S, E) -> (N, Hk, 1, S, E)
        key = key.unsqueeze(2)
        # (N, Hk, S, Ev) -> (N, Hk, 1, S, Ev)
        value = value.unsqueeze(2)

    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor

    # `to(attn_weight.dtype)` here is only necessary for eager execution; patched torch.compile handles this type conversion implicitly
    alibi_bias = score_mod(attn_weight.size(-2), attn_weight.size(-1)).to(attn_weight.dtype)
    if enable_gqa:
        alibi_bias = alibi_bias.view(attn_weight.size(1), attn_weight.size(2), attn_weight.size(3), attn_weight.size(4))
    attn_weight = attn_weight + alibi_bias

    attn_weight = torch.softmax(attn_weight, dim=-1)

    # Cast to value's dtype before matmul (ensuring dtype match)
    # attn_weight = attn_weight.to(value.dtype)

    # Matrix multiply with value tensor
    return attn_weight @ value


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
    assert_close(o0, o1.to(torch.float32), atol=1e-2, rtol=1e-2)

    print("done")
