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
        query = query.view(query.size(0), key.size(1), -1, query.size(-2), query.size(-1))
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)

    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor

    # `to(attn_weight.dtype)` here is only necessary for eager execution; patched torch.compile handles this type conversion implicitly
    alibi_bias = score_mod(attn_weight.size(-2), attn_weight.size(-1)).to(attn_weight.dtype)
    if enable_gqa:
        alibi_bias = alibi_bias.view(attn_weight.size(1), attn_weight.size(2), attn_weight.size(3), attn_weight.size(4))
    attn_weight = attn_weight + alibi_bias

    attn_weight = torch.softmax(attn_weight, dim=-1)

    return attn_weight @ value
