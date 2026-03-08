import torch
import math
from typing import Optional

def attention_pytorch_prefix_lm(
    query: torch.Tensor,  # [B, H, S, D]
    key: torch.Tensor,
    value: torch.Tensor,
    prefix_lengths: int = 256,
    scale: Optional[float] = None,
    enable_gqa = False
) -> torch.Tensor:

    scale = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale

    attn_mask = get_prefix_lm_mask(query, prefix_lengths)

    if enable_gqa:
        query = query.view(query.size(0), key.size(1), -1, query.size(-2), query.size(-1))
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)
        attn_mask = attn_mask.unsqueeze(2).expand(attn_mask.size(0), key.size(1), -1, attn_mask.size(-2), attn_mask.size(-1))

    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    attn_scores = attn_scores.masked_fill(attn_mask, -1e10)

    attn_weights = torch.softmax(attn_scores, dim=-1)

    return attn_weights @ value  # [B, H, S, D]


def get_prefix_lm_mask(query, prefix_lengths):
    prefix_lengths = torch.full((query.size(0),), prefix_lengths, dtype=torch.long, device=query.device)
     # Build combined prefix-lm-causal mask: allow k <= max(prefix_len[b]-1, q)
    q_idx = torch.arange(query.size(-2), device=query.device).view(1, 1, query.size(-2), 1)  # [1, 1, S, 1]
    k_idx = torch.arange(query.size(-2), device=query.device).view(1, 1, 1, query.size(-2))  # [1, 1, 1, S]
    prefix_idx = prefix_lengths.view(query.size(0), 1, 1, 1) - 1  # [B, 1, 1, 1]

    max_idx = torch.maximum(prefix_idx, q_idx)  # [B, 1, S, 1]
    causal_prefix_mask = k_idx > max_idx  # [B, 1, S, S]
    return causal_prefix_mask
