import torch
import math
from typing import Optional, List, Literal, Union

def attention_pytorch_prefix_lm(
    query: torch.Tensor,  # [B, H, S, D]
    key: torch.Tensor,
    value: torch.Tensor,
    prefix_lengths: Union[int, torch.Tensor],  # scalar or [B]
    attn_mask = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    training: bool = False,
) -> torch.Tensor:
    B, H, S, D = query.shape

    # if isinstance(prefix_lengths, int):
    #     prefix_lengths = torch.full((B,), prefix_lengths, dtype=torch.long, device=query.device)
    # assert prefix_lengths.shape == (B,), f"Expected prefix_lengths shape [B], got {prefix_lengths.shape}"

    # Scale factor
    scale = scale or (1.0 / math.sqrt(D))

    # Compute attention scores: [B, H, S, S]
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # # Build combined prefix-lm-causal mask: allow k <= max(prefix_len[b]-1, q)
    # q_idx = torch.arange(S, device=query.device).view(1, 1, S, 1)  # [1, 1, S, 1]
    # k_idx = torch.arange(S, device=query.device).view(1, 1, 1, S)  # [1, 1, 1, S]
    # prefix_idx = prefix_lengths.view(B, 1, 1, 1) - 1  # [B, 1, 1, 1]

    # max_idx = torch.maximum(prefix_idx, q_idx)  # [B, 1, S, 1]
    # causal_prefix_mask = k_idx > max_idx  # [B, 1, S, S]

    attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))

    # Compute softmax over attention scores
    attn_weights = torch.softmax(attn_scores, dim=-1)

    return attn_weights @ value  # [B, H, S, D]


def get_prefix_lm_mask(query, prefix_lengths):
    B, H, S, D = query.shape
    prefix_lengths = torch.full((B,), prefix_lengths, dtype=torch.long, device=query.device)
     # Build combined prefix-lm-causal mask: allow k <= max(prefix_len[b]-1, q)
    q_idx = torch.arange(S, device=query.device).view(1, 1, S, 1)  # [1, 1, S, 1]
    k_idx = torch.arange(S, device=query.device).view(1, 1, 1, S)  # [1, 1, 1, S]
    prefix_idx = prefix_lengths.view(B, 1, 1, 1) - 1  # [B, 1, 1, 1]

    max_idx = torch.maximum(prefix_idx, q_idx)  # [B, 1, S, 1]
    causal_prefix_mask = k_idx > max_idx  # [B, 1, S, S]
    return causal_prefix_mask