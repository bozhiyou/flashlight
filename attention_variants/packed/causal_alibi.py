"""Packed-doc causal + ALiBi attention for the Flashlight e2e backend.

Same split as ``causal.py``: mask and bias built eagerly, attention compiled.
"""
import math

import torch


def build_packed_causal_alibi_mask_and_bias(
    doc_id: torch.Tensor,
    offsets: torch.Tensor,
    S: int,
    H_q: int,
    device: torch.device,
    dtype: torch.dtype,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Build ``(S, S)`` bool mask and ``(H_q, S, S)`` ALiBi bias."""
    idx = torch.arange(S, device=device)
    same_doc = doc_id[idx[:, None]] == doc_id[idx[None, :]]
    q_local = idx[:, None] - offsets[doc_id[idx[:, None]]]
    kv_local = idx[None, :] - offsets[doc_id[idx[None, :]]]
    causal = q_local >= kv_local
    mask = ~(same_doc & causal)

    slopes = torch.exp2(
        -((torch.arange(H_q, device=device, dtype=torch.int32) + 1) * 8.0 / H_q)
    )
    distance = (kv_local - q_local).to(dtype)
    alibi = slopes[:, None, None] * distance[None, :, :]  # (H_q, S, S)
    return mask, alibi


def attention_packed_causal_alibi(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor,
    alibi_bias: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Causal + ALiBi attention with precomputed mask and bias.

    Args:
        attn_mask: ``(S, S)`` bool — True means mask out
        alibi_bias: ``(H_q, S, S)`` additive bias
    """
    S = query.size(-2)
    scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale

    gqa = query.size(1) != key.size(1)
    if gqa:
        H_kv = key.size(1)
        G = query.size(1) // H_kv
        query = query.view(1, H_kv, G, S, query.size(-1))
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)
        alibi_bias = alibi_bias.view(H_kv, G, S, S).unsqueeze(0)
    else:
        alibi_bias = alibi_bias.unsqueeze(0)

    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    attn_weight = attn_weight + alibi_bias
    attn_weight = attn_weight.masked_fill(attn_mask, float("-inf"))
    attn_weight = torch.softmax(attn_weight, dim=-1)

    out = attn_weight @ value
    if gqa:
        out = out.reshape(1, -1, S, out.size(-1))
    return out
