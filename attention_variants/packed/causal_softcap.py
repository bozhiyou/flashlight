"""Packed-doc causal + softcap attention for the Flashlight e2e backend.

Same split as ``causal.py``: mask built eagerly, attention compiled.
"""
import math

import torch


def build_packed_causal_mask(
    doc_id: torch.Tensor,
    offsets: torch.Tensor,
    S: int,
    device: torch.device,
) -> torch.Tensor:
    """Build ``(S, S)`` bool mask (True = mask out) for packed-doc causal."""
    idx = torch.arange(S, device=device)
    same_doc = doc_id[idx[:, None]] == doc_id[idx[None, :]]
    q_local = idx[:, None] - offsets[doc_id[idx[:, None]]]
    kv_local = idx[None, :] - offsets[doc_id[idx[None, :]]]
    causal = q_local >= kv_local
    return ~(same_doc & causal)


def attention_packed_causal_softcap(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor,
    softcap: float = 30.0,
    scale: float | None = None,
) -> torch.Tensor:
    """Causal + tanh soft-capped attention with precomputed mask."""
    S = query.size(-2)
    scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale

    gqa = query.size(1) != key.size(1)
    if gqa:
        query = query.view(1, key.size(1), -1, S, query.size(-1))
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)

    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    attn_weight = softcap * torch.tanh(attn_weight / softcap)
    attn_weight = attn_weight.masked_fill(attn_mask, float("-inf"))
    attn_weight = torch.softmax(attn_weight, dim=-1)

    out = attn_weight @ value
    if gqa:
        out = out.reshape(1, -1, S, out.size(-1))
    return out
