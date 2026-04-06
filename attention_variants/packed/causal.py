"""Packed-doc causal attention for the Flashlight e2e backend.

Layout: ``(1, H, S, D)`` with a precomputed ``(S, S)`` boolean mask.

The mask is built *outside* the compiled boundary (data-dependent indexing
into ``doc_id`` / ``offsets`` trips the fusion patches' block-reduction pass).
The compiled function receives only ``(q, k, v, attn_mask)`` — the same
pattern as the kernel-bench ``attention_variants/causal.py``, which Flashlight
is proven to fuse into a single Triton kernel.
"""
import math

import torch


def build_packed_causal_mask(
    doc_id: torch.Tensor,
    offsets: torch.Tensor,
    S: int,
    device: torch.device,
) -> torch.Tensor:
    """Build ``(S, S)`` bool mask (True = mask out) for packed-doc causal.

    Called in eager Python before the compiled attention function.
    """
    idx = torch.arange(S, device=device)
    same_doc = doc_id[idx[:, None]] == doc_id[idx[None, :]]
    q_local = idx[:, None] - offsets[doc_id[idx[:, None]]]
    kv_local = idx[None, :] - offsets[doc_id[idx[None, :]]]
    causal = q_local >= kv_local
    return ~(same_doc & causal)


def attention_packed_causal(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Causal attention with a precomputed packed-doc mask.

    Args:
        query, key, value: ``(1, H, S, D)`` (or 5D after GQA reshape)
        attn_mask: ``(S, S)`` bool — True means mask out
        scale: optional scaling factor (default ``1/sqrt(D)``)
    """
    S = query.size(-2)
    scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale

    gqa = query.size(1) != key.size(1)
    if gqa:
        query = query.view(1, key.size(1), -1, S, query.size(-1))
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)

    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    attn_weight = attn_weight.masked_fill(attn_mask, float("-inf"))
    attn_weight = torch.softmax(attn_weight, dim=-1)

    out = attn_weight @ value
    if gqa:
        out = out.reshape(1, -1, S, out.size(-1))
    return out
