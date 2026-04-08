"""Packed-doc causal + ALiBi attention for the Flashlight e2e backend.

Same split as ``causal.py``: mask built eagerly, attention compiled.

Only ``local_pos`` (1D, length S) is precomputed eagerly — it requires
``doc_id``/``offsets`` indirect indexing that trips ``block_reduction``.
The ALiBi bias itself (slopes × distance) is computed inline inside the
compiled function, matching the kernel-bench pattern that TorchInductor
fuses into the tiled kernel.
"""
import math

import torch


def build_packed_causal_alibi_mask(
    doc_id: torch.Tensor,
    offsets: torch.Tensor,
    S: int,
    device: torch.device,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Build ``(S, S)`` additive float mask and ``(S,)`` local-position vector.

    Returns:
        attn_mask: ``(S, S)`` float32 — 0 for valid, ``-inf`` for masked-out
        local_pos: ``(S,)`` int64 — document-relative position of each token
    """
    idx = torch.arange(S, device=device)
    same_doc = doc_id[idx[:, None]] == doc_id[idx[None, :]]
    q_local = idx[:, None] - offsets[doc_id[idx[:, None]]]
    kv_local = idx[None, :] - offsets[doc_id[idx[None, :]]]
    causal = q_local >= kv_local
    bool_mask = ~(same_doc & causal)

    attn_mask = torch.where(bool_mask, -1e10, 0.0)  # (S, S) float

    local_pos = idx - offsets[doc_id[idx]]  # (S,)
    return attn_mask, local_pos


def attention_packed_causal_alibi(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor,
    local_pos: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Causal + ALiBi attention with precomputed additive mask; ALiBi inline.

    Args:
        attn_mask: ``(S, S)`` float — 0 for valid, ``-inf`` for masked-out
        local_pos: ``(S,)`` int64 — document-relative position per token
    """
    S = query.size(-2)
    H_q = query.size(1)
    scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale

    # ALiBi slopes and bias — computed inline, fused by TorchInductor
    slopes = torch.exp2(
        -((torch.arange(H_q, device=query.device, dtype=torch.int32) + 1)
          * 8.0 / H_q)
    )
    distance = local_pos[None, :] - local_pos[:, None]       # (S, S)
    alibi = slopes[:, None, None] * distance[None, :, :]     # (H_q, S, S)

    gqa = query.size(1) != key.size(1)
    if gqa:
        H_kv = key.size(1)
        G = H_q // H_kv
        query = query.view(1, H_kv, G, S, query.size(-1))
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)
        alibi = alibi.view(H_kv, G, S, S).unsqueeze(0)
    else:
        alibi = alibi.unsqueeze(0)

    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor + alibi + attn_mask
    attn_weight = torch.softmax(attn_weight, dim=-1)

    out = attn_weight @ value
    if gqa:
        out = out.reshape(1, -1, S, out.size(-1))
    return out
