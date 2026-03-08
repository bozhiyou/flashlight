import torch
import math

def attention_softcapped(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: None,
    attn_mask=None,
    dropout_p=0.0,
    is_causal: bool=False,
    scale=None,
    enable_gqa: bool=False,
    softcap_threshold: float = 30.0,
) -> torch.Tensor:
    r"""
    Scaled dot-product attention with soft-capping.

    Args:
        query (Tensor): Query tensor; shape :math:`(N, ..., Hq, L, E)`.
        key (Tensor): Key tensor; shape :math:`(N, ..., H, S, E)`.
        value (Tensor): Value tensor; shape :math:`(N, ..., H, S, Ev)`.
        attn_mask (optional Tensor): Broadcastable to :math:`(N,..., L, S)`.
            If bool, True means attend; if float, added to attention scores.
        dropout_p (float): Dropout probability.
        is_causal (bool): If True, apply lower-triangular causal mask.
        scale (optional float): Scale factor for logits (default: 1 / sqrt(E)).
        enable_gqa (bool): If True, enable Grouped Query Attention.
        softcap_threshold (float): Soft cap threshold; larger logits saturate to this value.

    Returns:
        Output tensor of shape :math:`(N, ..., Hq, L, Ev)`.
    """
    N, Hq, L, E = query.shape
    H = key.size(-3)
    S = key.size(-2)
    Ev = value.size(-1)

    scale_factor = 1 / math.sqrt(E) if scale is None else scale

    if enable_gqa:
        query = query.view(query.size(0), key.size(1), -1, query.size(-2), query.size(-1))
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor

    # Apply soft-capping: cap * tanh(score / cap)
    attn_weight = score_mod(attn_weight,  N, Hq, query, key)

    attn_weight = torch.softmax(attn_weight, dim=-1)

    return attn_weight @ value
