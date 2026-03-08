import torch
import math

def attention_pytorch_causal(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    attn_mask=None, dropout_p=0.0, is_causal: bool=False, scale=None, enable_gqa=False) -> torch.Tensor:
    r"""
    Args:
        query (Tensor): Query tensor; shape :math:`(N, ..., Hq, L, E)`.
        key (Tensor): Key tensor; shape :math:`(N, ..., H, S, E)`.
        value (Tensor): Value tensor; shape :math:`(N, ..., H, S, Ev)`.
        attn_mask (optional Tensor): Attention mask; shape must be broadcastable to the shape of attention weights,
            which is :math:`(N,..., L, S)`. Two types of masks are supported.
            A boolean mask where a value of True indicates that the element *should* take part in attention.
            A float mask of the same type as query, key, value that is added to the attention score.
        dropout_p (float): Dropout probability; if greater than 0.0, dropout is applied
        is_causal (bool): If set to true, the attention masking is a lower triangular matrix when the mask is a
            square matrix. The attention masking has the form of the upper left causal bias due to the alignment
            (see :class:`torch.nn.attention.bias.CausalBias`) when the mask is a non-square matrix.
            An error is thrown if both attn_mask and is_causal are set.
        scale (optional float, keyword-only): Scaling factor applied prior to softmax. If None, the default value is set
            to :math:`\frac{1}{\sqrt{E}}`.
        enable_gqa (bool): If set to True, Grouped Query Attention (GQA) is enabled, by default it is set to False.
    """
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    if enable_gqa:
        query = query.view(query.size(0), key.size(1), -1, query.size(-2), query.size(-1))
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)

    mask = torch.triu(torch.ones(query.size(-2), key.size(-2), device=query.device, dtype=torch.bool), diagonal=1)
    attn_weight = torch.where(mask, float('-inf'), query @ key.transpose(-2, -1) * scale_factor)

    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value
