import torch
import math

def attention_pytorch_nogqa(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale=None) -> torch.Tensor:
    r"""
    Args:
        query (Tensor): Query tensor; shape :math:`(N, ..., Hq, L, E)`.
        key (Tensor): Key tensor; shape :math:`(N, ..., H, S, E)`.
        value (Tensor): Value tensor; shape :math:`(N, ..., H, S, Ev)`.
        scale (optional float, keyword-only): Scaling factor applied prior to softmax. If None, the default value is set
            to :math:`\frac{1}{\sqrt{E}}`.
    """
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value


def attention_pytorch(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale=None, enable_gqa=False) -> torch.Tensor:
    r"""
    Args:
        enable_gqa (bool): If set to True, Grouped Query Attention (GQA) is enabled, by default it is set to False.
    """
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    if enable_gqa:
        query = query.view(query.size(0), key.size(1), -1, query.size(-2), query.size(-1))
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return (attn_weight @ value).view(value.size(0), -1, value.size(-2), value.size(-1))
