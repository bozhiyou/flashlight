"""Row-wise gated self-attention in Evoformer (Jumper et al. 2021)"""
import math
import torch
from typing import List, Optional
from torch.nn import functional as F


def attention_reference(
    q_input: torch.Tensor,  # [*, Dim_Q, H, C_hid]
    k_input: torch.Tensor,  # [*, Dim_Q, H, C_hid]
    v_input: torch.Tensor,  # [*, Dim_Q, H, C_hid]
    biases: List[torch.Tensor],
    sm_scale: Optional[float] = None,
    **kwargs
) -> torch.Tensor:
    # Original shape: [*, Dim_Q, H, C_hid] -> Transpose to: [*, H, Dim_Q, C_hid]
    q = q_input.transpose(-2, -3)
    k = k_input.transpose(-2, -3)
    v = v_input.transpose(-2, -3)

    k_t = k.transpose(-1, -2)

    sm_scale = 1 / math.sqrt(q.size(-1)) if sm_scale is None else sm_scale
    a = torch.matmul(q, k_t) * sm_scale

    for b in biases:
        a += b

    a = F.softmax(a, dim=-1)

    a_v = torch.matmul(a, v)

    # [*, Dim_Q, H, C_hid]
    o = a_v.transpose(-2, -3)

    return o


N = 256

def make_input(config, attention_name='ipa'):
    batch_size, seqlen, nheads, headdim, causal, dropout_p = config
    Q = torch.randn(batch_size, N, seqlen, nheads, headdim, dtype=torch.bfloat16, device="cuda", requires_grad=False)
    K = torch.randn(batch_size, N, seqlen, nheads, headdim, dtype=torch.bfloat16, device="cuda", requires_grad=False)
    V = torch.randn(batch_size, N, seqlen, nheads, headdim, dtype=torch.bfloat16, device="cuda", requires_grad=False)
    bias1 = torch.randn(batch_size, N, 1, 1, seqlen, dtype=torch.bfloat16, device="cuda", requires_grad=False)
    bias2 = torch.randn(batch_size, 1, nheads, seqlen, seqlen, dtype=torch.bfloat16, device="cuda", requires_grad=False)

    return Q, K, V, [bias1, bias2]
