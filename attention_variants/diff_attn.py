"""Differential Attention (Ye et al. 2024)"""
import torch
import math


def diffattn(q, k, v, scale=None, lambda_full=0.2, **kwargs):
    """
    q (bsz, 2 * self.num_heads, tgt_len, head_dim)
    k (bsz, 2 * self.num_kv_heads, self.head_dim, src_len)
    v (bsz, self.num_kv_heads, src_len, 2 * self.head_dim)
    """
    scale_factor = 1 / math.sqrt(q.size(-1)) if scale is None else scale

    q0 = q[:, :q.size(1) // 2, :, :]
    q1 = q[:, q.size(1) // 2:, :, :]
    k0 = k[:, :k.size(1) // 2, :, :]
    k1 = k[:, k.size(1) // 2:, :, :]

    attn_weights0 = torch.matmul(q0, k0.transpose(-1, -2)) * scale_factor
    attn_weights1 = torch.matmul(q1, k1.transpose(-1, -2)) * scale_factor

    attn_weights0 = torch.softmax(attn_weights0, dim=-1).type_as(
        attn_weights0
    )
    attn_weights1 = torch.softmax(attn_weights1, dim=-1).type_as(
        attn_weights1
    )

    attn = torch.matmul(attn_weights0, v) - lambda_full * torch.matmul(attn_weights1, v)
    return attn


def make_input(config, attention_name='diffattn'):
    batch_size, seqlen, nheads, headdim, causal, dropout_p = config
    Q = torch.randn(batch_size, nheads, seqlen, headdim, dtype=torch.bfloat16, device="cuda", requires_grad=False)
    K = torch.randn(batch_size, nheads, seqlen, headdim, dtype=torch.bfloat16, device="cuda", requires_grad=False)
    V = torch.randn(batch_size, nheads // 2, seqlen, headdim * 2, dtype=torch.bfloat16, device="cuda", requires_grad=False)
    return Q, K, V
