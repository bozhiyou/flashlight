"""
debug envvars:
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_CACHE_DIR=torchinductor_cache TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1
TORCH_COMPILE_DEBUG: print debugging information
TORCHINDUCTOR_CACHE_DIR: inductor cache location
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM: enable autotuned Triton backend
"""
import torch
import math

def attention_pytorch(
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
    # L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    # attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


if __name__ == '__main__':
    # comment the line to disable the patch
    from monkeypatch import disable_flashattention_replacement
    from monkeypatch.fusion import dependent_reduction_fusion
    from monkeypatch.fusion import block_reduction
    from monkeypatch.fusion import reduction_kernel_fusion

    from torch.testing import assert_close, make_tensor
    DEVICE = torch.device("cuda:0")
    BATCH = 2
    HEAD = 1
    N_CTX = 16*1024
    HEAD_DIM = 128
    DTYPE = torch.bfloat16
    # torch.set_float32_matmul_precision('high')
    disable_flashattention_replacement()
    q = make_tensor((BATCH, HEAD, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)
    k = make_tensor((BATCH, HEAD, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)
    v = make_tensor((BATCH, HEAD, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)

    o0 = attention_pytorch(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32))
    o1 = torch.compile(attention_pytorch)(q, k, v)
    assert_close(o0, o1.to(torch.float32), atol=1e-2, rtol=1e-2)

    print("done")
