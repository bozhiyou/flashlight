"""
debug envvars:
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_CACHE_DIR=torchinductor_cache TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1
TORCH_COMPILE_DEBUG: print debugging information
TORCHINDUCTOR_CACHE_DIR: inductor cache location
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM: enable autotuned Triton backend
"""
import torch
import math

def grouped_query_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale=None
) -> torch.Tensor:
    r"""
    Args:
        query (Tensor): Query tensor; shape :math:`(N, ..., Hq, L, E)`.
        key (Tensor): Key tensor; shape :math:`(N, ..., Hk, S, E)`.
        value (Tensor): Value tensor; shape :math:`(N, ..., Hk, S, Ev)`.
        attn_mask (optional Tensor): Attention mask; shape must be broadcastable to the shape of attention weights,
            which is :math:`(N,..., Hq, L, S)`. Two types of masks are supported.
            A boolean mask where a value of True indicates that the element *should* take part in attention.
            A float mask of the same type as query, key, value that is added to the attention score.
        dropout_p (float): Dropout probability; if greater than 0.0, dropout is applied
        is_causal (bool): If set to true, the attention masking is a lower triangular matrix when the mask is a
            square matrix. The attention masking has the form of the upper left causal bias due to the alignment
            (see :class:`torch.nn.attention.bias.CausalBias`) when the mask is a non-square matrix.
            An error is thrown if both attn_mask and is_causal are set.
        scale (optional float, keyword-only): Scaling factor applied prior to softmax. If None, the default value is set
            to :math:`\frac{1}{\sqrt{E}}`.
        enable_gqa (bool): If set to True, Grouped Query Attention (GQA) is enabled.
                           Assumes Hq (query heads) is a multiple of Hk (key/value heads).
    """
    # L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    # Reshape query to align with groups
    # (N, Hq, L, E) -> (N, Hk, num_groups, L, E)
    query = query.view(query.size(0), key.size(1), -1, query.size(-2), query.size(-1))
    # (N, Hk, S, E) -> (N, Hk, 1, S, E)
    key = key.unsqueeze(2)
    # (N, Hk, S, Ev) -> (N, Hk, 1, S, Ev)
    value = value.unsqueeze(2)

    # Calculate attention weights
    # (N, Hk, num_groups, L, E) @ (N, Hk, 1, E, S) -> (N, Hk, num_groups, L, S)
    attn_weight = (query @ key.transpose(-2, -1)) * scale_factor
    
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    # Apply attention weights to values
    # (N, Hk, num_groups, L, S) @ (N, Hk, 1, S, Ev) -> (N, Hk, num_groups, L, Ev)
    output = attn_weight @ value

    # Reshape output back to original query head dimension
    # (N, Hk, num_groups, L, Ev) -> (N, Hq, L, Ev)
    return output.view(value.size(0), -1, value.size(-2), value.size(-1))


if __name__ == '__main__':
    # comment the line to disable the patch
    from monkeypatch import disable_flashattention_replacement
    from monkeypatch.fusion import dependent_reduction_fusion
    from monkeypatch.fusion import block_reduction
    from monkeypatch.fusion import reduction_kernel_fusion

    from torch.testing import assert_close, make_tensor
    DEVICE = torch.device("cuda")
    BATCH = 32
    HEAD_Q = 16       # Query heads
    HEAD_KV = 2      # Key/Value heads (must divide HEAD_Q)
    N_CTX = 4*1024
    HEAD_DIM = 64
    DTYPE = torch.bfloat16

    # Ensure query heads is a multiple of key/value heads
    if HEAD_Q % HEAD_KV != 0:
        raise ValueError(f"Query heads {HEAD_Q} must be a multiple of Key/Value heads {HEAD_KV} for GQA.")
    
    # torch.set_float32_matmul_precision('high')
    disable_flashattention_replacement()

    print(f"GQA with {HEAD_Q} query heads and {HEAD_KV} key/value heads.")

    q = make_tensor((BATCH, HEAD_Q, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)
    k = make_tensor((BATCH, HEAD_KV, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)
    v = make_tensor((BATCH, HEAD_KV, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)

    o0 = grouped_query_attention(
        q.to(torch.float32), 
        k.to(torch.float32), 
        v.to(torch.float32), 
    )
    o1 = torch.compile(dynamic=False)(grouped_query_attention)(q, k, v)
    
    assert_close(o0, o1.to(torch.float32), atol=1e-2, rtol=1e-2)

    print("GQA test done")
