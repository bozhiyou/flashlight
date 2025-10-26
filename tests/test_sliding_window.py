import torch
import math

def attention_pytorch_sliding_window(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window_size: int = 1024,
    attn_mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    scale: float = None,
    enable_gqa: bool = False
) -> torch.Tensor:
    """
    PyTorch implementation matching FlexAttention's generate_sliding_window:
    - Left-only attention (causal)
    - Sliding window of size `window_size` behind each token
    """
    N, H, L, D = query.shape
    assert L == key.size(-2) == value.size(-2)

    scale_factor = 1.0 / math.sqrt(D) if scale is None else scale

    if enable_gqa:
        # Reshape query to align with groups
        # (N, Hq, L, E) -> (N, Hk, num_groups, L, E)
        query = query.view(query.size(0), key.size(1), -1, query.size(-2), query.size(-1))
        # (N, Hk, S, E) -> (N, Hk, 1, S, E)
        key = key.unsqueeze(2)
        # (N, Hk, S, Ev) -> (N, Hk, 1, S, Ev)
        value = value.unsqueeze(2)

    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor  # (N, H, L, L)

    attn_scores = attn_scores.masked_fill(attn_mask, -1e10)

    # Apply softmax and dropout
    attn_weights = torch.softmax(attn_scores, dim=-1)

    return attn_weights @ value

def get_sliding_mask(query, window_size):
    # Create (L, L) mask: True where j ∉ [i - window_size, i]
    N, H, L, D = query.shape
    q_idx = torch.arange(L, device=query.device).view(L, 1)
    k_idx = torch.arange(L, device=query.device).view(1, L)
    causal_sliding_mask = (q_idx < k_idx) | ((q_idx - k_idx) > window_size)  # shape (L, L)

    # Expand to (1, 1, L, L)
    full_mask = causal_sliding_mask.unsqueeze(0).unsqueeze(0)
    return full_mask


if __name__ == '__main__':
    # comment the line to disable the patch
    from monkeypatch.fusion import dependent_reduction_fusion
    from monkeypatch.fusion import block_reduction
    from monkeypatch.fusion import reduction_kernel_fusion

    from torch.testing import assert_close, make_tensor
    DEVICE = torch.device("cuda:0")
    BATCH = 4
    HEAD = 16
    GROUP_SIZE = 8
    N_CTX = 1024
    HEAD_DIM = 64
    DTYPE = torch.bfloat16
    # torch.set_float32_matmul_precision('high')

    q = make_tensor((BATCH, HEAD, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)
    k = make_tensor((BATCH, HEAD // GROUP_SIZE, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)
    v = make_tensor((BATCH, HEAD // GROUP_SIZE, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)

    o0 = attention_pytorch_sliding_window(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), window_size=256, attn_mask=get_sliding_mask(q, 256), enable_gqa=(q.size(1) != k.size(1))
    )
    o1 = torch.compile(dynamic=False)(attention_pytorch_sliding_window)(
        q, k, v, window_size=256, attn_mask=get_sliding_mask(q, 256), enable_gqa=(q.size(1) != k.size(1))
    )
    assert_close(o0, o1.to(torch.float32), atol=1e-2, rtol=1e-2)

    print("done")
