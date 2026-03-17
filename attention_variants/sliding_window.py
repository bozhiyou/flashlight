import torch
import math

def get_sliding_mask(query, window_size):
    # Create (L, L) mask: True where j not in [i - window_size, i]
    q_idx = torch.arange(query.size(-2), device=query.device).view(query.size(-2), 1)
    k_idx = torch.arange(query.size(-2), device=query.device).view(1, query.size(-2))
    causal_sliding_mask = (q_idx < k_idx) | ((q_idx - k_idx) > window_size)  # shape (L, L)

    # Expand to (1, 1, L, L)
    full_mask = causal_sliding_mask.unsqueeze(0).unsqueeze(0)
    return full_mask

def attention_pytorch_sliding_window(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window_size: int = 1024,
    scale = None,
    enable_gqa: bool = False
) -> torch.Tensor:
    """
    PyTorch implementation matching FlexAttention's generate_sliding_window:
    - Left-only attention (causal)
    - Sliding window of size `window_size` behind each token
    """
    assert query.size(-2) == key.size(-2) == value.size(-2)

    scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale

    if enable_gqa:
        query = query.view(query.size(0), key.size(1), -1, query.size(-2), query.size(-1))
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)

    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor  # (N, H, L, L)

    attn_mask = get_sliding_mask(query, window_size)

    attn_scores = attn_scores.masked_fill(attn_mask, -1e10)

    attn_weights = torch.softmax(attn_scores, dim=-1)

    return attn_weights @ value
