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
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor  # (N, H, L, L)

    attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))

    # Apply softmax and dropout
    attn_weights = torch.softmax(attn_scores, dim=-1)

    return attn_weights @ value

def get_sliding_mask(query, window_size):
    # Create (L, L) mask: True where j ∉ [i - window_size, i]
    N, H, L, D = query.shape
    q_idx = torch.arange(L, device=query.device).view(L, 1)
    k_idx = torch.arange(L, device=query.device).view(1, L)
    causal_sliding_mask = (q_idx < k_idx) | ((q_idx - k_idx) > window_size)  # shape (L, L)

    # Expand to (N, H, L, L)
    full_mask = causal_sliding_mask.unsqueeze(0).unsqueeze(0).expand(N, H, L, L)
    return full_mask