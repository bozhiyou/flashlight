import torch
import math

def get_sliding_mask(window_size, query, key=None):
    # Matches attention-gym FlexAttention mask:
    #   causal_mask:     q_idx >= kv_idx
    #   sliding_window:  q_idx - kv_idx <= window_size
    #   combined:        (q_idx >= kv_idx) AND (q_idx - kv_idx <= window_size)
    #
    # Right-align q_idx so position 0 of Q maps to absolute position (Sk - Sq),
    # which is correct for both prefill (Sq == Sk) and decode (Sq < Sk).
    seq_len_q = query.size(-2)
    seq_len_k = key.size(-2) if key is not None else seq_len_q
    offset = seq_len_k - seq_len_q
    q_idx = torch.arange(seq_len_q, device=query.device).view(seq_len_q, 1) + offset
    kv_idx = torch.arange(seq_len_k, device=query.device).view(1, seq_len_k)
    # Mask out = NOT attend; invert the combined attend condition
    causal_sliding_mask = (q_idx < kv_idx) | ((q_idx - kv_idx) > window_size)  # (Sq, Sk)

    return causal_sliding_mask.unsqueeze(0).unsqueeze(0)

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
    scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale

    if enable_gqa:
        query = query.view(query.size(0), key.size(1), -1, query.size(-2), query.size(-1))
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)

    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor  # (N, H, Sq, Sk)

    attn_mask = get_sliding_mask(window_size, query, key)

    attn_scores = attn_scores.masked_fill(attn_mask, -1e10)

    attn_weights = torch.softmax(attn_scores, dim=-1)

    return attn_weights @ value
