import torch
import math
from typing import Optional

def attention_pytorch_prefix_lm(
    query: torch.Tensor,  # [B, H, S, D]
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor,
    scale: Optional[float] = None,
    enable_gqa = False
) -> torch.Tensor:
    B, H, S, D = query.shape

    # if isinstance(prefix_lengths, int):
    #     prefix_lengths = torch.full((B,), prefix_lengths, dtype=torch.long, device=query.device)
    # assert prefix_lengths.shape == (B,), f"Expected prefix_lengths shape [B], got {prefix_lengths.shape}"

    # Scale factor
    scale = scale or (1.0 / math.sqrt(D))

    if enable_gqa:
        # Reshape query to align with groups
        # (N, Hq, L, E) -> (N, Hk, num_groups, L, E)
        query = query.view(query.size(0), key.size(1), -1, query.size(-2), query.size(-1))
        # (N, Hk, S, E) -> (N, Hk, 1, S, E)
        key = key.unsqueeze(2)
        # (N, Hk, S, Ev) -> (N, Hk, 1, S, Ev)
        value = value.unsqueeze(2)
        attn_mask = attn_mask.unsqueeze(2).expand(attn_mask.size(0), key.size(1), -1, attn_mask.size(-2), attn_mask.size(-1))

    # Compute attention scores: [B, H, S, S]
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # # Build combined prefix-lm-causal mask: allow k <= max(prefix_len[b]-1, q)
    # q_idx = torch.arange(S, device=query.device).view(1, 1, S, 1)  # [1, 1, S, 1]
    # k_idx = torch.arange(S, device=query.device).view(1, 1, 1, S)  # [1, 1, 1, S]
    # prefix_idx = prefix_lengths.view(B, 1, 1, 1) - 1  # [B, 1, 1, 1]

    # max_idx = torch.maximum(prefix_idx, q_idx)  # [B, 1, S, 1]
    # causal_prefix_mask = k_idx > max_idx  # [B, 1, S, S]

    attn_scores = attn_scores.masked_fill(attn_mask, -1e10)

    # Compute softmax over attention scores
    attn_weights = torch.softmax(attn_scores, dim=-1)

    return attn_weights @ value  # [B, H, S, D]


def get_prefix_lm_mask(query, prefix_lengths):
    B, H, S, D = query.shape
    prefix_lengths = torch.full((B,), prefix_lengths, dtype=torch.long, device=query.device)
     # Build combined prefix-lm-causal mask: allow k <= max(prefix_len[b]-1, q)
    q_idx = torch.arange(S, device=query.device).view(1, 1, S, 1)  # [1, 1, S, 1]
    k_idx = torch.arange(S, device=query.device).view(1, 1, 1, S)  # [1, 1, 1, S]
    prefix_idx = prefix_lengths.view(B, 1, 1, 1) - 1  # [B, 1, 1, 1]

    max_idx = torch.maximum(prefix_idx, q_idx)  # [B, 1, S, 1]
    causal_prefix_mask = k_idx > max_idx  # [B, 1, S, S]
    return causal_prefix_mask


if __name__ == '__main__':
    # comment the line to disable the patch
    from monkeypatch import disable_flashattention_replacement
    from monkeypatch.fusion import dependent_reduction_fusion
    from monkeypatch.fusion import block_reduction
    from monkeypatch.fusion import reduction_kernel_fusion

    from torch.testing import assert_close, make_tensor
    DEVICE = torch.device("cuda:0")
    BATCH = 32
    HEAD = 16
    GROUP_SIZE = 8
    N_CTX = 1024
    HEAD_DIM = 64
    DTYPE = torch.bfloat16
    # torch.set_float32_matmul_precision('high')
    disable_flashattention_replacement()
    q = make_tensor((BATCH, HEAD, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)
    k = make_tensor((BATCH, HEAD // GROUP_SIZE, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)
    v = make_tensor((BATCH, HEAD // GROUP_SIZE, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)

    o0 = attention_pytorch_prefix_lm(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), attn_mask=get_prefix_lm_mask(q, 256), enable_gqa=(q.size(1) != k.size(1))
    )
    o1 = torch.compile(dynamic=False)(attention_pytorch_prefix_lm)(
        q, k, v, attn_mask=get_prefix_lm_mask(q, 256), enable_gqa=(q.size(1) != k.size(1))
    )
    assert_close(o0, o1.to(torch.float32), atol=1e-2, rtol=1e-2)

    print("done")