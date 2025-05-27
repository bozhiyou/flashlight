"""
demo envars:
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_CACHE_DIR=torchinductor_cache TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1
TORCH_COMPILE_DEBUG: print debugging information
TORCHINDUCTOR_CACHE_DIR: inductor cache location
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM: enable autotuned Triton backend
"""
import torch
from torch.testing import assert_close, make_tensor

# comment the line to disable the patch
from monkeypatch import dependent_reduction_fusion
from monkeypatch.experimental import block_reduction
from monkeypatch.experimental import reduction_kernel_fusion

DEVICE = torch.device("cuda:0")
BATCH = 2
HEAD = 2
N_CTX = 16*1024
HEAD_DIM = 128
DTYPE = torch.bfloat16
# torch.set_float32_matmul_precision('high')

from torch.nn.attention.flex_attention import flex_attention
from attn_gym.mods import generate_alibi_bias


def alibi_bias(h, q_len, kv_len):
    return torch.exp2(-((torch.arange(h, dtype=torch.int32, device=DEVICE) + 1) * 8.0 / h))[:, None, None] * (torch.arange(kv_len, dtype=torch.int32, device=DEVICE)[None, :] - torch.arange(q_len, dtype=torch.int32, device=DEVICE)[:, None])

def generate_alibi_bias_pytorch(nheads): return lambda q, k: alibi_bias(nheads, q, k)

import math
def attention_pytorch_alibi(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, score_mod=None,
        attn_mask=None, dropout_p=0.0, is_causal: bool=False, scale=None, enable_gqa=False) -> torch.Tensor:
        
    
    # Scale factor calculation
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    # # ALiBi slope generation (compiler-friendly)
    # Hq = query.size(-3)  # number of heads in query
    # device, dtype = query.device, query.dtype
    
    # # Generating the slope factor based on head index and number of heads (same as in flex_attention)
    # slopes = torch.pow(2, torch.arange(-8, -8*(Hq+1), -8, device=device) / Hq).to(dtype)

    # # Attention computation (query @ key) scaled by scale_factor
    attn_weight = torch.matmul(query, key) * scale_factor

    # # ALiBi bias injection
    # L, S = query.size(-2), key.size(-2)
    # q_idx = torch.arange(L, device=device).view(-1, 1).to(dtype)
    # k_idx = torch.arange(S, device=device).view(1, -1).to(dtype)
    # rel_dist = (q_idx - k_idx)  # [L, S]

    # Head-specific bias addition as in flex_attention
    # attn_weight = attn_weight + (slopes.view(-1, 1, 1) * rel_dist).unsqueeze(0)
    # N, Hq, L, E = query.shape

    attn_weight = attn_weight + score_mod(attn_weight.size(-2), attn_weight.size(-1))

    attn_weight = torch.softmax(attn_weight, dim=-1)

    # Cast to value's dtype before matmul (ensuring dtype match)
    # attn_weight = attn_weight.to(value.dtype)

    # Matrix multiply with value tensor
    return attn_weight @ value




def main():
    q = make_tensor((BATCH, HEAD, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)
    k = make_tensor((BATCH, HEAD, HEAD_DIM, N_CTX), dtype=DTYPE, device=DEVICE, requires_grad=False)
    v = make_tensor((BATCH, HEAD, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)
    # q.fill_(0)
    # k.fill_(0)
    # v.fill_(1)
    # o = torch.compile(generate_alibi_bias(q.size(-3)))(q, 1, torch.ones(1, dtype=DTYPE, device=DEVICE, requires_grad=False), 1, 1)
    # return
    # o0 = flex_attention(q, k.transpose(-1, -2), v, score_mod=generate_alibi_bias(q.size(-3)))
    # o1 = torch.compile(flex_attention)(q, k.transpose(-1, -2), v)
    # assert_close(o0, o1)


    # o0 = row_reduction(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32))
    # o1 = torch.compile(row_reduction)(q, k, v)
    o0 = attention_pytorch_alibi(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), score_mod=generate_alibi_bias_pytorch(q.size(-3))# dropout_p=dropout_p#, is_causal=causal
    )
    o1 = torch.compile(attention_pytorch_alibi)(
        q, k, v, score_mod=generate_alibi_bias_pytorch(q.size(-3))# dropout_p=dropout_p#, is_causal=causal
    )
    # o0 = torch.compile(row_reduction)(q, k, v)
    # o1 = torch.compile(flex_wrapper)(q, k, v)
    # assert_close(o0.to(torch.float32), o1.to(torch.float32))
    assert_close(o0, o1.to(torch.float32), atol=1e-2, rtol=1e-2)
    assert_close(o0, o1)
    # assert_close(o0[0], o1[1])
    assert_close(o0[:, 0, :N_CTX//2, :HEAD_DIM//2], o1.to(torch.float32)[:, 0, :N_CTX//2, :HEAD_DIM//2])
    assert_close(o0[:, 1, N_CTX//2:, HEAD_DIM//2:], o1.to(torch.float32)[:, 1, N_CTX//2:, HEAD_DIM//2:])

if __name__ == '__main__':
    main()
    print("done")
