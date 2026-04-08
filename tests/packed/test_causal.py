"""
Test packed-doc causal attention: eager fp32 reference vs compiled bf16.
"""
import torch
from attention_variants.packed import attention_packed_causal, build_packed_causal_mask


if __name__ == '__main__':
    from monkeypatch import disable_flashattention_replacement
    from monkeypatch.fusion import dependent_reduction_fusion
    from monkeypatch.fusion import block_reduction
    from monkeypatch.fusion import reduction_kernel_fusion

    from torch.testing import assert_close, make_tensor
    DEVICE = torch.device("cuda:0")
    HEAD = 16
    HEAD_KV = 2
    N_CTX = 1024
    HEAD_DIM = 128
    DTYPE = torch.bfloat16
    disable_flashattention_replacement()

    # Packed documents: 3 docs of lengths 300, 400, 324 = 1024
    doc_lengths = [300, 400, 324]
    offsets = torch.tensor([0, 300, 700, 1024], device=DEVICE, dtype=torch.int64)
    doc_id = torch.cat([torch.full((l,), i, device=DEVICE, dtype=torch.int64)
                        for i, l in enumerate(doc_lengths)])

    q = make_tensor((1, HEAD, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)
    k = make_tensor((1, HEAD_KV, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)
    v = make_tensor((1, HEAD_KV, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE, requires_grad=False)

    mask = build_packed_causal_mask(doc_id, offsets, N_CTX, DEVICE)

    o0 = attention_packed_causal(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), mask
    )
    o1 = torch.compile(dynamic=False)(attention_packed_causal)(q, k, v, mask)
    assert_close(o0, o1.to(torch.float32), atol=2e-2, rtol=1e-2)

    print("done")
