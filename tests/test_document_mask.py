"""
Document Masking/Sample Packing/Jagged Tensors
"""
import torch
import math
import random

def attention_pytorch_document_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    document_id: torch.Tensor,
    scale = None,
) -> torch.Tensor:
    """
    Computes attention with document masking using torch.where for performance.

    Args:
        query (Tensor): shape (N, Hq, L, E)
        key (Tensor): shape (N, H, S, E)
        value (Tensor): shape (N, H, S, Ev)
        segment_ids (LongTensor): shape (N, L); segment id for each token.
    """
    scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale

    # Create the attention mask from segment IDs
    # `seg_i` has shape (N, L, 1) and `seg_j` has shape (N, 1, S)
    seg_i = document_id.unsqueeze(2)
    seg_j = document_id.unsqueeze(1)
    # `mask` is True where tokens should NOT attend (i.e., are in different documents)
    # Unsqueeze to broadcast across the head dimension: (N, L, S) -> (N, 1, L, S)
    mask = (seg_i != seg_j).unsqueeze(1) # Shape: (N, 1, L, S)

    # Calculate raw attention scores
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor  # Shape: (N, Hq, L, S)

    # 3. Apply the mask using torch.where
    # This efficiently applies -inf to invalid positions.
    # attn_scores = attn_scores.masked_fill(~mask, -1e10)
    attn_scores = torch.where(mask, -1e10, attn_scores)

    attn_weights = torch.softmax(attn_scores, dim=-1)

    return attn_weights @ value


def create_document_id(batch_size: int, seq_len: int, num_docs: int = 12, device: torch.device = torch.device('cuda')) -> torch.Tensor:
    """
    Generates a tensor of segment IDs for packing multiple documents into one sequence.
    Example: seq_len=8, num_docs=3 -> [0, 0, 0, 1, 1, 1, 2, 2] (random lengths)
    """
    if seq_len < num_docs:
        raise ValueError("Sequence length must be at least equal to the number of documents.")

    def generate_random_lengths(total_length, num_documents):
        # Initialize all lengths to 1 to ensure each document has at least one token
        lengths = [1] * num_documents
        remaining_length = total_length - num_documents

        # Randomly distribute the remaining length
        for _ in range(remaining_length):
            index = random.randint(0, num_documents - 1)
            lengths[index] += 1

        return lengths

    lengths = generate_random_lengths(seq_len, num_docs)

    # Create the segment ID tensor
    return torch.repeat_interleave(
        torch.arange(num_docs, device=device, dtype=torch.int64),
        torch.tensor(lengths, device=device, dtype=torch.int64)
    ).unsqueeze(0).expand(batch_size, -1)


if __name__ == '__main__':
    # comment the line to disable the patch
    from monkeypatch import disable_flashattention_replacement
    from monkeypatch.fusion import dependent_reduction_fusion
    from monkeypatch.fusion import block_reduction
    from monkeypatch.fusion import reduction_kernel_fusion

    from torch.testing import assert_close, make_tensor
    DEVICE = torch.device("cuda")
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

    # Generate the same segment IDs for the entire batch
    NUM_DOCS = 12 # Number of documents to pack into the sequence
    document_id = create_document_id(BATCH, N_CTX, NUM_DOCS, device=DEVICE)

    o0 = attention_pytorch_document_mask(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), document_id)
    o1 = torch.compile(dynamic=False)(attention_pytorch_document_mask)(q, k, v, document_id)
    assert_close(o0, o1.to(torch.float32), atol=1e-2, rtol=1e-2)

    print("done")
