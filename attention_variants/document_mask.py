"""Document Masking/Sample Packing/Jagged Tensors"""
import torch
import math
import random

def attention_pytorch_document_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    document_id: torch.Tensor,
    scale = None,
    enable_gqa=False
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

    seg_i = document_id.unsqueeze(2)
    seg_j = document_id.unsqueeze(1)
    mask = (seg_i != seg_j).unsqueeze(1) # Shape: (N, 1, L, S)

    if enable_gqa:
        query = query.view(query.size(0), key.size(1), -1, query.size(-2), query.size(-1))
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)
        mask = mask.unsqueeze(2).expand(mask.size(0), key.size(1), -1, mask.size(-2), mask.size(-1))

    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor  # Shape: (N, Hq, L, S)

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
        lengths = [1] * num_documents
        remaining_length = total_length - num_documents

        for _ in range(remaining_length):
            index = random.randint(0, num_documents - 1)
            lengths[index] += 1

        return lengths

    lengths = generate_random_lengths(seq_len, num_docs)

    return torch.repeat_interleave(
        torch.arange(num_docs, device=device, dtype=torch.int64),
        torch.tensor(lengths, device=device, dtype=torch.int64)
    ).unsqueeze(0).expand(batch_size, -1)
