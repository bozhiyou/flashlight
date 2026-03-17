import torch
import random
from .utils import FlashInferWrapper

def get_document_mask(batch_size, seqlen, device="cuda"):
    """
    Creates a block-diagonal mask based on random document lengths combined with a causal mask.
    Returns a flattened 1D boolean tensor as expected by FlashInfer.
    """
    num_docs = 12
    # Ensure reproducibility for benchmark
    rng = random.Random(0)
    
    # Generate random document lengths
    lengths = [1] * num_docs
    remaining_length = seqlen - num_docs
    for _ in range(remaining_length):
        index = rng.randint(0, num_docs - 1)
        lengths[index] += 1
        
    # Create document ids
    doc_ids = torch.repeat_interleave(
        torch.arange(num_docs, device=device, dtype=torch.int32),
        torch.tensor(lengths, device=device, dtype=torch.int32)
    )
    
    # Create boolean mask (True means tokens are in same document)
    # Shape: (seqlen, seqlen)
    same_doc_mask = doc_ids.unsqueeze(1) == doc_ids.unsqueeze(0)
    
    # Create causal mask (True means valid causal connection)
    causal_mask = torch.tril(torch.ones(seqlen, seqlen, dtype=torch.bool, device=device))
    
    # Combine masks (True means not masked out)
    final_mask = same_doc_mask & causal_mask
    
    # Repeat for batch size and flatten
    return final_mask.expand(batch_size, seqlen, seqlen).flatten()

attention_flashinfer_document_mask = FlashInferWrapper(custom_mask_fn=get_document_mask)
