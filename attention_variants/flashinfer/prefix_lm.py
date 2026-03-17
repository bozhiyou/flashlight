import torch
from .utils import FlashInferWrapper

def get_prefix_lm_mask(batch_size, seqlen, device="cuda"):
    """
    Creates a causal mask, but sets the first 256 rows to all True.
    Returns a flattened 1D boolean tensor as expected by FlashInfer.
    """
    # Create the causal mask
    # True means not masked out in FlashInfer custom_mask convention
    mask = torch.tril(torch.ones(seqlen, seqlen, dtype=torch.bool, device=device))
    
    # Set the first 256 rows to all True
    mask[:256, :] = True
    
    # Repeat for batch size and flatten
    # Shape becomes (batch_size * seqlen * seqlen,)
    return mask.expand(batch_size, seqlen, seqlen).flatten()

attention_flashinfer_prefix_lm = FlashInferWrapper(custom_mask_fn=get_prefix_lm_mask)
