import torch

# Initialize flashinfer lazily so it doesn't fail on import if not installed
_flashinfer_module = None

def get_flashinfer():
    global _flashinfer_module
    if _flashinfer_module is None:
        import flashinfer
        _flashinfer_module = flashinfer
    return _flashinfer_module

class FlashInferWrapper:
    """Stateful wrapper for FlashInfer's BatchPrefillWithRaggedKVCacheWrapper to avoid 
    plan overhead in benchmark loops."""
    
    def __init__(self, causal=False, pos_encoding_mode="NONE", window_left=-1, logits_soft_cap=None, custom_mask_fn=None):
        self.wrapper = None
        self.causal = causal
        self.pos_encoding_mode = pos_encoding_mode
        self.window_left = window_left
        self.logits_soft_cap = logits_soft_cap
        self.custom_mask_fn = custom_mask_fn
        # Workspace size (128MB) for FlashInfer
        self.workspace_size = 128 * 1024 * 1024
        
        # Track configurations so we can re-plan if shapes change
        self.last_config = None

    def __call__(self, q, k, v, enable_gqa=False, **kwargs):
        batch_size, num_qo_heads, seqlen, head_dim = q.shape
        num_kv_heads = k.size(1)
        
        flashinfer = get_flashinfer()

        config = (batch_size, num_qo_heads, num_kv_heads, seqlen, head_dim, q.dtype)
        if self.wrapper is None or self.last_config != config:
            self.wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                torch.empty(self.workspace_size, dtype=torch.uint8, device=q.device), "NHD"
            )
            qo_indptr = torch.arange(0, (batch_size + 1) * seqlen, seqlen, dtype=torch.int32, device=q.device)
            kv_indptr = torch.arange(0, (batch_size + 1) * seqlen, seqlen, dtype=torch.int32, device=q.device)

            custom_mask = self.custom_mask_fn(batch_size, seqlen, device=q.device) if self.custom_mask_fn else None

            self.wrapper.plan(
                qo_indptr,
                kv_indptr,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                causal=self.causal,
                pos_encoding_mode=self.pos_encoding_mode,
                window_left=self.window_left,
                logits_soft_cap=self.logits_soft_cap,
                custom_mask=custom_mask,
                q_data_type=q.dtype,
            )
            self.last_config = config

        # Transpose to NHD and flatten sequence across batches for ragged prefill
        q_flat = q.transpose(1, 2).reshape(-1, num_qo_heads, head_dim)
        k_flat = k.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)
        v_flat = v.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)

        out = self.wrapper.run(q_flat, k_flat, v_flat)
        return out.view(batch_size, seqlen, num_qo_heads, head_dim).transpose(1, 2)
