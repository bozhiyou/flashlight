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


class FlashInferJITWrapper:
    """Stateful wrapper for FlashInfer's gen_customize_batch_prefill_module.
    JIT compiles the custom C++ variant and caches the compiled module."""
    
    def __init__(self, variant_name, variant_decl_fa2, variant_decl_fa3=None,
                 additional_tensor_names=(), additional_tensor_dtypes=(),
                 additional_scalar_names=(), additional_scalar_dtypes=(),
                 pos_encoding_mode=0, use_sliding_window=False,
                 use_logits_soft_cap=False, causal=False,
                 extra_kwargs_fn=None):
        self.variant_name = variant_name
        self.variant_decl_fa2 = variant_decl_fa2
        self.variant_decl_fa3 = variant_decl_fa3
        self.additional_tensor_names = additional_tensor_names
        self.additional_tensor_dtypes = additional_tensor_dtypes
        self.additional_scalar_names = additional_scalar_names
        self.additional_scalar_dtypes = additional_scalar_dtypes
        self.pos_encoding_mode = pos_encoding_mode
        self.use_sliding_window = use_sliding_window
        self.use_logits_soft_cap = use_logits_soft_cap
        self.causal = causal
        self.extra_kwargs_fn = extra_kwargs_fn

        self.workspace_size = 128 * 1024 * 1024
        
        # Cache for compiled modules keyed by (head_dim, dtype, backend)
        self._modules = {}
        # Track configuration for replanning
        self._last_config = None
        self._current_wrapper = None
        self._workspace = None

    def __call__(self, q, k, v, enable_gqa=False, **kwargs):
        batch_size, num_qo_heads, seqlen, head_dim = q.shape
        num_kv_heads = k.size(1)

        flashinfer = get_flashinfer()
        
        dtype_str = "f16" if q.dtype == torch.float16 else "bf16" if q.dtype == torch.bfloat16 else "f32"
        idtype = "i32"

        # Determine backend
        backend = flashinfer.utils.determine_attention_backend(
            device=q.device,
            pos_encoding_mode=self.pos_encoding_mode,
            use_fp16_qk_reductions=False,
            use_custom_mask=False,
            dtype_q=q.dtype,
            dtype_kv=k.dtype
        )

        if backend == "fa3" and self.variant_decl_fa3 is None:
            backend = "fa2"

        variant_decl = self.variant_decl_fa3 if backend == "fa3" else self.variant_decl_fa2
        uri = f"{self.variant_name}_{backend}_{head_dim}_{dtype_str}"

        config = (batch_size, num_qo_heads, num_kv_heads, seqlen, head_dim, q.dtype, uri)
        
        extra_kwargs = {}
        if self.extra_kwargs_fn is not None:
            extra_kwargs = self.extra_kwargs_fn(q, k, v)

        # Build positional args for JIT modules
        extra_args = []
        for name in self.additional_tensor_names:
            extra_args.append(extra_kwargs[name])
        for name in self.additional_scalar_names:
            extra_args.append(extra_kwargs[name])

        if self._current_wrapper is None or self._last_config != config:
            if self._workspace is None or self._workspace.device != q.device:
                self._workspace = torch.empty(self.workspace_size, dtype=torch.uint8, device=q.device)
                
            jit_args = [
                uri,
                q.dtype,
                k.dtype,
                q.dtype,
                torch.int32,
                head_dim,
                head_dim,
                list(self.additional_tensor_names),
                list(self.additional_tensor_dtypes),
                list(self.additional_scalar_names),
                list(self.additional_scalar_dtypes),
                self.variant_name,
                variant_decl,
                self.pos_encoding_mode,
                self.use_sliding_window,
                self.use_logits_soft_cap,
            ]
            self._current_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                self._workspace, "NHD", backend=backend, jit_args=jit_args
            )
            
            qo_indptr = torch.arange(0, (batch_size + 1) * seqlen, seqlen, dtype=torch.int32, device=q.device)
            kv_indptr = torch.arange(0, (batch_size + 1) * seqlen, seqlen, dtype=torch.int32, device=q.device)

            self._current_wrapper.plan(
                qo_indptr,
                kv_indptr,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                causal=self.causal,
                q_data_type=q.dtype
            )
            self._last_config = config

        # Transpose to NHD and flatten sequence across batches for ragged prefill
        q_flat = q.transpose(1, 2).reshape(-1, num_qo_heads, head_dim)
        k_flat = k.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)
        v_flat = v.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)

        out = self._current_wrapper.run(q_flat, k_flat, v_flat, *extra_args)
        return out.view(batch_size, seqlen, num_qo_heads, head_dim).transpose(1, 2)
