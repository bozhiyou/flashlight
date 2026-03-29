import torch
import random
from .utils import FlashInferJITWrapper

document_mask_fa2_decl = """
#include <flashinfer/attention/variant_helper.cuh>
#include <flashinfer/math.cuh>

using namespace flashinfer;

template <typename T>
__device__ __host__ auto get_doc_ids_offset(const T& params, uint32_t batch_idx, int) -> decltype(params.kv_indptr[batch_idx]) {
  return params.kv_indptr[batch_idx];
}

template <typename T>
__device__ __host__ int32_t get_doc_ids_offset(const T& params, uint32_t batch_idx, float) {
  return 0; // Fallback for PagedParams
}

struct DocumentMaskAttention : AttentionVariantBase {
  static constexpr bool use_softmax = true;
  int32_t* doc_ids;
  float sm_scale_log2;
  int32_t window_left;
  uint32_t kv_len;
  uint32_t qo_len;

  template <typename Params>
  __device__ __host__ DocumentMaskAttention(const Params& params, uint32_t batch_idx, uint8_t* smem_ptr) {
    sm_scale_log2 = params.sm_scale * math::log2e;
    window_left = -1; // Fallback
#ifndef __CUDA_ARCH__
    kv_len = 0;
    qo_len = 0;
    doc_ids = nullptr;
#else
    kv_len = params.get_kv_len(batch_idx);
    qo_len = params.get_qo_len(batch_idx);
    // doc_ids pointer offset by batch using SFINAE helper
    doc_ids = params.doc_ids + get_doc_ids_offset(params, batch_idx, 0);
#endif
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return logits;
  })

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return (doc_ids[qo_idx] == doc_ids[kv_idx]) && (kv_idx <= qo_idx);
  })
};
"""

document_mask_fa3_decl = """
#include <flashinfer/attention/hopper/variant_helper.cuh>
#include <flashinfer/attention/hopper/attention_updater.cuh>
#include <flashinfer/math.cuh>

using namespace flashinfer;

struct DocumentMaskAttention : AttentionVariantBase {
  float sm_scale_log2;
  int32_t* doc_ids;
  int32_t window_left;
  uint32_t kv_len;
  uint32_t qo_len;

  template <typename MainloopParams, typename BlockCoord>
  __device__ DocumentMaskAttention(const MainloopParams& params, const BlockCoord& block_coord) {
    sm_scale_log2 = params.additional_params.sm_scale * math::log2e;
    auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len_val, kv_len_val] = block_coord;
    doc_ids = params.additional_params.doc_ids + kv_indptr;
    window_left = params.window_left;
    kv_len = kv_len_val;
    qo_len = qo_len_val;
  }

  template <int NUM_ROWS_PER_THREAD>
  __device__ auto GetAttentionUpdater() {
    return OnlineSoftmax<NUM_ROWS_PER_THREAD, true>(sm_scale_log2);
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return logits;
  })

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return (doc_ids[qo_idx] == doc_ids[kv_idx]) && (kv_idx <= qo_idx);
  })
};
"""

def _get_document_ids(q, k, v):
    batch_size, num_qo_heads, seqlen, head_dim = q.shape
    device = q.device
    num_docs = 12
    rng = random.Random(0)
    
    lengths = [1] * num_docs
    remaining_length = seqlen - num_docs
    for _ in range(remaining_length):
        index = rng.randint(0, num_docs - 1)
        lengths[index] += 1
        
    doc_ids = torch.repeat_interleave(
        torch.arange(num_docs, device=device, dtype=torch.int32),
        torch.tensor(lengths, device=device, dtype=torch.int32)
    )
    
    doc_ids = doc_ids.unsqueeze(0).expand(batch_size, -1).flatten().contiguous()
    
    return {
        "doc_ids": doc_ids,
        "sm_scale": 1.0 / (head_dim ** 0.5)
    }

attention_flashinfer_document_mask = FlashInferJITWrapper(
    variant_name="DocumentMaskAttention",
    variant_decl_fa2=document_mask_fa2_decl,
    variant_decl_fa3=document_mask_fa3_decl,
    additional_tensor_names=["doc_ids"],
    additional_tensor_dtypes=["int32_t"],
    additional_scalar_names=["sm_scale"],
    additional_scalar_dtypes=["double"],
    extra_kwargs_fn=_get_document_ids,
    pos_encoding_mode=0
)
