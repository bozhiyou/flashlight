import torch
from .utils import FlashInferJITWrapper

prefix_lm_fa2_decl = """
#include <flashinfer/attention/variant_helper.cuh>
#include <flashinfer/math.cuh>

using namespace flashinfer;

struct PrefixLMAttention : AttentionVariantBase {
  static constexpr bool use_softmax = true;
  int64_t prefix_length;
  float sm_scale_log2;
  int32_t window_left;
  uint32_t kv_len;
  uint32_t qo_len;

  template <typename Params>
  __device__ __host__ PrefixLMAttention(const Params& params, uint32_t batch_idx, uint8_t* smem_ptr) {
    prefix_length = params.prefix_length;
    sm_scale_log2 = params.sm_scale * math::log2e;
    window_left = -1; // Fallback
#ifndef __CUDA_ARCH__
    kv_len = 0;
    qo_len = 0;
#else
    kv_len = params.get_kv_len(batch_idx);
    qo_len = params.get_qo_len(batch_idx);
#endif
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return logits;
  })

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return (qo_idx < prefix_length) || (kv_idx <= qo_idx);
  })
};
"""

prefix_lm_fa3_decl = """
#include <flashinfer/attention/hopper/variant_helper.cuh>
#include <flashinfer/attention/hopper/attention_updater.cuh>
#include <flashinfer/math.cuh>

using namespace flashinfer;

struct PrefixLMAttention : AttentionVariantBase {
  float sm_scale_log2;
  int64_t prefix_length;
  int32_t window_left;
  uint32_t kv_len;
  uint32_t qo_len;

  template <typename MainloopParams, typename BlockCoord>
  __device__ PrefixLMAttention(const MainloopParams& params, const BlockCoord& block_coord) {
    sm_scale_log2 = params.additional_params.sm_scale * math::log2e;
    prefix_length = params.additional_params.prefix_length;
    window_left = params.window_left;
    auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len_val, kv_len_val] = block_coord;
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
    return (qo_idx < prefix_length) || (kv_idx <= qo_idx);
  })
};
"""

attention_flashinfer_prefix_lm = FlashInferJITWrapper(
    variant_name="PrefixLMAttention",
    variant_decl_fa2=prefix_lm_fa2_decl,
    variant_decl_fa3=prefix_lm_fa3_decl,
    additional_scalar_names=["prefix_length", "sm_scale"],
    additional_scalar_dtypes=["int64_t", "double"],
    extra_kwargs_fn=lambda q, k, v: {
        "prefix_length": 256,
        "sm_scale": 1.0 / (q.shape[-1] ** 0.5)
    },
    pos_encoding_mode=0
)
