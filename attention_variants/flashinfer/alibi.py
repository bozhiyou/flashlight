import torch
from .utils import FlashInferJITWrapper

alibi_fa2_decl = """
  #include <flashinfer/attention/variant_helper.cuh>
  #include <flashinfer/pos_enc.cuh>
  #include <flashinfer/math.cuh>

  using namespace flashinfer;

struct ALiBiAttention : AttentionVariantBase {
  static constexpr bool use_softmax = true;
  uint32_t num_qo_heads;
  float sm_scale;
  float sm_scale_log2;
  int32_t window_left;
  uint32_t kv_len;
  uint32_t qo_len;

  template <typename Params>
  __device__ __host__ ALiBiAttention(const Params& params, uint32_t batch_idx, uint8_t* smem_ptr) {
    num_qo_heads = params.num_qo_heads;
    sm_scale = params.sm_scale;
    sm_scale_log2 = math::log2e; // Fixed double scaling
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
      float slope = get_alibi_slope(qo_head_idx, num_qo_heads);
      return logits * sm_scale + slope * float(int(kv_idx) - int(qo_idx));
    })
  };
  """

alibi_fa3_decl = """
#include <flashinfer/attention/hopper/variant_helper.cuh>
#include <flashinfer/attention/hopper/attention_updater.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/math.cuh>

using namespace flashinfer;

struct ALiBiAttention : AttentionVariantBase {
  float sm_scale_log2;
  float sm_scale;
  uint32_t num_qo_heads;
  int32_t window_left;
  uint32_t kv_len;
  uint32_t qo_len;

  template <typename MainloopParams, typename BlockCoord>
  __device__ ALiBiAttention(const MainloopParams& params, const BlockCoord& block_coord) {
    sm_scale = params.additional_params.sm_scale;
    sm_scale_log2 = math::log2e; // Fixed double scaling
    num_qo_heads = params.additional_params.alibi_num_heads;
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
    float slope = get_alibi_slope(qo_head_idx, num_qo_heads);
    return logits * sm_scale + slope * float(int(kv_idx) - int(qo_idx));
  })
};
"""

attention_flashinfer_alibi = FlashInferJITWrapper(
    variant_name="ALiBiAttention",
    variant_decl_fa2=alibi_fa2_decl,
    variant_decl_fa3=alibi_fa3_decl,
    pos_encoding_mode=0,  # Use NONE to let JIT FA3 activate
    additional_scalar_names=["sm_scale", "alibi_num_heads"],
    additional_scalar_dtypes=["double", "int64_t"],
    extra_kwargs_fn=lambda q, k, v: {
        "sm_scale": 1.0 / (q.shape[-1] ** 0.5),
        "alibi_num_heads": q.shape[1]
    }
)
