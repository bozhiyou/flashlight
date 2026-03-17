from .utils import FlashInferWrapper

attention_flashinfer_sliding_window = FlashInferWrapper(causal=True, window_left=256)
