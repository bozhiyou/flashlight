from .utils import FlashInferWrapper

attention_flashinfer_softcapped = FlashInferWrapper(logits_soft_cap=30.0)
