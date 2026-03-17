from .vanilla import attention_flashinfer
from .alibi import attention_flashinfer_alibi
from .softcap import attention_flashinfer_softcapped
from .causal import attention_flashinfer_causal
from .sliding_window import attention_flashinfer_sliding_window
from .prefix_lm import attention_flashinfer_prefix_lm
from .document_mask import attention_flashinfer_document_mask

FLASHINFER_ATTENTION_REGISTRY = {
    "flashinfer_full": attention_flashinfer,
    "flashinfer_full_with_alibi": attention_flashinfer_alibi,
    "flashinfer_full_with_softcap": attention_flashinfer_softcapped,
    "flashinfer_full_with_causal": attention_flashinfer_causal,
    "flashinfer_full_with_sliding_window": attention_flashinfer_sliding_window,
    "flashinfer_full_with_prefix_lm": attention_flashinfer_prefix_lm,
    "flashinfer_full_with_document_mask": attention_flashinfer_document_mask,
}
