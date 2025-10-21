- FlexAttention examples benchmark: `variants.py`

NOTE: comment `apply_patch()` for FlexAttention benchmark

- Diff-Attn benchmark: `diff_attn.py`
- Evoformer/IPA benchmark: `evo_attn.py`

---
To observe generated kernel, set `TORCHINDUCTOR_CACHE_DIR=<path>` and find the kernel in `<path>` (default to `/tmp/torchinductor_${USER}`).

Autotune space is hardcoded in `monkeypatch/fusion/triton_heuristics.py`. Configuration interface is TODO.