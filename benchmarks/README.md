- FlexAttention examples benchmark: `variants.py`

NOTE: comment `apply_patch()` for FlexAttention benchmark

- Diff-Attn benchmark: `diff_attn.py`
- Evoformer/IPA benchmark: `evo_attn.py`

---
To observe generated kernel, set `TORCHINDUCTOR_CACHE_DIR=<path>` and find the kernel in `<path>` (default to `/tmp/torchinductor_${USER}`).

Autotune space is hardcoded in `monkeypatch/fusion/triton_heuristics.py`. Configuration interface is TODO.


# Benchmark Dependencies
```
# common dependency
nvidia-ml-py    # for frequency-aware warmup; previously `pynvml`
tabulate        # for pretty print

# variants.py
attention-gym   # need to install from source
```

## Install `attention-gym` (6a65742f)
```
git clone https://github.com/meta-pytorch/attention-gym.git
```
```
cd attention-gym && git checkout 6a65742f
```
```
pip install .
# ...
# Successfully installed attn_gym-0.0.4.dev15+g6a65742f7.d20251022
```