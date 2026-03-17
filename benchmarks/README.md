# Benchmarks

Performance benchmarks for FlashLight attention variants.

## Scripts

| Category | Data script | Plot script |
|---|---|---|
| FlexAttention-supported variants | `run_flex_variants.py` | `plot_flex_variants.py` |
| Differential Attention | `run_diff_attn.py` | `plot_custom_variants.py` |
| Evoformer / IPA | `run_evoformer.py` | `plot_custom_variants.py` |

`run_flex_variants.py` is run once per system (FlashLight, FlexAttention, torch.compile); see the flags below:

| System | Flag | Default output |
|---|---|---|
| FlashLight | `--flashlight` | `results/all.csv` |
| FlexAttention (cache hit) | `--flex` | `results/all_flex.csv` |
| FlexAttention (cache miss) | `--flex --no-mask-cache` | `results/all_flexnocache.csv` |
| torch.compile | `--torch.compile` | `results/all_torchcompile.csv` |

## Hardware requirements

- **GPU:** NVIDIA A100 or H100 (one GPU).
- **SM frequency:** for reproducible results, cap SM frequency to the GPU's steady-state clock:
  ```bash
  sudo nvidia-smi -lgc 1290,1290   # adjust if your steady-state differs
  ```
  **No sudo (e.g. TACC):** If you cannot run `nvidia-smi -lgc`, set `FL_GPU_CLOCK_FREQ_MHZ` so the benchmark warmup targets that frequency instead:
  ```bash
  FL_GPU_CLOCK_FREQ_MHZ=1290 python benchmarks/run_diff_attn.py
  # or export FL_GPU_CLOCK_FREQ_MHZ=1290
  ```
- **Software:** Python 3.12, PyTorch 2.5.0, Triton 3.1.0, CUDA 12.9.

## Expected runtime

Wall-clock times measured on a single A100-PCIE-40GB (default configs, no SM frequency cap):

| Script | Flag | Approximate time |
|---|---|---|
| `run_flex_variants.py` | `--flashlight` | ~35 min |
| `run_flex_variants.py` | `--flex` | ~7 min |
| `run_flex_variants.py` | `--flex --no-mask-cache` | ~5 min |
| `run_flex_variants.py` | `--torch.compile` | ~7 min |
| `run_diff_attn.py` | | ~14 min |
| `run_evoformer.py` | | ~9 min |
| Plotting scripts | | < 1 min (CPU) |

First runs include `torch.compile` compilation overhead; subsequent runs with a warm inductor cache are faster.

## Output

- **CSVs** go to `results/` (gitignored).
- **Figures** are saved by the plot scripts to the working directory.
- Plot scripts read from `results/` by default; paths are configurable via CLI flags.

## Verification

For reproducible results, cap SM frequency. With frequency capping, run-to-run variance on the same machine should be within 1 % standard deviation (20 runs, 10 warm-up).

### Reference results

Static reference CSVs live in `results/reference/` and are committed to the repo. These were collected with SM frequencies capped to ensure stable measurements:

| File | Description |
|------|-------------|
| `diff_attn_a100.csv` | DiffAttn benchmark on A100 |
| `diff_attn_h100.csv` | DiffAttn benchmark on H100 |
| `evo_attn_a100.csv` | Evoformer benchmark on A100 |
| `evo_attn_h100.csv` | Evoformer benchmark on H100 |

## Dependencies

**Common (pip):**

- `nvidia-ml-py` — frequency-aware warmup
- `tabulate` — pretty-print
- `pandas`, `seaborn`, `matplotlib` — plotting

**For `run_flex_variants.py` only:** `attention-gym` (install from source):

```bash
git clone https://github.com/meta-pytorch/attention-gym.git
cd attention-gym && git checkout 6a65742f
pip install .
# → attn_gym-0.0.4.dev15+g6a65742f7.d20251022
```

## Development

- **Inductor cache:** set `TORCHINDUCTOR_CACHE_DIR=<path>` to inspect generated kernels (default: `/tmp/torchinductor_${USER}`).
- **Autotune:** search space is hardcoded in `monkeypatch/fusion/triton_heuristics.py`; configuration interface is TODO.
