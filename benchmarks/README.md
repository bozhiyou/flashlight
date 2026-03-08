# Benchmarks — Artifact Evaluation

Scripts to reproduce the FlashLight paper evaluation (Figures 2–4).

## Quick start

```bash
cd benchmarks/

make data       # collect all benchmark data (GPU required)
make figures    # generate figures from CSVs   (CPU only)
make all        # both of the above
```

Individual figures:

```bash
make fig2_fig3  # FlexAttention-supported variants  (Figures 2 & 3)
make fig4       # DiffAttn + Evoformer               (Figure 4)
```

## Claim-to-script mapping

| Paper figure | Data script | Plot script | Output |
|---|---|---|---|
| **Fig. 2 / 3** — FlexAttention-supported variants (H100 / A100) | `run_fig2_fig3_flex_variants.py` | `plot_fig2_fig3.py` | `results/fig2_fig3.png` |
| **Fig. 4 (left)** — Differential Attention | `run_fig4a_diff_attn.py` | `plot_fig4.py` | `results/fig4.png` |
| **Fig. 4 (right)** — Evoformer / IPA | `run_fig4b_evoformer.py` | `plot_fig4.py` | `results/fig4.png` |

`run_fig2_fig3_flex_variants.py` is run once per system (FlashLight, FlexAttention, torch.compile); see the Makefile targets or the flags below:

| System | Flag | Default output |
|---|---|---|
| FlashLight | `--flashlight` | `results/all.csv` |
| FlexAttention (cache hit) | `--flex` | `results/all_flex.csv` |
| FlexAttention (cache miss) | `--flex --no-mask-cache` | `results/all_flexnocache.csv` |
| torch.compile | `--torch.compile` | `results/all_torchcompile.csv` |

## Hardware requirements

- **GPU:** NVIDIA A100 80 GB or H100 80 GB (one GPU).
- **SM frequency:** for reproducible results, cap SM frequency to the GPU's steady-state clock. The paper uses 1290 MHz, the observed steady-state on the authors' A100 and H100 systems (Section 4.1). This value may differ on other instances:
  ```bash
  sudo nvidia-smi -lgc 1290,1290   # adjust if your steady-state differs
  ```
- **Software:** Python 3.12, PyTorch 2.5.0, Triton 3.1.0, CUDA 12.9.

## Expected runtime

Wall-clock estimates on a single GPU (default configs). **TODO: fill in measured times before submission.**

| Script | Approximate time |
|---|---|
| `run_fig2_fig3_flex_variants.py` (per system) | TBD |
| `run_fig4a_diff_attn.py` | TBD |
| `run_fig4b_evoformer.py` | TBD |
| Plotting scripts | < 1 min (CPU) |

## Output

- **CSVs** go to `results/` (gitignored).
- **Figures** are saved by the plot scripts to the working directory.
- Plot scripts read from `results/` by default; paths are configurable via CLI flags.

## Verification

For reproducible results, cap SM frequency following the protocol in Section 4.1 of the paper. With frequency capping, run-to-run variance on the same machine should be within 1 % standard deviation (20 runs, 10 warm-up).

Key qualitative expectations (from Sections 4.2–4.3):
- **Fig. 2/3:** FlashLight is competitive with or faster than FlexAttention for score_mod variants; for block_mask variants, FlexAttention kernel is faster but block-mask creation overhead makes FlashLight faster end-to-end.
- **Fig. 4:** FlashLight is always faster than torch.compile. Evoformer speedups are 5× or more (Section 4.3).

**TODO:** commit reference CSVs to `results/reference/` so reviewers can compare against known-good outputs.

## Dependencies

**Common (pip):**

- `nvidia-ml-py` — frequency-aware warmup
- `tabulate` — pretty-print
- `pandas`, `seaborn`, `matplotlib` — plotting

**For `run_fig2_fig3_flex_variants.py` only:** `attention-gym` (install from source):

```bash
git clone https://github.com/meta-pytorch/attention-gym.git
cd attention-gym && git checkout 6a65742f
pip install .
# → attn_gym-0.0.4.dev15+g6a65742f7.d20251022
```

## Development

- **Inductor cache:** set `TORCHINDUCTOR_CACHE_DIR=<path>` to inspect generated kernels (default: `/tmp/torchinductor_${USER}`).
- **Autotune:** search space is hardcoded in `monkeypatch/fusion/triton_heuristics.py`; configuration interface is TODO.
