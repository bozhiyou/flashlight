# Benchmarks

Performance benchmarks for FlashLight attention variants.

## Kernel Microbenchmarks

Measure raw attention kernel performance (forward pass) for various attention variants.

| Category | Data script | Plot script |
|---|---|---|
| FlexAttention-supported variants | `run_flex_variants.py` | `plot_flex_variants.py` |
| Differential Attention | `run_diff_attn.py` | `plot_custom_variants.py` |
| Evoformer / IPA | `run_evoformer.py` | `plot_custom_variants.py` |

### `run_flex_variants.py`

`run_flex_variants.py` is run once per system (FlashLight, FlexAttention, torch.compile); see the flags below:

| System | Flag | Default output |
|---|---|---|
| FlashLight | `--flashlight` | `results/all.csv` |
| FlexAttention (cache hit) | `--flex` | `results/all_flex.csv` |
| FlexAttention (cache miss) | `--flex --no-mask-cache` | `results/all_flexnocache.csv` |
| FlashInfer | `--flashinfer` | `results/all_flashinfer.csv` |
| torch.compile | `--torch.compile` | `results/all_torchcompile.csv` |

## End-to-End Inference Benchmarks

Measure full-stack performance in vLLM, capturing Time-to-First-Token (TTFT) and Inter-Token Latency (ITL).

### `vllm_e2e_infer.py`

`vllm_e2e_infer.py` runs a real vLLM generation pipeline with custom attention backends (plotted with `plot_vllm_e2e.py`). It supports three workload types:

### Modes

| Mode | Flag | Description |
|---|---|---|
| Baseline | `--mode baseline` | Unpatched vLLM (control) |
| FlexAttention | `--mode flex --variant <v>` | Hooks prefill with a FlexAttention variant |
| FlexAttention (no cache) | `--mode flex --variant <v> --no-mask-cache` | Same, with block-mask caching disabled |
| Flashlight | `--mode flashlight --variant <v>` | Hooks prefill with a Flashlight-compiled variant |
| Debug | `--mode debug` | Verify hooks reach the attention layer |

Available variants: `causal`, `sliding_window`, `prefix_lm`, `document_mask`, `full`, `alibi`, `softcap`.

### Workload Types

| Workload | Trigger | Description |
|---|---|---|
| **Synthetic** | (default w/o `--trace`) | Sweeps batch sizes (1–32) and input lengths (512–16K). |
| **Trace (Online)** | `--trace <path>` | Streaming inference respecting trace timestamps. |
| **Trace (Offline)** | `--trace <path> --offline` | Bulk inference using a Mooncake JSONL trace. |

#### Trace-driven flags

- `--trace <path>`: Path to Mooncake JSONL trace.
- `--max-requests <n>`: Limit number of requests from the trace.
- `--max-input-len <n>`: Filter requests by input length.
- `--online` / `--offline`: Toggle streaming vs. bulk mode.
- `--enable-prefix-caching`: Enable vLLM's prefix caching (useful for multi-turn traces).

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
- **Software:** Python 3.12, PyTorch 2.5.1, Triton 3.1.0, CUDA 12.9.

## Expected runtime

End-to-end wall-clock time for the default workflow (`make all`) is approximately **15 minutes** on an A100-class GPU. Runtimes on other recent NVIDIA GPUs may differ modestly. Note that the first run includes compilation overhead for `torch.compile` and Flashlight; subsequent runs with a warm TorchInductor cache are typically faster.

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

**For `vllm_e2e_infer.py` only:** `vllm`

**For `run_flex_variants.py` and `vllm_e2e_infer.py`:** `attention-gym` (install from source):

```bash
git clone https://github.com/meta-pytorch/attention-gym.git
cd attention-gym && git checkout 6a65742f
pip install .
# → attn_gym-0.0.4.dev15+g6a65742f7.d20251022
```

## Development

- **Inductor cache:** set `TORCHINDUCTOR_CACHE_DIR=<path>` to inspect generated kernels (default: `/tmp/torchinductor_${USER}`).
- **Autotune:** search space is hardcoded in `monkeypatch/fusion/triton_heuristics.py`; configuration interface is TODO.
