# Flashlight: PyTorch Compiler Extensions to Accelerate Attention Variants

Flashlight is a compiler-native framework within the PyTorch ecosystem that automatically generates fused, FlashAttention-style kernels for a wide range of attention programs, including data-dependent variants.

## Quick Start

We provide two automated workflows to run the full benchmark suite on a single **NVIDIA A100 or H100 GPU**. Both generate benchmark results and figures in `benchmarks/results/`.

### Option A: Local run (via `uv`)

This one-command runner creates an isolated `uv` virtual environment, installs all dependencies (including PyTorch 2.5.0 with CUDA 12.1 wheels and a pinned version of `attention-gym`), and runs the end-to-end evaluation:

```bash
./scripts/run_benchmarks_local.sh
```

Optional overrides:

```bash
# Torch wheel index (defaults to CUDA 12.1 wheels)
TORCH_WHL_INDEX_URL=https://download.pytorch.org/whl/cu121 ./scripts/run_benchmarks_local.sh

# Simulated frequency capping target (defaults to 1290 MHz)
FL_GPU_CLOCK_FREQ_MHZ=1290 ./scripts/run_benchmarks_local.sh
```

### Option B: Apptainer (for shared clusters like TACC)

If you are on a cluster where Apptainer is preferred:

```bash
cd apptainer
module load tacc-apptainer/1.4.1
make all
```

See [apptainer/README.md](apptainer/README.md) for more details.

### Notes on Hardware and Reproducibility

- **Hardware:** The benchmarks are designed for modern datacenter GPUs (A100 80GB or H100 80GB). Runtimes on other GPUs will differ.
- **Frequency Capping:** To accurately reproduce speedups, locking the GPU SM clock frequency is strongly recommended. The automated scripts use a fallback heuristic (`FL_GPU_CLOCK_FREQ_MHZ=1290`) to "warm up" the GPU to the target frequency when `sudo` is unavailable (e.g., on TACC). While this improves consistency, it is best-effort and may still introduce minor discrepancies.
- **Compilation Overhead:** The first run incurs substantial JIT compilation time (up to several minutes) as Triton kernels are generated and autotuned. Subsequent runs are fast. Generated kernels can be inspected in the TorchInductor cache directory (typically `/tmp/torchinductor_${USER}`).

---

## Manual Installation and Usage

If you wish to explore the code outside the automated workflows, you can install the package and its dependencies manually.

### Dependencies

```text
python>3.10,<3.13   # monkey-patching `contextlib.contextmanager` requires `co_qualname` field of `PyCodeObject` introduced in Python 3.11 (bpo-44530)
torch==2.5.0        # which requires 'python<3.13'
numpy               # optional; torch raises `UserWarning: Failed to initialize NumPy: No module named 'numpy'`
```

To install PyTorch 2.5.0 (e.g., for CUDA 12.1):

```bash
pip install 'torch==2.5.0' --index-url https://download.pytorch.org/whl/cu121  # or cu118, cu124
```

*Note: For other CUDA versions, you may need to build PyTorch from source.*

### Installation

```bash
pip install -e .
```

### Running scripts without installing

If you prefer not to install the package, you can run all scripts from the repository root by setting `PYTHONPATH=.` so that the `monkeypatch`, `attention_variants`, and `tests` modules are importable:

```bash
PYTHONPATH=. python tests/test_causal.py
PYTHONPATH=. python benchmarks/run_diff_attn.py
```
