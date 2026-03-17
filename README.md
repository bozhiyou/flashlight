# Dependencies
```
python>3.10     # monkey-patching `contextlib.contextmanager` requires `co_qualname` field of `PyCodeObject` introduced in Python 3.11 (bpo-44530)
torch==2.5.0    # which requires 'python<3.13'
numpy           # optional; torch raises `UserWarning: Failed to initialize NumPy: No module named 'numpy'`
```

## Install PyTorch 2.5.0
Supports CUDA 11.8/12.1/12.4
```
pip install 'torch==2.5.0' --index-url https://download.pytorch.org/whl/cu121  # or cu118, cu124
```
Need to build from source for other cuda versions.

# Installation
```
pip install -e .
```

## Running scripts without installing

If you prefer not to install the package, run all scripts from the repository
root with `PYTHONPATH=.` so that `monkeypatch`, `attention_variants`, and
`tests` are importable:

```bash
PYTHONPATH=. python tests/test_causal.py
PYTHONPATH=. python benchmarks/run_diff_attn.py
```

# Apptainer (Container on TACC)

Follow [apptainer/README.md](apptainer/README.md).