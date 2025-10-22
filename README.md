# Dependencies
```
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