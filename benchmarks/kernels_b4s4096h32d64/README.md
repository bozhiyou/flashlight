Config
- Flex variants and DiffAttn: batch_size=4, seqlen=4096, nheads=32, headdim=64
- EvoAttn (IPA): batch_size=32, seqlen=256, nheads=4, headdim=64

For each variant:

- `flex.py`: Runnable wrapper code for FlexAttention. The kernel is embeded as a string and the configuration is in `meta0 = {...}`. `BLOCK_M` and `BLOCK_N` are tile sizes along sequence length dimension for Q and K/V, respectively.
- `xxx.best_config`: Best configuration used for Flashlight. `XBLOCK0` is the tile size in Q sequence length dimension, `RBLOCK` is the tile size in K/V sequence length dimension.
- `xxx.py`: Flashlight Triton kernel. This kernel is also embeded as a string in the wrapper.
- `yyy.py`: Runnable wrapper code for Flashlight.