```
Shape parameters:
batch_size=32, nheads=32, seqlen=512, headdim=64
```

files with `flex` in their names are FlexAttention, otherwise our implementation.

`output_` contains the wrapper code (which is runnable).

`kernel_` contains the kernel code, which is embeded as a string in the corresponding wrapper code.
