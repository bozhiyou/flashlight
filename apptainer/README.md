# Apptainer image for FlashLight benchmarks

The image provides the environment (CUDA, PyTorch, dependencies). The codebase runs from your host repository via automatic bind mounts.

Before building or running the image, first load Apptainer (required on TACC):

```bash
module load tacc-apptainer/1.4.1
```

## Quick Start (via Makefile)

A `Makefile` in this directory automates build and run. It uses `$SCRATCH/flashlight.sif` when `$SCRATCH` is set (to avoid workspace disk quota). From the `apptainer/` directory:

```bash
make all          # Build container, run benchmarks, and point to outputs
```

Or run steps individually:

```bash
make image        # Build the container
make run          # Run full benchmark suite (requires GPU)
make interactive  # Interactive shell in the container (requires GPU)
```

## Build

From the **repository root**. Write the SIF to `$SCRATCH` to avoid workspace disk quota:

```bash
apptainer build --notest $SCRATCH/flashlight.sif apptainer/flashlight.def
```

To also run the built-in smoke test during build (needs no GPU):

```bash
apptainer build $SCRATCH/flashlight.sif apptainer/flashlight.def
```

### (Optional) Verify GPU visibility

```bash
apptainer exec --nv --unsquash $SCRATCH/flashlight.sif nvidia-smi
```

## Run

The benchmarks require GPUs; use `--nv` for host NVIDIA driver access. On TACC, squashfuse may time out, so use `--unsquash` to extract the SIF to a temp sandbox:

```bash
apptainer run --nv --unsquash $SCRATCH/flashlight.sif
```

Interactive shell:

```bash
apptainer shell --nv --unsquash $SCRATCH/flashlight.sif
```

### Simulated frequency capping

The runscript sets `FL_GPU_CLOCK_FREQ_MHZ=1290` by default so benchmarks warm up to that SM frequency when `sudo nvidia-smi -lgc` is unavailable (e.g. on TACC). To override:

```bash
apptainer run --nv --unsquash --env FL_GPU_CLOCK_FREQ_MHZ=1350 $SCRATCH/flashlight.sif
```

## Contents

- **flashlight.def** — Apptainer definition (base: PyTorch 2.5.1 with CUDA 12.4, FlashInfer 0.2.5; default run is `run_benchmarks.sh`).
- **run_benchmarks.sh** — Entrypoint script; runs `make -C benchmarks all` inside the container.
- **Makefile** — Host-side build and run targets.
