#!/usr/bin/env bash
# MLSys'26 Artifact Evaluation entrypoint.
# NOTE: This script is intended to be executed INSIDE the Apptainer container.
# Do not run this directly on the host.

unset LD_PRELOAD
export PATH="/opt/conda/bin:$PATH"
export PYTHONHOME="/opt/conda"
export PYTHONPATH="$PWD"
# Simulate SM frequency capping when sudo nvidia-smi -lgc is unavailable (default 1290 MHz)
export FL_GPU_CLOCK_FREQ_MHZ="${FL_GPU_CLOCK_FREQ_MHZ:-1290}"

# The container is run from the repo root, so $PWD is the repo root.
make -C benchmarks all "$@"
