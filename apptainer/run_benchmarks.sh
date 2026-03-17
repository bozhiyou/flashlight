#!/usr/bin/env bash
# Benchmark entrypoint.
# NOTE: This script is intended to be executed INSIDE the Apptainer container.
# Do not run this directly on the host.

unset LD_PRELOAD
export PATH="/opt/conda/bin:$PATH"
export PYTHONHOME="/opt/conda"
export PYTHONPATH="$PWD"

# The container is run from the repo root, so $PWD is the repo root.
make -C benchmarks all "$@"
