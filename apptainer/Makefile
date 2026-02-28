# Makefile for host-side execution to build and run the Apptainer image.
# Usage: Run from the apptainer/ directory.
# Uses $SCRATCH/flashlight.sif when SCRATCH is set (e.g. on TACC) to avoid disk quotas.

SHELL := /bin/bash

IMAGE_PATH ?= $(if $(SCRATCH),$(SCRATCH)/flashlight.sif,flashlight.sif)
# Path to pass to apptainer from repo root: use IMAGE_PATH if it contains a directory, else apptainer/IMAGE
SIF_PATH := $(if $(findstring /,$(IMAGE_PATH)),$(IMAGE_PATH),apptainer/$(IMAGE_PATH))

.DEFAULT_GOAL := all
.PHONY: all image run interactive clean check-apptainer

all:
	@echo "Starting full AE reproduction..."
	@$(MAKE) check-apptainer
	@$(MAKE) image
	@$(MAKE) run
	@echo ""
	@echo "============================================================"
	@echo "Reproduction complete!"
	@echo "Outputs are located at:"
	@echo "  CSVs:    ../benchmarks/results/*.csv"
	@echo "  Figures: ../benchmarks/results/*.png"
	@echo "============================================================"

check-apptainer:
	@command -v apptainer >/dev/null 2>&1 || { \
		echo ""; \
		echo "ERROR: 'apptainer' not found in PATH."; \
		echo ""; \
		echo "On TACC, load the module in your shell, then re-run:"; \
		echo "  module load tacc-apptainer/1.4.1"; \
		echo "  make -C apptainer"; \
		echo ""; \
		echo "If your site uses a different module name/version, load that instead."; \
		exit 127; \
	}

image:
	@$(MAKE) check-apptainer
	@echo "Building Apptainer image at $(IMAGE_PATH)..."
	apptainer build --notest $(IMAGE_PATH) flashlight.def

run:
	@$(MAKE) check-apptainer
	@echo "Running full Artifact Evaluation..."
	cd .. && apptainer run --nv --unsquash $(SIF_PATH)

interactive:
	@$(MAKE) check-apptainer
	cd .. && apptainer shell --nv --unsquash $(SIF_PATH)

clean:
	rm -f $(IMAGE_PATH)