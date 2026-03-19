# Makefile – local benchmark runner (wraps scripts/run_benchmarks_local.sh)
#
# Usage:
#   make              — full pipeline: venv → deps → bench
#   make venv         — create .venv (no-op if already present)
#   make deps         — (re-)install all Python dependencies into .venv
#   make bench        — run benchmarks via benchmarks/Makefile
#   make clean-venv   — remove .venv
#
# Variables (override on command line or via environment):
#   VENV_PATH              — create venv at this path and symlink .venv to it
#                            (useful when $HOME has limited disk; e.g. VENV_PATH=$(mktemp -d))
#   TORCH_WHL_INDEX_URL    — PyTorch wheel index    (default: .../cu124)
#   FL_GPU_CLOCK_FREQ_MHZ  — target GPU clock freq   (default: 1290)
#   BENCH_ARGS             — extra args forwarded to benchmarks/Makefile

SHELL := /bin/bash
.DEFAULT_GOAL := all

ROOT_DIR    := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
VENV_DIR    := $(ROOT_DIR)/.venv
DEPS_DIR    := $(ROOT_DIR)/.deps
PY          := $(VENV_DIR)/bin/python

ATTN_GYM_DIR    := $(DEPS_DIR)/attention-gym
ATTN_GYM_COMMIT := 6a65742f

TORCH_WHL_INDEX_URL      ?= https://download.pytorch.org/whl/cu124
CUDA_TAG                 := $(notdir $(TORCH_WHL_INDEX_URL))
FLASHINFER_WHL_INDEX_URL := https://flashinfer.ai/whl/$(CUDA_TAG)/torch2.5/
FL_GPU_CLOCK_FREQ_MHZ    ?= 1290

BENCH_ARGS ?=

# ─── Phony targets ───────────────────────────────────────────────────────────

.PHONY: all venv deps bench clean-venv

all:
	@$(MAKE) --no-print-directory venv
	@$(MAKE) --no-print-directory deps
	@echo "Done installing dependencies"
	@echo "Now you can run the benchmarks"
	@echo "  $(MAKE) bench"

# ─── venv ─────────────────────────────────────────────────────────────────────

venv: $(PY)

$(PY):
	@command -v uv >/dev/null 2>&1 || { \
		echo ""; \
		echo "ERROR: uv is not installed."; \
		echo ""; \
		echo "Install (recommended):"; \
		echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		echo ""; \
		echo "Then re-run:"; \
		echo "  make venv"; \
		exit 1; \
	}
	@command -v git >/dev/null 2>&1 || { \
		echo "ERROR: git is required but not found on PATH."; exit 1; \
	}
	mkdir -p "$(DEPS_DIR)"
ifdef VENV_PATH
	@echo ""
	@echo "==> Creating virtualenv at $(VENV_PATH) (symlinked to .venv)"
	uv venv "$(VENV_PATH)"
	ln -sfn "$(VENV_PATH)" "$(VENV_DIR)"
else
	@echo ""
	@echo "==> Creating virtualenv at .venv"
	cd "$(ROOT_DIR)" && uv venv .venv
endif

# ─── deps ─────────────────────────────────────────────────────────────────────

deps: | $(PY)
	@echo ""
	@echo "==> Installing Python dependencies into .venv"
	@echo "PyTorch index-url: $(TORCH_WHL_INDEX_URL)"
	cd "$(ROOT_DIR)" && uv pip install --python "$(PY)" \
		"torch~=2.5.1" --index-url "$(TORCH_WHL_INDEX_URL)"
	@echo ""
	@echo "==> Installing attention-gym (pinned $(ATTN_GYM_COMMIT))"
	@if [ ! -d "$(ATTN_GYM_DIR)/.git" ]; then \
		rm -rf "$(ATTN_GYM_DIR)"; \
		git clone https://github.com/meta-pytorch/attention-gym.git "$(ATTN_GYM_DIR)"; \
	fi
	cd "$(ATTN_GYM_DIR)" && git fetch --all --tags >/dev/null 2>&1 || true
	cd "$(ATTN_GYM_DIR)" && git checkout "$(ATTN_GYM_COMMIT)"
	cd "$(ATTN_GYM_DIR)" && uv pip install --python "$(PY)" -e .
	@echo ""
	@echo "==> Installing FlashInfer 0.2.5"
	@echo "FlashInfer index-url: $(FLASHINFER_WHL_INDEX_URL)"
	cd "$(ROOT_DIR)" && uv pip install --python "$(PY)" \
		"flashinfer-python==0.2.5" --index-url "$(FLASHINFER_WHL_INDEX_URL)"
	@echo ""
	@echo "==> Installing vLLM 0.6.6"
	cd "$(ROOT_DIR)" && uv pip install --python "$(PY)" \
		"vllm>0.6.3,<0.7.0" "transformers<4.47"
	cd "$(ROOT_DIR)" && uv pip install --python "$(PY)" \
		"numpy<2.3.0" pandas seaborn matplotlib tabulate nvidia-ml-py
	cd "$(ROOT_DIR)" && uv pip install --python "$(PY)" -e .

# ─── bench ────────────────────────────────────────────────────────────────────

bench: | $(PY)
	@echo ""
	@echo "==> Running benchmarks"
	@echo "FL_GPU_CLOCK_FREQ_MHZ=$(FL_GPU_CLOCK_FREQ_MHZ)"
	export PYTHONPATH="$(ROOT_DIR)" && \
	export FL_GPU_CLOCK_FREQ_MHZ="$(FL_GPU_CLOCK_FREQ_MHZ)" && \
	export PATH="$(VENV_DIR)/bin:$$PATH" && \
	$(MAKE) -C benchmarks $(BENCH_ARGS)
	@echo ""
	@echo "==> Done"
	@echo "Outputs:"
	@echo "  - benchmarks/results/flex_variants.png"
	@echo "  - benchmarks/results/vllm_e2e.png"
	@echo "  - benchmarks/results/custom_variants.png"

# ─── clean ────────────────────────────────────────────────────────────────────

clean-venv:
	@if [ -L "$(VENV_DIR)" ]; then \
		rm -rf "$$(readlink -f "$(VENV_DIR)")"; \
	fi
	rm -rf "$(VENV_DIR)"
