#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

step() { echo ""; echo "==> $*"; }
die() { echo ""; echo "ERROR: $*" 1>&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || return 1
}

step "Flashlight Benchmark Runner"
echo "Repo: ${ROOT_DIR}"

if ! require_cmd uv; then
  die "uv is not installed.

Install (recommended):
  curl -LsSf https://astral.sh/uv/install.sh | sh

Then re-run:
  ${ROOT_DIR}/scripts/run_benchmarks_local.sh"
fi

if ! require_cmd git; then
  die "git is required but not found on PATH."
fi

if ! require_cmd make; then
  die "make is required but not found on PATH."
fi

VENV_DIR="${ROOT_DIR}/.venv"
DEPS_DIR="${ROOT_DIR}/.deps"
ATTN_GYM_DIR="${DEPS_DIR}/attention-gym"
ATTN_GYM_COMMIT="6a65742f"

TORCH_WHL_INDEX_URL="${TORCH_WHL_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
CUDA_TAG="${TORCH_WHL_INDEX_URL##*/}"                       # e.g. "cu121"
FLASHINFER_WHL_INDEX_URL="https://flashinfer.ai/whl/${CUDA_TAG}/torch2.5/"
FL_GPU_CLOCK_FREQ_MHZ="${FL_GPU_CLOCK_FREQ_MHZ:-1290}"

step "Creating/using virtualenv at .venv"
mkdir -p "${DEPS_DIR}"
if [[ ! -d "${VENV_DIR}" ]]; then
  (cd "${ROOT_DIR}" && uv venv .venv)
fi

PY="${VENV_DIR}/bin/python"
if [[ ! -x "${PY}" ]]; then
  die "Expected venv python at ${PY} but it was not found/executable."
fi

step "Installing Python dependencies into .venv"
echo "PyTorch index-url: ${TORCH_WHL_INDEX_URL}"
(cd "${ROOT_DIR}" && uv pip install --python "${PY}" \
  "torch==2.5.0" --index-url "${TORCH_WHL_INDEX_URL}")

step "Installing attention-gym (pinned commit ${ATTN_GYM_COMMIT})"
if [[ ! -d "${ATTN_GYM_DIR}/.git" ]]; then
  rm -rf "${ATTN_GYM_DIR}"
  git clone https://github.com/meta-pytorch/attention-gym.git "${ATTN_GYM_DIR}"
fi
(cd "${ATTN_GYM_DIR}" && git fetch --all --tags >/dev/null 2>&1 || true)
(cd "${ATTN_GYM_DIR}" && git checkout "${ATTN_GYM_COMMIT}")
(cd "${ATTN_GYM_DIR}" && uv pip install --python "${PY}" -e .)

step "Installing FlashInfer 0.2.5"
echo "FlashInfer index-url: ${FLASHINFER_WHL_INDEX_URL}"
(cd "${ROOT_DIR}" && uv pip install --python "${PY}" \
  "flashinfer-python==0.2.5" --index-url "${FLASHINFER_WHL_INDEX_URL}")

(cd "${ROOT_DIR}" && uv pip install --python "${PY}" \
  "numpy<2.3.0" pandas seaborn matplotlib tabulate nvidia-ml-py)

(cd "${ROOT_DIR}" && uv pip install --python "${PY}" -e .)

step "Running benchmarks (same defaults as Apptainer)"
export PYTHONPATH="${ROOT_DIR}"
export FL_GPU_CLOCK_FREQ_MHZ="${FL_GPU_CLOCK_FREQ_MHZ}"
export PATH="${VENV_DIR}/bin:${PATH}"
echo "Venv python: ${PY}"
echo "PATH=${PATH}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "FL_GPU_CLOCK_FREQ_MHZ=${FL_GPU_CLOCK_FREQ_MHZ}"

(cd "${ROOT_DIR}" && make -C benchmarks "$@")

step "Done"
echo "Outputs:"
echo "  - benchmarks/results/flex_variants.png"
echo "  - benchmarks/results/custom_variants.png"
