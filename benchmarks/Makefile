# Makefile for reproducing FlashLight benchmark evaluation.
#
# Usage:
#   make all              — run all benchmarks and generate all figures
#   make data             — collect all benchmark data (GPU required)
#   make figures          — generate all figures from existing CSVs (CPU only)
#   make flex_variants    — end-to-end for FlexAttention-supported variants
#   make vllm_e2e         — end-to-end vLLM inference (requires vllm + flash_attn)
#   make custom_variants  — end-to-end for DiffAttn + Evoformer
#   make clean            — remove generated CSVs and figures (keeps results/reference/)
#
# Each data-collection target launches a separate GPU job.
# Run from the benchmarks/ directory or use `make -C benchmarks/`.

.DEFAULT_GOAL := all
PYTHON ?= python
# Target SM frequency (MHz) for warmup when sudo nvidia-smi -lgc is unavailable (e.g. TACC)
FL_GPU_CLOCK_FREQ_MHZ ?= 1290

.PHONY: all data flex_variants vllm_e2e vllm_e2e_synth traces custom_variants figures clean

all: data figures

# ─── FlexAttention-supported variants ────────────────────────────────────────

FLEX_SCRIPT   := run_flex_variants.py
FLEX_CSVS     := results/all.csv \
                 results/all_flex.csv \
                 results/all_flexnocache.csv \
                 results/all_flashinfer.csv
FLEX_PLOT     := plot_flex_variants.py

results/all.csv:
	FL_GPU_CLOCK_FREQ_MHZ=$(FL_GPU_CLOCK_FREQ_MHZ) $(PYTHON) $(FLEX_SCRIPT) --flashlight --output results/all

results/all_flex.csv:
	FL_GPU_CLOCK_FREQ_MHZ=$(FL_GPU_CLOCK_FREQ_MHZ) $(PYTHON) $(FLEX_SCRIPT) --flex --output results/all_flex

results/all_flexnocache.csv:
	FL_GPU_CLOCK_FREQ_MHZ=$(FL_GPU_CLOCK_FREQ_MHZ) $(PYTHON) $(FLEX_SCRIPT) --flex --no-mask-cache --output results/all_flexnocache

results/all_flashinfer.csv:
	FL_GPU_CLOCK_FREQ_MHZ=$(FL_GPU_CLOCK_FREQ_MHZ) $(PYTHON) $(FLEX_SCRIPT) --flashinfer --output results/all_flashinfer

results/all_torchcompile.csv:
	FL_GPU_CLOCK_FREQ_MHZ=$(FL_GPU_CLOCK_FREQ_MHZ) $(PYTHON) $(FLEX_SCRIPT) --torch.compile --output results/all_torchcompile

# ─── vLLM end-to-end inference ─────────────────────────────────────────────────

VLLM_SCRIPT   := vllm_e2e_infer.py
# The e2e runner writes two joined CSVs per run. Make tracks the summary
# file as the build product; the per_request file is written alongside it.
VLLM_CSV          := results/vllm_e2e_online_summary.csv
VLLM_PER_REQ_CSV  := results/vllm_e2e_online_per_request.csv
VLLM_PLOT     := plot_vllm_e2e.py

# The four e2e variants (2x2 matrix: mask shape x score_mod). All sit on the
# packed-doc substrate forced by vLLM's varlen layout.
VLLM_VARIANTS := causal_packed sliding_window_packed causal_alibi_packed causal_softcap_packed

vllm_e2e_synth:
	$(foreach v,$(VLLM_VARIANTS),$(PYTHON) $(VLLM_SCRIPT) --mode flashlight --variant $(v);)
	$(foreach v,$(VLLM_VARIANTS),$(PYTHON) $(VLLM_SCRIPT) --mode flex --variant $(v);)

# ─── Mooncake trace-driven benchmark ──────────────────────────────────────────

TRACE_DIR := /tmp/traces
TRACE_URL := https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces
TRACE_MAX_REQUESTS  ?= 200
# Cap input length below vLLM's 32K chunked-prefill auto-enable threshold and
# at/below the largest default bucket (32768). The e2e hook raises on Path B
# (chunked prefill / paged K/V) so --max-input-len must stay under this.
TRACE_MAX_INPUT_LEN ?= 16384
TRACE_ARGS := --max-input-len $(TRACE_MAX_INPUT_LEN) --max-requests $(TRACE_MAX_REQUESTS) --online

$(TRACE_DIR)/conversation_trace.jsonl:
	mkdir -p $(TRACE_DIR)
	curl -fL -o $@ $(TRACE_URL)/conversation_trace.jsonl

traces: $(TRACE_DIR)/conversation_trace.jsonl

# Trace-driven e2e data collection. Each invocation appends one row to
# results/vllm_e2e_online_summary.csv and N rows to
# results/vllm_e2e_online_per_request.csv (joined on the auto-generated
# run_id). Order: Flashlight first (modifies inductor, needs cold start),
# then FlexAttention, then baseline (unpatched flash-attn).
$(VLLM_CSV): $(TRACE_DIR)/conversation_trace.jsonl
	$(foreach v,$(VLLM_VARIANTS),$(PYTHON) $(VLLM_SCRIPT) --mode flashlight --variant $(v) --trace $(TRACE_DIR)/conversation_trace.jsonl $(TRACE_ARGS);)
	$(foreach v,$(VLLM_VARIANTS),$(PYTHON) $(VLLM_SCRIPT) --mode flex --variant $(v) --trace $(TRACE_DIR)/conversation_trace.jsonl $(TRACE_ARGS);)
	$(PYTHON) $(VLLM_SCRIPT) --mode baseline --trace $(TRACE_DIR)/conversation_trace.jsonl $(TRACE_ARGS)

# The per_request CSV is produced as a side-effect of the summary rule
# above (one invocation of vllm_e2e_infer.py writes both files). Declared
# here so downstream rules can list it as a prerequisite; the recipe just
# asserts the side-effect happened. Switch to `&:` grouped targets if GNU
# Make 4.3+ becomes the baseline.
$(VLLM_PER_REQ_CSV): $(VLLM_CSV)
	@test -f $@ || { echo "missing $@ (should be written alongside $<)"; exit 1; }

# ─── Custom variants (DiffAttn, Evoformer) ───────────────────────────────────

results/diff_attn.csv:
	FL_GPU_CLOCK_FREQ_MHZ=$(FL_GPU_CLOCK_FREQ_MHZ) $(PYTHON) run_diff_attn.py

results/evo_attn.csv:
	FL_GPU_CLOCK_FREQ_MHZ=$(FL_GPU_CLOCK_FREQ_MHZ) $(PYTHON) run_evoformer.py

CUSTOM_CSVS   := results/diff_attn.csv \
                 results/evo_attn.csv
CUSTOM_PLOT   := plot_custom_variants.py

data: $(FLEX_CSVS) $(CUSTOM_CSVS) $(VLLM_CSV)

# Plot scripts write to CWD with hardcoded names; move into results/.

# TODO: figures E1-E5 come from benchmarks/analyze_e2e.py, which reads the
# summary + per_request CSV pair. Until that script lands, this target only
# verifies the CSVs exist and leaves a placeholder PNG so the `figures` goal
# still resolves.
results/vllm_e2e_online.png: $(VLLM_CSV) $(VLLM_PER_REQ_CSV)
	@echo "[vllm_e2e] TODO: plot_vllm_e2e.py needs updating for the"
	@echo "          two-CSV schema ($(VLLM_CSV) + $(VLLM_PER_REQ_CSV))."
	@echo "          Writing placeholder $@ so 'make figures' succeeds."
	@touch $@
# Phase 2 replacement (uncomment when the script is ready):
# results/vllm_e2e_online.png: $(VLLM_CSV) $(VLLM_PER_REQ_CSV) $(VLLM_PLOT)
# 	$(PYTHON) $(VLLM_PLOT) --summary $(VLLM_CSV) --per-request $(VLLM_PER_REQ_CSV) --output $@

results/flex-able-bar.png: $(FLEX_CSVS) $(FLEX_PLOT)
	$(PYTHON) $(FLEX_PLOT) --plot bar
	mv flex-able-bar.png $@

results/flex-able-torch.png: $(FLEX_CSVS) results/all_torchcompile.csv $(FLEX_PLOT)
	$(PYTHON) $(FLEX_PLOT) --plot line
	mv flex-able-torch.png $@

results/flex-unable-bar.png: $(CUSTOM_CSVS) $(CUSTOM_PLOT)
	$(PYTHON) $(CUSTOM_PLOT) \
	  --diff-attn results/diff_attn.csv \
	  --evo-attn results/evo_attn.csv
	mv flex-unable-bar.png $@

flex_variants: results/flex-able-bar.png results/flex-able-torch.png

vllm_e2e: results/vllm_e2e_online.png

custom_variants: results/flex-unable-bar.png

figures: flex_variants custom_variants vllm_e2e
	@echo "Figures generated successfully:"
	@echo "	- results/flex-able-bar.png"
	@echo "	- results/flex-able-torch.png"
	@echo "	- results/flex-unable-bar.png"
	@echo "	- results/vllm_e2e_online.png"

# Generated outputs only; do not remove results/reference/
clean:
	rm -f $(FLEX_CSVS) $(CUSTOM_CSVS) \
	      $(VLLM_CSV) $(VLLM_PER_REQ_CSV) \
	      results/vllm_e2e_online.png results/vllm_e2e_online.csv \
	      results/flex-able-bar.png results/flex-able-torch.png \
	      results/vllm_e2e.png results/vllm_e2e_throughput.png \
	      results/flex-unable-bar.png
