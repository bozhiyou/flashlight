# Makefile for reproducing FlashLight benchmark evaluation.
#
# Usage:
#   make all              — run all benchmarks and generate all figures
#   make data             — collect all benchmark data (GPU required)
#   make figures          — generate all figures from existing CSVs (CPU only)
#   make flex_variants    — end-to-end for FlexAttention-supported variants
#   make custom_variants  — end-to-end for DiffAttn + Evoformer
#   make clean            — remove generated CSVs and figures (keeps results/reference/)
#
# Each data-collection target launches a separate GPU job.
# Run from the benchmarks/ directory or use `make -C benchmarks/`.

.DEFAULT_GOAL := all
PYTHON ?= python
# Target SM frequency (MHz) for warmup when sudo nvidia-smi -lgc is unavailable (e.g. TACC)
FL_GPU_CLOCK_FREQ_MHZ ?= 1290

all: data figures

# ─── FlexAttention-supported variants ────────────────────────────────────────

FLEX_SCRIPT   := run_flex_variants.py
FLEX_CSVS     := results/all.csv \
                 results/all_flex.csv \
                 results/all_flexnocache.csv \
                 results/all_flashinfer.csv
                 # results/all_torchcompile.csv
FLEX_PLOT     := plot_flex_variants.py

results/all.csv:
	FL_GPU_CLOCK_FREQ_MHZ=$(FL_GPU_CLOCK_FREQ_MHZ) $(PYTHON) $(FLEX_SCRIPT) --flashlight --output results/all

results/all_flex.csv:
	FL_GPU_CLOCK_FREQ_MHZ=$(FL_GPU_CLOCK_FREQ_MHZ) $(PYTHON) $(FLEX_SCRIPT) --flex --output results/all_flex

results/all_flexnocache.csv:
	FL_GPU_CLOCK_FREQ_MHZ=$(FL_GPU_CLOCK_FREQ_MHZ) $(PYTHON) $(FLEX_SCRIPT) --flex --no-mask-cache --output results/all_flexnocache

results/all_flashinfer.csv:
	FL_GPU_CLOCK_FREQ_MHZ=$(FL_GPU_CLOCK_FREQ_MHZ) $(PYTHON) $(FLEX_SCRIPT) --flashinfer --output results/all_flashinfer

# results/all_torchcompile.csv:
# 	FL_GPU_CLOCK_FREQ_MHZ=$(FL_GPU_CLOCK_FREQ_MHZ) $(PYTHON) $(FLEX_SCRIPT) --torch.compile --output results/all_torchcompile

# ─── Custom variants (DiffAttn, Evoformer) ───────────────────────────────────

CUSTOM_CSVS   := results/diff_attn.csv \
                 results/evo_attn.csv
CUSTOM_PLOT   := plot_custom_variants.py

results/diff_attn.csv:
	FL_GPU_CLOCK_FREQ_MHZ=$(FL_GPU_CLOCK_FREQ_MHZ) $(PYTHON) run_diff_attn.py

results/evo_attn.csv:
	FL_GPU_CLOCK_FREQ_MHZ=$(FL_GPU_CLOCK_FREQ_MHZ) $(PYTHON) run_evoformer.py

# ─── Figure generation (CPU only) ────────────────────────────────────────────
# Plot scripts write to CWD with hardcoded names; move into results/.

results/flex_variants.png: $(FLEX_CSVS) $(FLEX_PLOT)
	$(PYTHON) $(FLEX_PLOT) --plot bar
	mv flex-able-bar.png $@

results/custom_variants.png: $(CUSTOM_CSVS) $(CUSTOM_PLOT)
	$(PYTHON) $(CUSTOM_PLOT) \
	  --diff-attn results/diff_attn.csv \
	  --evo-attn results/evo_attn.csv
	mv flex-unable-bar.png $@

# ─── Aggregate targets ───────────────────────────────────────────────────────

.PHONY: all data flex_variants custom_variants figures clean

data: $(FLEX_CSVS) $(CUSTOM_CSVS)

flex_variants: results/flex_variants.png

custom_variants: results/custom_variants.png

figures: flex_variants custom_variants
	@echo "Figures generated successfully:"
	@echo "	- results/flex_variants.png"
	@echo "	- results/custom_variants.png"

# Generated outputs only; do not remove results/reference/
clean:
	rm -f $(FLEX_CSVS) $(CUSTOM_CSVS) results/flex_variants.png results/custom_variants.png
