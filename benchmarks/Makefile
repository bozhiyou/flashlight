# Makefile for reproducing FlashLight paper evaluation (Figures 2–4).
#
# Usage:
#   make all          — run all benchmarks and generate all figures
#   make data         — collect all benchmark data (GPU required)
#   make figures      — generate all figures from existing CSVs (CPU only)
#   make fig2_fig3    — end-to-end for Figures 2 & 3
#   make fig4         — end-to-end for Figure 4
#   make clean        — remove generated CSVs and figures (keeps results/reference/)
#
# Each data-collection target launches a separate GPU job.
# Run from the benchmarks/ directory or use `make -C benchmarks/`.

.DEFAULT_GOAL := all
PYTHON ?= python
# Target SM frequency (MHz) for warmup when sudo nvidia-smi -lgc is unavailable (e.g. TACC)
FL_GPU_CLOCK_FREQ_MHZ ?= 1290

all: data figures

# ─── Figures 2 & 3: FlexAttention-supported variants ─────────────────────────

FIG23_SCRIPT  := run_fig2_fig3_flex_variants.py
FIG23_CSVS    := results/all.csv \
                 results/all_flex.csv \
                 results/all_flexnocache.csv
                 # results/all_torchcompile.csv
FIG23_PLOT    := plot_fig2_fig3.py

results/all.csv:
	FL_GPU_CLOCK_FREQ_MHZ=$(FL_GPU_CLOCK_FREQ_MHZ) $(PYTHON) $(FIG23_SCRIPT) --flashlight --output results/all

results/all_flex.csv:
	FL_GPU_CLOCK_FREQ_MHZ=$(FL_GPU_CLOCK_FREQ_MHZ) $(PYTHON) $(FIG23_SCRIPT) --flex --output results/all_flex

results/all_flexnocache.csv:
	FL_GPU_CLOCK_FREQ_MHZ=$(FL_GPU_CLOCK_FREQ_MHZ) $(PYTHON) $(FIG23_SCRIPT) --flex --no-mask-cache --output results/all_flexnocache

# results/all_torchcompile.csv:
# 	FL_GPU_CLOCK_FREQ_MHZ=$(FL_GPU_CLOCK_FREQ_MHZ) $(PYTHON) $(FIG23_SCRIPT) --torch.compile --output results/all_torchcompile

# ─── Figure 4: FlexAttention-unsupported variants ────────────────────────────

FIG4_CSVS     := results/diff_attn.csv \
                 results/evo_attn.csv
FIG4_PLOT     := plot_fig4.py

results/diff_attn.csv:
	FL_GPU_CLOCK_FREQ_MHZ=$(FL_GPU_CLOCK_FREQ_MHZ) $(PYTHON) run_fig4_diff_attn.py

results/evo_attn.csv:
	FL_GPU_CLOCK_FREQ_MHZ=$(FL_GPU_CLOCK_FREQ_MHZ) $(PYTHON) run_fig4_evoformer.py

# ─── Figure generation (CPU only) ────────────────────────────────────────────
# Plot scripts write to CWD with hardcoded names; move into results/.

results/fig2_fig3.png: $(FIG23_CSVS) $(FIG23_PLOT)
	$(PYTHON) $(FIG23_PLOT) --plot bar
	mv flex-able-bar.png $@

results/fig4.png: $(FIG4_CSVS) $(FIG4_PLOT)
	$(PYTHON) $(FIG4_PLOT) \
	  --diff-attn results/diff_attn.csv \
	  --evo-attn results/evo_attn.csv
	mv flex-unable-bar.png $@

# ─── Aggregate targets ───────────────────────────────────────────────────────

.PHONY: all data fig2_fig3 fig4 figures clean

data: $(FIG23_CSVS) $(FIG4_CSVS)

fig2_fig3: results/fig2_fig3.png

fig4: results/fig4.png

figures: fig2_fig3 fig4
	@echo "Figures generated successfully:"
	@echo "	- results/fig2_fig3.png"
	@echo "	- results/fig4.png"

# Generated outputs only; do not remove results/reference/
clean:
	rm -f $(FIG23_CSVS) $(FIG4_CSVS) results/fig2_fig3.png results/fig4.png
