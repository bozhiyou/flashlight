"""Plot results from vllm_e2e_infer.py benchmarks.

Produces a grouped bar chart of **TTFT** (time-to-first-token) — the only
metric that varies across modes since decode is unpatched and identical.

Layout mirrors ``plot_flex_variants.py``:
- One subplot per attention variant (columns)
- X-axis: (batch_size, input_len) configs
- Bars: Flashlight | FlexAttention (Kernel) + FlexAttention (Block Mask)
- Speedup labels: Flex(total) / Flashlight

Usage::

    python benchmarks/plot_vllm_e2e.py
    python benchmarks/plot_vllm_e2e.py --csv results/vllm_e2e.csv --output results/vllm_e2e.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Pretty names ────────────────────────────────────────────────────────────

VARIANT_DISPLAY = {
    "causal": "Causal",
    "sliding_window": "Sliding Window",
    "prefix_lm": "PrefixLM",
    "document_mask": "Document Mask",
    "full": "Vanilla",
    "alibi": "ALiBi",
    "softcap": "Softcap",
}

VARIANT_ORDER = list(VARIANT_DISPLAY.keys())


# ── Data loading ────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(bench_dir, csv_path)
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.", file=sys.stderr)
        sys.exit(1)

    # Drop baseline rows — baseline is unpatched vLLM (always "vanilla"),
    # not meaningful for comparing Flashlight vs FlexAttention variants.
    df = df[df["mode"] != "baseline"].copy()

    # Map (mode, mask_cache) → implementation label.
    def _impl(row: pd.Series) -> str:
        if row["mode"] == "flashlight":
            return "Flashlight"
        # mode == "flex"
        if row.get("mask_cache", True):
            return "Flex (Cache Hit)"
        return "Flex (Cache Miss)"

    df["impl"] = df.apply(_impl, axis=1)
    df["variant_display"] = df["variant"].map(VARIANT_DISPLAY).fillna(df["variant"])
    df["config"] = df.apply(
        lambda r: f"({int(r['batch_size'])}, {int(r['input_len'])})", axis=1,
    )
    return df


# ── TTFT bar chart (main figure) ───────────────────────────────────────────

def plot_ttft_bars(df: pd.DataFrame, output: str) -> None:
    """Grouped bar chart of TTFT, one subplot per variant.

    FlexAttention bars are stacked: kernel (cache hit) + mask overhead
    (cache miss − cache hit), mirroring the kernel benchmark decomposition.
    """
    variants = sorted(
        df["variant"].unique(),
        key=lambda v: VARIANT_ORDER.index(v) if v in VARIANT_ORDER else 999,
    )
    if not variants:
        print("No data to plot.", file=sys.stderr)
        return

    # Determine config labels (sorted by batch_size then input_len).
    config_keys = (
        df.drop_duplicates(subset=["batch_size", "input_len"])
        .sort_values(["batch_size", "input_len"])
    )
    configs = config_keys.apply(
        lambda r: f"({int(r['batch_size'])}, {int(r['input_len'])})", axis=1,
    ).tolist()

    n_variants = len(variants)
    fig, axes = plt.subplots(
        nrows=1, ncols=n_variants,
        figsize=(5 * n_variants, 5),
        squeeze=False,
    )

    legend_handles, legend_labels = [], []

    for col_idx, variant in enumerate(variants):
        ax = axes[0, col_idx]
        ind = np.arange(len(configs))

        # ── Collect per-impl TTFT arrays ────────────────────────────
        def _ttft_array(impl_label: str) -> np.ndarray:
            sub = df[(df["impl"] == impl_label) & (df["variant"] == variant)]
            vals = []
            for cfg in configs:
                row = sub[sub["config"] == cfg]
                vals.append(float(row["ttft_s"].iloc[0]) if not row.empty else 0.0)
            return np.array(vals)

        has_flashlight = "Flashlight" in df["impl"].values
        has_cache_hit = "Flex (Cache Hit)" in df["impl"].values
        has_cache_miss = "Flex (Cache Miss)" in df["impl"].values

        flashlight_ttft = _ttft_array("Flashlight") if has_flashlight else np.zeros(len(configs))
        cache_hit_ttft = _ttft_array("Flex (Cache Hit)") if has_cache_hit else np.zeros(len(configs))
        cache_miss_ttft = _ttft_array("Flex (Cache Miss)") if has_cache_miss else np.zeros(len(configs))

        # Flex mask overhead = cache miss − cache hit (clamped to 0).
        mask_overhead = np.maximum(cache_miss_ttft - cache_hit_ttft, 0.0)

        # ── Determine bar groups ────────────────────────────────────
        groups = []
        if has_flashlight:
            groups.append(("Flashlight", flashlight_ttft, None, "#1f77b4"))
        if has_cache_hit or has_cache_miss:
            groups.append(("FlexAttention (Kernel)", cache_hit_ttft, mask_overhead, "#ff7f0e"))

        n_groups = len(groups)
        if n_groups == 0:
            continue
        width = 0.8 / n_groups

        for g_idx, (label, bottom_vals, top_vals, color) in enumerate(groups):
            offset = ind + g_idx * width
            bars = ax.bar(offset, bottom_vals, width, color=color, edgecolor="white", linewidth=0.5)

            if top_vals is not None:
                ax.bar(
                    offset, top_vals, width, bottom=bottom_vals,
                    color="#2ca02c", edgecolor="white", linewidth=0.5,
                )

            # Speedup labels on Flex bars: Flex(total) / Flashlight.
            if "Kernel" in label and has_flashlight:
                for i_cfg in range(len(configs)):
                    flex_total = cache_hit_ttft[i_cfg] + mask_overhead[i_cfg]
                    fl_val = flashlight_ttft[i_cfg]
                    if fl_val > 0 and flex_total > 0:
                        speedup = flex_total / fl_val
                        bar_top = flex_total
                        ax.text(
                            offset[i_cfg], bar_top + 0.002,
                            f"{speedup:.2f}x",
                            ha="center", va="bottom", fontsize=10,
                            rotation=90, color="black",
                        )

            # Collect legend handles from first subplot only.
            if col_idx == 0:
                from matplotlib.patches import Patch
                if top_vals is None:
                    legend_handles.append(Patch(facecolor=color, edgecolor="white", label=label))
                    legend_labels.append(label)
                else:
                    legend_handles.append(Patch(facecolor=color, edgecolor="white", label=label))
                    legend_labels.append(label)
                    legend_handles.append(Patch(facecolor="#2ca02c", edgecolor="white", label="FlexAttention (Block Mask)"))
                    legend_labels.append("FlexAttention (Block Mask)")

        ax.set_xticks(ind + width * (n_groups - 1) / 2)
        ax.set_xticklabels(configs, rotation=90, fontsize=10)
        ax.set_xlabel("(Batch Size, Input Length)", fontsize=10)
        if col_idx == 0:
            ax.set_ylabel("TTFT (s)", fontsize=12)
        title = VARIANT_DISPLAY.get(variant, variant)
        ax.set_title(title, fontsize=12)

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="upper center", bbox_to_anchor=(0.5, 1.06),
            ncol=len(legend_labels), fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved {output}")


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot vLLM end-to-end inference benchmark results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        default="results/vllm_e2e.csv",
        help="Input CSV (produced by vllm_e2e_infer.py).",
    )
    parser.add_argument(
        "--output",
        default="results/vllm_e2e.png",
        help="Output figure path.",
    )
    args = parser.parse_args()

    df = load_data(args.csv)
    plot_ttft_bars(df, args.output)


if __name__ == "__main__":
    main()
