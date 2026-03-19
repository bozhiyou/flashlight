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

    if df.empty:
        print(f"Error: {csv_path} contains no non-baseline data rows. "
              "Re-run data collection (delete the CSV and run make again).",
              file=sys.stderr)
        sys.exit(1)

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


def _get_metric_arrays(df: pd.DataFrame, configs_or_variants: list, config_col: str, metric: str) -> tuple:
    has_flashlight = "Flashlight" in df["impl"].values
    has_cache_hit = "Flex (Cache Hit)" in df["impl"].values
    has_cache_miss = "Flex (Cache Miss)" in df["impl"].values

    def _array(impl_label: str) -> np.ndarray:
        vals = []
        for cv in configs_or_variants:
            if config_col == "config":
                # For synthetic workloads
                sub = df[(df["impl"] == impl_label) & (df["config"] == cv)]
                vals.append(float(sub[metric].iloc[0]) if not sub.empty else 0.0)
            else:
                # For trace workloads
                sub = df[(df["impl"] == impl_label) & (df["variant"] == cv)]
                vals.append(float(sub[metric].mean()) if not sub.empty else 0.0)
        return np.array(vals)

    flashlight_arr = _array("Flashlight") if has_flashlight else np.zeros(len(configs_or_variants))
    cache_hit_arr = _array("Flex (Cache Hit)") if has_cache_hit else np.zeros(len(configs_or_variants))
    cache_miss_arr = _array("Flex (Cache Miss)") if has_cache_miss else np.zeros(len(configs_or_variants))

    # Flex mask overhead = cache miss − cache hit (clamped to 0).
    mask_overhead = np.maximum(cache_miss_arr - cache_hit_arr, 0.0)

    groups = []
    if has_flashlight:
        groups.append(("Flashlight", flashlight_arr, None, "#1f77b4"))
    if has_cache_hit or has_cache_miss:
        groups.append(("FlexAttention (Kernel)", cache_hit_arr, mask_overhead, "#ff7f0e"))

    return groups, has_flashlight, has_cache_hit, has_cache_miss, flashlight_arr, cache_hit_arr, cache_miss_arr, mask_overhead


def _add_speedup_labels(ax, groups, ind, width, has_flashlight, cache_hit_arr, mask_overhead, flashlight_arr, n_groups, is_grouped_center=False):
    for g_idx, (label, bottom_vals, top_vals, color) in enumerate(groups):
        if is_grouped_center:
            offset = ind + (g_idx - (n_groups - 1) / 2) * width
        else:
            offset = ind + g_idx * width
            
        if "Kernel" in label and has_flashlight:
            for i_idx in range(len(ind)):
                flex_total = cache_hit_arr[i_idx] + mask_overhead[i_idx]
                fl_val = flashlight_arr[i_idx]
                if fl_val > 0 and flex_total > 0:
                    speedup = flex_total / fl_val
                    bar_top = flex_total
                    ax.text(
                        offset[i_idx], bar_top + 0.002,
                        f"{speedup:.2f}x",
                        ha="center", va="bottom", fontsize=10,
                        rotation=90, color="black",
                    )


def _draw_bars_and_legend(ax, groups, ind, width, is_grouped_center=False, collect_legend=True):
    legend_handles, legend_labels = [], []
    from matplotlib.patches import Patch
    
    n_groups = len(groups)
    for g_idx, (label, bottom_vals, top_vals, color) in enumerate(groups):
        if is_grouped_center:
            offset = ind + (g_idx - (n_groups - 1) / 2) * width
        else:
            offset = ind + g_idx * width
            
        ax.bar(offset, bottom_vals, width, color=color, edgecolor="white", linewidth=0.5)

        if top_vals is not None:
            ax.bar(
                offset, top_vals, width, bottom=bottom_vals,
                color="#2ca02c", edgecolor="white", linewidth=0.5,
            )

        if collect_legend:
            legend_handles.append(Patch(facecolor=color, edgecolor="white", label=label))
            legend_labels.append(label)
            if top_vals is not None:
                legend_handles.append(Patch(facecolor="#2ca02c", edgecolor="white", label="FlexAttention (Block Mask)"))
                legend_labels.append("FlexAttention (Block Mask)")
                
    return legend_handles, legend_labels

def plot_synth_bars(df: pd.DataFrame, output: str) -> None:
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

        df_sub = df[df["variant"] == variant]
        groups, has_flashlight, has_cache_hit, has_cache_miss, flashlight_arr, cache_hit_arr, cache_miss_arr, mask_overhead = _get_metric_arrays(df_sub, configs, "config", "ttft_s")

        n_groups = len(groups)
        if n_groups == 0:
            continue
        width = 0.8 / n_groups

        handles, labels = _draw_bars_and_legend(ax, groups, ind, width, is_grouped_center=False, collect_legend=(col_idx == 0))
        if col_idx == 0:
            legend_handles, legend_labels = handles, labels

        _add_speedup_labels(ax, groups, ind, width, has_flashlight, cache_hit_arr, mask_overhead, flashlight_arr, n_groups, is_grouped_center=False)

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


def plot_trace_bars(df: pd.DataFrame, output: str) -> None:
    """Grouped bar chart of TTFT and Throughput for trace workloads.
    
    Creates a 1x2 grid of subplots:
      - Left: TTFT (stacked Flex vs Flashlight)
      - Right: Throughput (tokens/s)
    """
    variants = sorted(
        df["variant"].unique(),
        key=lambda v: VARIANT_ORDER.index(v) if v in VARIANT_ORDER else 999,
    )
    if not variants:
        print("No trace data to plot.", file=sys.stderr)
        return

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), squeeze=False)
    
    ind = np.arange(len(variants))
    
    has_flashlight = "Flashlight" in df["impl"].values
    has_cache_hit = "Flex (Cache Hit)" in df["impl"].values
    has_cache_miss = "Flex (Cache Miss)" in df["impl"].values

    def _get_metric_array(impl_label: str, metric: str) -> np.ndarray:
        vals = []
        for variant in variants:
            sub = df[(df["impl"] == impl_label) & (df["variant"] == variant)]
            if not sub.empty:
                # If multiple traces, take the mean (or you could group by trace, but assuming 1 trace for now)
                vals.append(sub[metric].mean())
            else:
                vals.append(0.0)
        return np.array(vals)

    # ── Left Subplot: TTFT ──────────────────────────────────────────
    ax_ttft = axes[0, 0]
    
    groups_ttft, has_flashlight, has_cache_hit, has_cache_miss, flashlight_ttft, cache_hit_ttft, cache_miss_ttft, mask_overhead = _get_metric_arrays(df, variants, "variant", "ttft_s")

    n_groups = len(groups_ttft)
    if n_groups > 0:
        width = 0.8 / n_groups
        legend_handles, legend_labels = _draw_bars_and_legend(ax_ttft, groups_ttft, ind, width, is_grouped_center=True, collect_legend=True)
        _add_speedup_labels(ax_ttft, groups_ttft, ind, width, has_flashlight, cache_hit_ttft, mask_overhead, flashlight_ttft, n_groups, is_grouped_center=True)

        ax_ttft.set_xticks(ind)
        ax_ttft.set_xticklabels([VARIANT_DISPLAY.get(v, v) for v in variants], fontsize=10)
        ax_ttft.set_ylabel("TTFT (s)", fontsize=12)
        ax_ttft.set_title("Time To First Token", fontsize=12)

        fig.legend(
            legend_handles, legend_labels,
            loc="upper center", bbox_to_anchor=(0.5, 1.06),
            ncol=len(legend_labels), fontsize=9,
        )

    # ── Right Subplot: Throughput ───────────────────────────────────
    ax_tput = axes[0, 1]
    
    groups_tput = []
    if has_flashlight:
        groups_tput.append(("Flashlight", _get_metric_array("Flashlight", "tokens_per_s"), "#1f77b4"))
    if has_cache_hit:
        groups_tput.append(("Flex (Cache Hit)", _get_metric_array("Flex (Cache Hit)", "tokens_per_s"), "#ff7f0e"))
    if has_cache_miss:
        groups_tput.append(("Flex (Cache Miss)", _get_metric_array("Flex (Cache Miss)", "tokens_per_s"), "#2ca02c"))

    n_groups_tput = len(groups_tput)
    if n_groups_tput > 0:
        width = 0.8 / n_groups_tput
        
        for g_idx, (label, vals, color) in enumerate(groups_tput):
            offset = ind + (g_idx - (n_groups_tput - 1) / 2) * width
            bars = ax_tput.bar(offset, vals, width, color=color, edgecolor="white", linewidth=0.5)
            
            # Speedup labels on Flex bars vs Flashlight
            if "Flex" in label and has_flashlight:
                fl_tput = _get_metric_array("Flashlight", "tokens_per_s")
                for i_v in range(len(variants)):
                    if fl_tput[i_v] > 0 and vals[i_v] > 0:
                        speedup = vals[i_v] / fl_tput[i_v]
                        ax_tput.text(
                            offset[i_v], vals[i_v] + (max(vals)*0.01),
                            f"{speedup:.2f}x",
                            ha="center", va="bottom", fontsize=10,
                            rotation=90, color="black",
                        )

        ax_tput.set_xticks(ind)
        ax_tput.set_xticklabels([VARIANT_DISPLAY.get(v, v) for v in variants], fontsize=10)
        ax_tput.set_ylabel("Throughput (tokens/s)", fontsize=12)
        ax_tput.set_title("Decoding Throughput", fontsize=12)

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
    
    if "trace" in df.columns:
        plot_trace_bars(df, args.output)
    else:
        plot_synth_bars(df, args.output)


if __name__ == "__main__":
    main()
