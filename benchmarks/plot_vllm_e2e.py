"""Plot vLLM end-to-end online-trace benchmark results.

Input CSV (produced by ``vllm_e2e_infer.py``):

  **summary CSV** (``--summary``, default ``results/vllm_e2e_online_summary.csv``)
      One row per (mode, variant) run.  Aggregated metrics over all
      requests: mean/p50/p95/p99 TTFT and ITL, throughput (tok/s), total
      wall time, cumulative block-mask build time (FlexAttention only),
      and ``torch.compile`` wall time.

Output figures (4 PNGs, each with 3 panels -- TTFT | ITL | Throughput):

  - **mean**: Mean TTFT, Mean ITL, Throughput
  - **p99**:  P99 TTFT,  P99 ITL,  Throughput
  - **p95**:  P95 TTFT,  P95 ITL,  Throughput
  - **p50**:  P50 TTFT,  P50 ITL,  Throughput

Speedup labels show the Flashlight-over-FlexAttention ratio: a value
**> 1 means Flashlight is better** (lower latency / higher throughput).

Usage::

    python benchmarks/plot_vllm_e2e.py
    python benchmarks/plot_vllm_e2e.py \\
        --summary results/vllm_e2e_online_summary.csv \\
        --output results/vllm_e2e_online.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# ── Display constants ──────────────────────────────────────────────────────

VARIANT_DISPLAY = {
    "causal_packed": "Causal",
    "sliding_window_packed": "Sliding\nWindow",
    "causal_alibi_packed": "ALiBi",
    "causal_softcap_packed": "Softcap",
}

VARIANT_ORDER = [
    "causal_packed",
    "sliding_window_packed",
    "causal_alibi_packed",
    "causal_softcap_packed",
]

VARIANT_COLORS = {
    "causal_packed": "#1b9e77",
    "sliding_window_packed": "#d95f02",
    "causal_alibi_packed": "#7570b3",
    "causal_softcap_packed": "#e7298a",
}

FL_COLOR = "#1f77b4"
FX_COLOR = "#ff7f0e"

MODE_DISPLAY = {"flashlight": "Flashlight", "flex": "FlexAttention"}
MODE_LS = {"flashlight": "-", "flex": "--"}


# ── Data helpers ───────────────────────────────────────────────────────────

def _resolve(path: str) -> str:
    """Resolve *path* relative to the benchmarks/ directory."""
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _load(path: str) -> pd.DataFrame:
    p = _resolve(path)
    try:
        return pd.read_csv(p)
    except FileNotFoundError:
        print(f"Error: {p} not found.", file=sys.stderr)
        sys.exit(1)


def _variants(df: pd.DataFrame) -> list[str]:
    """Variants present in *df*, in canonical order."""
    present = set(df["variant"].unique())
    return [v for v in VARIANT_ORDER if v in present]


def _arr(
    df: pd.DataFrame, mode: str, variants: list[str], col: str,
) -> np.ndarray:
    """One scalar per variant for a given mode and column."""
    vals = []
    for v in variants:
        sub = df[(df["mode"] == mode) & (df["variant"] == v)]
        vals.append(
            float(sub[col].iloc[0])
            if (not sub.empty and col in sub.columns)
            else 0.0
        )
    return np.array(vals)


# ── Bar chart panel ────────────────────────────────────────────────────────

def _bar_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    variants: list[str],
    col: str,
    ylabel: str,
    title: str,
    *,
    higher_is_better: bool = False,
) -> None:
    """Grouped Flashlight / FlexAttention bars for *col*."""
    ind = np.arange(len(variants))
    has_fl = "flashlight" in summary["mode"].values
    has_fx = "flex" in summary["mode"].values
    n = int(has_fl) + int(has_fx)
    if n == 0:
        return
    w = 0.35

    fl = _arr(summary, "flashlight", variants, col) if has_fl else None
    fx = _arr(summary, "flex", variants, col) if has_fx else None

    g = 0
    if has_fl:
        off = ind + (g - (n - 1) / 2) * w
        ax.bar(off, fl, w, color=FL_COLOR, edgecolor="white", lw=0.5)
        g += 1
    if has_fx:
        off = ind + (g - (n - 1) / 2) * w
        ax.bar(off, fx, w, color=FX_COLOR, edgecolor="white", lw=0.5)
        # Speedup labels: ratio so that >1 always means Flex is better.
        if has_fl:
            ymax = max(np.max(fx), np.max(fl))
            for i in range(len(variants)):
                if fl[i] > 0 and fx[i] > 0:
                    sp = (
                        (fx[i] / fl[i])
                        if higher_is_better
                        else (fl[i] / fx[i])
                    )
                    ax.text(
                        off[i], fx[i] + ymax * 0.02,
                        f"{sp:.2f}\u00d7",
                        ha="center", va="bottom", fontsize=8, rotation=90,
                    )

    ax.set_xticks(ind)
    ax.set_xticklabels(
        [VARIANT_DISPLAY.get(v, v) for v in variants], fontsize=9,
    )
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)


# ── Plot specs ────────────────────────────────────────────────────────────

PLOT_SPECS: list[tuple[str, str, str, str]] = [
    # (suffix, ttft_col, itl_col, label_prefix)
    ("mean", "mean_ttft_s", "mean_itl_ms", "Mean"),
    ("p99",  "p99_ttft_s",  "p99_itl_ms",  "P99"),
    ("p95",  "p95_ttft_s",  "p95_itl_ms",  "P95"),
    ("p50",  "p50_ttft_s",  "p50_itl_ms",  "P50"),
]


# ── Main figure ────────────────────────────────────────────────────────────

def _output_path(base: str, suffix: str) -> str:
    """Derive per-plot output path: ``base_suffix.ext``."""
    root, ext = os.path.splitext(base)
    return f"{root}_{suffix}{ext}"


def plot_online(summary: pd.DataFrame, output: str) -> None:
    summary = summary[summary["mode"] != "baseline"].copy()
    variants = _variants(summary)
    if not variants:
        print("No recognised variants in data.", file=sys.stderr)
        return

    tput_col = "tput_tok_s" if "tput_tok_s" in summary.columns else "tokens_per_s"

    for suffix, ttft_col, itl_col, label in PLOT_SPECS:
        if ttft_col not in summary.columns or itl_col not in summary.columns:
            print(f"Skipping {label} plot: missing {ttft_col} or {itl_col}",
                  file=sys.stderr)
            continue

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        _bar_panel(
            axes[0], summary, variants, ttft_col,
            f"{label} TTFT (s)", f"(a) {label} TTFT",
        )
        _bar_panel(
            axes[1], summary, variants, itl_col,
            f"{label} ITL (ms)", f"(b) {label} ITL",
        )
        _bar_panel(
            axes[2], summary, variants, tput_col,
            "Throughput (tok/s)", "(c) Throughput",
            higher_is_better=True,
        )

        fig.legend(
            [Patch(facecolor=FL_COLOR, edgecolor="white"),
             Patch(facecolor=FX_COLOR, edgecolor="white")],
            ["Flashlight", "FlexAttention"],
            loc="upper center", bbox_to_anchor=(0.5, 1.02),
            ncol=2, fontsize=10,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        path = _output_path(output, suffix)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot vLLM end-to-end online-trace results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--summary", default="results/vllm_e2e_online_summary.csv",
        help="Summary CSV (one row per mode x variant run).",
    )
    parser.add_argument(
        "--output", "-o", default="results/vllm_e2e_online.png",
        help="Output figure base path (suffixed per stat level).",
    )
    # Legacy single-CSV path (e.g. results/vllm_e2e_online.csv)
    parser.add_argument("--csv", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    summary = _load(args.csv if args.csv else args.summary)
    plot_online(summary, _resolve(args.output))


if __name__ == "__main__":
    main()
