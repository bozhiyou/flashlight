"""Plot vLLM end-to-end online-trace benchmark results.

Input CSVs (produced by ``vllm_e2e_infer.py``):

  **summary CSV** (``--summary``, default ``results/vllm_e2e_online_summary.csv``)
      One row per (mode, variant) run.  Aggregated metrics over all
      requests: mean/p50/p95/p99 TTFT, throughput (tok/s), total wall
      time, cumulative block-mask build time (FlexAttention only), and
      ``torch.compile`` wall time.

  **per-request CSV** (``--per-request``, default
      ``results/vllm_e2e_online_per_request.csv``)
      One row per individual request.  Records arrival time, TTFT, decode
      time, output length, and (when available) batch size at prefill.
      Used to build distributional plots that the summary cannot capture.

Output figure (2x2 panels):

  **(a) Mean TTFT** -- Time to First Token averaged over all requests
      (lower is better).  Measures how quickly the first output token is
      returned after a request arrives; dominated by the prefill pass,
      which is where Flashlight patching takes effect.

  **(b) Throughput** -- Total output tokens / total wall time (higher is
      better).  End-to-end metric including both prefill and decode;
      since decode is unpatched, differences stem from prefill efficiency
      and scheduling overhead.

  **(c) TTFT CDF** -- Cumulative Distribution Function of per-request
      TTFT.  Each curve plots the fraction of requests (y-axis) whose
      TTFT is at most x seconds.  Curves further *left* indicate lower
      (better) latency.  Solid lines = Flashlight, dashed =
      FlexAttention; colour = attention variant.

  **(d) Compile overhead** -- Wall-clock time spent inside
      ``torch.compile`` during the run (lower is better).  One-time
      cold-start cost amortised across requests; measured via
      ``time.perf_counter`` around the first compiled call that triggers
      graph capture and code generation.

Speedup labels on panels (a), (b), (d) show the Flashlight-over-
FlexAttention ratio: a value **> 1 means Flashlight is better** (lower
latency, higher throughput, or shorter compile time).

Usage::

    python benchmarks/plot_vllm_e2e.py
    python benchmarks/plot_vllm_e2e.py \\
        --summary results/vllm_e2e_online_summary.csv \\
        --per-request results/vllm_e2e_online_per_request.csv \\
        --output results/vllm_e2e_online.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


# ── CDF panel ─────────────────────────────────────────────────────────────

def _cdf_panel(
    ax: plt.Axes, per_req: pd.DataFrame, variants: list[str],
) -> None:
    """Per-request TTFT cumulative distribution curves."""
    for variant in variants:
        color = VARIANT_COLORS.get(variant, "gray")
        for mode in ("flashlight", "flex"):
            sub = per_req[
                (per_req["mode"] == mode) & (per_req["variant"] == variant)
            ]
            if sub.empty:
                continue
            ttft = np.sort(sub["ttft_s"].dropna().values)
            cdf = np.arange(1, len(ttft) + 1) / len(ttft)
            ax.plot(ttft, cdf, ls=MODE_LS[mode], color=color, lw=1.5)

    ax.set_xlabel("TTFT (s)", fontsize=10)
    ax.set_ylabel("CDF", fontsize=10)
    ax.set_title("(c) TTFT Distribution", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Compact legend: linestyle = mode, colour = variant
    handles = [
        Line2D([], [], color="gray", ls="-", lw=1.5, label="Flashlight"),
        Line2D([], [], color="gray", ls="--", lw=1.5, label="FlexAttention"),
    ]
    for v in variants:
        handles.append(
            Line2D(
                [], [],
                color=VARIANT_COLORS.get(v, "gray"),
                ls="-", lw=2.5,
                label=VARIANT_DISPLAY.get(v, v).replace("\n", " "),
            ),
        )
    ax.legend(handles=handles, fontsize=7, loc="lower right", ncol=2)


# ── Main figure ────────────────────────────────────────────────────────────

def plot_online(
    summary: pd.DataFrame,
    per_req: pd.DataFrame | None,
    output: str,
) -> None:
    summary = summary[summary["mode"] != "baseline"].copy()
    variants = _variants(summary)
    if not variants:
        print("No recognised variants in data.", file=sys.stderr)
        return

    ttft_col = "mean_ttft_s" if "mean_ttft_s" in summary.columns else "ttft_s"
    tput_col = "tput_tok_s" if "tput_tok_s" in summary.columns else "tokens_per_s"

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) Mean TTFT
    _bar_panel(
        axes[0, 0], summary, variants, ttft_col,
        "Mean TTFT (s)", "(a) Time to First Token",
    )

    # (b) Throughput
    _bar_panel(
        axes[0, 1], summary, variants, tput_col,
        "Throughput (tok/s)", "(b) Decoding Throughput",
        higher_is_better=True,
    )

    # (c) Per-request TTFT CDF
    if per_req is not None and not per_req.empty:
        _cdf_panel(axes[1, 0], per_req, variants)
    else:
        fig.delaxes(axes[1, 0])

    # (d) Compile overhead
    if "compile_time_s" in summary.columns:
        _bar_panel(
            axes[1, 1], summary, variants, "compile_time_s",
            "Compile time (s)", "(d) Compilation Overhead",
        )
    else:
        fig.delaxes(axes[1, 1])

    # Shared bar-chart legend
    fig.legend(
        [Patch(facecolor=FL_COLOR, edgecolor="white"),
         Patch(facecolor=FX_COLOR, edgecolor="white")],
        ["Flashlight", "FlexAttention"],
        loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved {output}")


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
        "--per-request", default="results/vllm_e2e_online_per_request.csv",
        dest="per_request", help="Per-request CSV for CDF plots.",
    )
    parser.add_argument(
        "--output", "-o", default="results/vllm_e2e_online.png",
        help="Output figure path.",
    )
    # Legacy single-CSV path (e.g. results/vllm_e2e_online.csv)
    parser.add_argument("--csv", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.csv:
        summary = _load(args.csv)
        per_req = None
    else:
        summary = _load(args.summary)
        pr_path = _resolve(args.per_request)
        per_req = pd.read_csv(pr_path) if os.path.exists(pr_path) else None

    plot_online(summary, per_req, _resolve(args.output))


if __name__ == "__main__":
    main()
