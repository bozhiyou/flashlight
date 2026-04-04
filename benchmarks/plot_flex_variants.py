import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def load_and_prepare_data(ours_csv, flex_csv, flexnocache_csv, torchcompile_csv=None, flashinfer_csv=None):
    """
    Loads and preprocesses the benchmark data from CSV files.
    """
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    ours_csv = os.path.join(bench_dir, ours_csv) if not os.path.isabs(ours_csv) else ours_csv
    flex_csv = os.path.join(bench_dir, flex_csv) if not os.path.isabs(flex_csv) else flex_csv
    flexnocache_csv = os.path.join(bench_dir, flexnocache_csv) if not os.path.isabs(flexnocache_csv) else flexnocache_csv
    if torchcompile_csv is not None:
        torchcompile_csv = os.path.join(bench_dir, torchcompile_csv) if not os.path.isabs(torchcompile_csv) else torchcompile_csv
    if flashinfer_csv is not None:
        flashinfer_csv = os.path.join(bench_dir, flashinfer_csv) if not os.path.isabs(flashinfer_csv) else flashinfer_csv
    benchmark_names = {
        "full": "vanilla",
        "full_with_alibi": "ALiBi",
        "full_with_softcap": "softcap",
        "full_with_causal": "causal",
        "full_with_sliding_window": "sliding window",
        "full_with_prefix_lm": "prefix LM",
        "full_with_document_mask": "document mask",
    }

    try:
        ours_df = pd.read_csv(ours_csv)
        flex_df = pd.read_csv(flex_csv)
        flexnocache_df = pd.read_csv(flexnocache_csv)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure '{ours_csv}', '{flex_csv}', and '{flexnocache_csv}' are in the same directory.")
        sys.exit(1)

    ours_df = ours_df.rename(columns={"Implementation": "Benchmark"})
    ours_df["Implementation"] = "Flashlight"

    flex_df = flex_df.rename(columns={"Implementation": "Benchmark"})
    flex_df["Benchmark"] = flex_df["Benchmark"].str.replace("flex_", "")
    flex_df["Implementation"] = "Flex (Cache Hit)"
    # flex_df["FW_Time_ms"] = flex_df["FW_Time_ms"].replace(-1, float('nan'))

    flexnocache_df = flexnocache_df.rename(columns={"Implementation": "Benchmark"})
    flexnocache_df["Benchmark"] = flexnocache_df["Benchmark"].str.replace("flex_", "")
    flexnocache_df["Implementation"] = "Flex (Cache Miss)"
    # flexnocache_df["FW_Time_ms"] = flexnocache_df["FW_Time_ms"].replace(-1, float('nan'))

    dfs_to_concat = [ours_df, flex_df, flexnocache_df]

    if torchcompile_csv is not None:
        try:
            torchcompile_df = pd.read_csv(torchcompile_csv)
        except FileNotFoundError as e:
            print(f"Error: {e}. Make sure '{torchcompile_csv}' is in the same directory.")
            sys.exit(1)
    else:
        torchcompile_df = pd.DataFrame(columns=ours_df.columns)

    if not torchcompile_df.empty:
        torchcompile_df = torchcompile_df.rename(columns={"Implementation": "Benchmark"})
        torchcompile_df["Implementation"] = "torch.compile"
        dfs_to_concat.append(torchcompile_df)

    # Optional: FlashInfer
    if flashinfer_csv is not None:
        try:
            flashinfer_df = pd.read_csv(flashinfer_csv)
        except FileNotFoundError:
            print(f"INFO: FlashInfer data not found at '{flashinfer_csv}'; skipping.")
            flashinfer_df = pd.DataFrame(columns=ours_df.columns)
    else:
        flashinfer_df = pd.DataFrame(columns=ours_df.columns)

    if not flashinfer_df.empty:
        flashinfer_df = flashinfer_df.rename(columns={"Implementation": "Benchmark"})
        flashinfer_df["Benchmark"] = flashinfer_df["Benchmark"].str.replace("flashinfer_", "")
        flashinfer_df["Implementation"] = "FlashInfer"
        dfs_to_concat.append(flashinfer_df)

    combined_df = pd.concat(dfs_to_concat, ignore_index=True)
    combined_df['Benchmark'] = combined_df['Benchmark'].replace(benchmark_names)
    combined_df["nheads_kv"] = combined_df["nheads"] // combined_df["group_size"]

    # Calculate speedup
    speedup_df = combined_df.pivot_table(index=["Benchmark", "batch_size", "seqlen", "nheads", "nheads_kv", "headdim", "group_size", "dropout_p"],
                                         columns="Implementation",
                                         values="FW_Time_ms").reset_index()
    # result NaN if "Flex" is NaN
    speedup_df["Speedup"] = speedup_df["Flex (Cache Hit)"] / speedup_df["Flashlight"]

    # Prepare data for plotting
    combined_df["nheads_headdim"] = combined_df.apply(lambda row: f"({row['nheads']}, {row['headdim']})", axis=1)
    combined_df["batch_seqlen"] = combined_df.apply(lambda row: f"({row['batch_size']}, {row['seqlen']})", axis=1)
    speedup_df["batch_seqlen"] = speedup_df.apply(lambda row: f"({row['batch_size']}, {row['seqlen']})", axis=1)
    speedup_df["nheads_headdim"] = speedup_df.apply(lambda row: f"({row['nheads']}, {row['headdim']})", axis=1)

    benchmarks = combined_df["Benchmark"].unique()

    return combined_df, speedup_df, benchmarks


def _format_benchmark_title(benchmark: str) -> str:
    """Title case for subplot titles (matches bar plot / paper figures)."""
    if benchmark in ("ALiBi", "Prefix LM"):
        return benchmark
    title = benchmark[0].upper() + benchmark[1:].lower()
    title = " ".join(t[0].upper() + t[1:].lower() for t in title.split())
    return title


def plot_line_charts(combined_df, benchmarks, output_path: str = 'flex-able-torch.png'):
    """
    Generates and saves the line charts as in the original script.
    Rows follow group_size (same as the bar plot): one row per attention grouping (e.g. MHA vs GQA).
    Columns are benchmarks.

    Visual style matches the paper line figure (e.g. flex-able-torch-h100.png): linear time axis,
    no grid, legend above the grid, implementation-specific colors / dashes / markers.
    """
    # Line aesthetics aligned with paper appendix figure (e.g. flex-able-torch-h100.png)
    _LINE_IMPL_ORDER = [
        "Flashlight",
        "Flex (Cache Hit)",
        "Flex (Cache Miss)",
        "torch.compile",
        "FlashInfer",
    ]
    _LINE_PALETTE = {
        "Flashlight": "#1f77b4",
        "Flex (Cache Hit)": "#ff7f0e",
        "Flex (Cache Miss)": "#2ca02c",
        "torch.compile": "#d62728",
        "FlashInfer": "#9467bd",
    }
    _LINE_DASHES = {
        "Flashlight": (),
        "Flex (Cache Hit)": (4, 2),
        "Flex (Cache Miss)": (1, 2),
        "torch.compile": (3, 2, 1, 2),
        "FlashInfer": (5, 2),
    }
    # All filled markers: seaborn forbids mixing filled (o, s) with line-art (x, +).
    _LINE_MARKERS = {
        "Flashlight": "o",
        "Flex (Cache Hit)": "X",
        "Flex (Cache Miss)": "s",
        "torch.compile": "P",
        "FlashInfer": "^",
    }


    def _mpl_linestyle(dash_tuple: tuple) -> str | tuple:
        """Map seaborn-style dash tuples to a matplotlib linestyle."""
        if dash_tuple is None or len(dash_tuple) == 0:
            return "-"
        return (0, dash_tuple)

    def _batch_seqlen_sort_key(label) -> tuple:
        inner = str(label).strip("()")
        a, b = inner.split(",")
        return (int(a.strip()), int(b.strip()))

    print("Generating line charts...")
    sns.set_theme(style="white")
    group_sizes = sorted(combined_df["group_size"].unique())
    n_rows = len(group_sizes)
    n_cols = len(benchmarks)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
        squeeze=False,
    )
    fig.set_size_inches(17, max(7, 3.5 * n_rows))

    for row_idx, group_size in enumerate(group_sizes):
        for col_idx, benchmark in enumerate(benchmarks):
            ax1 = axes[row_idx, col_idx]

            benchmark_data = combined_df[
                (combined_df["Benchmark"] == benchmark)
                & (combined_df["group_size"] == group_size)
            ]
            if benchmark_data.empty:
                ax1.set_title(_format_benchmark_title(benchmark))
                ax1.text(
                    0.5,
                    0.5,
                    "No data",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax1.transAxes,
                    color="gray",
                )
                ax1.set_xticks([])
                ax1.set_yticks([])
                continue

            hq = benchmark_data["nheads"].unique()
            hkv = benchmark_data["nheads_kv"].unique()
            assert len(hq) == 1 and len(hkv) == 1, (hq, hkv)

            impl_present = [k for k in _LINE_IMPL_ORDER if k in benchmark_data["Implementation"].values]
            x_order = sorted(benchmark_data["batch_seqlen"].unique(), key=_batch_seqlen_sort_key)
            plot_df = benchmark_data.copy()
            plot_df["batch_seqlen"] = pd.Categorical(
                plot_df["batch_seqlen"], categories=x_order, ordered=True
            )
            palette = {k: _LINE_PALETTE[k] for k in impl_present}
            dashes = {k: _LINE_DASHES[k] for k in impl_present}
            markers = {k: _LINE_MARKERS[k] for k in impl_present}

            sns.lineplot(
                data=plot_df,
                x="batch_seqlen",
                y="FW_Time_ms",
                hue="Implementation",
                style="Implementation",
                hue_order=impl_present,
                style_order=impl_present,
                markers=markers,
                dashes=dashes,
                ax=ax1,
                palette=palette,
                linewidth=1.6,
                markersize=5.5,
            )
            ax1.set_xlabel("")
            if col_idx == 0:
                label = "MHA" if hq[0] == 16 and hkv[0] == 16 else "GQA"
                ax1.set_ylabel(f"Time (ms) for {label}", fontsize=12)
            else:
                ax1.yaxis.get_label().set_visible(False)
            ax1.tick_params(
                axis="x",
                rotation=90,
                labelsize=11,
                direction="out",
                length=4,
                width=0.8,
                bottom=True,
                top=False,
            )
            ax1.tick_params(
                axis="y",
                labelsize=11,
                direction="out",
                length=4,
                width=0.8,
                left=True,
                right=False,
            )
            ax1.grid(False)
            ax1.set_facecolor("white")
            ax1.set_title(_format_benchmark_title(benchmark), fontsize=12)

            ax1.get_legend().remove()

    # Full-dataset legend: the first subplot may omit an implementation (e.g. no cache-miss row there).
    impls_legend = [k for k in _LINE_IMPL_ORDER if k in combined_df["Implementation"].unique()]
    legend_handles = [
        Line2D(
            [],
            [],
            color=_LINE_PALETTE[k],
            linestyle=_mpl_linestyle(_LINE_DASHES[k]),
            marker=_LINE_MARKERS[k],
            markersize=5.5,
            linewidth=1.6,
            label=k.replace("Flex (Cache Hit)", "FlexAttention (Cache Hit)").replace(
            "Flex (Cache Miss)", "FlexAttention (Cache Miss)")
        )
        for k in impls_legend
    ]
    legend_labels = [str(h.get_label()) for h in legend_handles]

    ncol = max(1, len(legend_handles))
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=ncol,
            frameon=True,
            fontsize=10,
        )

    plt.tight_layout()
    # Extra bottom margin: rotated x tick labels sit below axes; supxlabel sits in the strip below them.
    plt.subplots_adjust(
        top=0.90 if legend_handles else 0.94,
        bottom=0.16,
    )
    fig.supxlabel("(Batch Size, Sequence Length)", fontsize=11, y=-0.02)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Line plot saved to '{output_path}'")


def plot_bar_charts(combined_df, speedup_df, benchmarks, output_path: str = 'flex-able-bar.png'):
    """
    Generates and saves bar charts comparing implementations and showing speedup
    as text labels on the "Flashlight" bars.
    Rows are determined by 'group_size'.
    """
    print("Generating bar charts...")
    
    # Create a combined column for hue to separate (impl, nheads_headdim)
    combined_df["impl_config"] = combined_df["Implementation"] + "\n" + combined_df["nheads_headdim"]
    # Create a matching column in speedup_df to look up values
    speedup_df["impl_config"] = "Flashlight\n" + speedup_df["nheads_headdim"]

    # Sort for consistent plotting
    combined_df = combined_df.sort_values(by=["Benchmark", "batch_size", "seqlen", "nheads", "headdim"])
    speedup_df = speedup_df.sort_values(by=["Benchmark", "batch_size", "seqlen", "nheads", "headdim"])

    # Determine grid layout
    group_sizes = sorted(combined_df["group_size"].unique())
    n_rows = len(group_sizes)
    n_cols = len(benchmarks)

    fig, axes = plt.subplots(
        nrows=n_rows, 
        ncols=n_cols, 
        figsize=(5 * n_cols, 6 * n_rows), 
        squeeze=False # Ensure axes is always 2D
    )
    fig.set_size_inches(17, 7)
    plt.tight_layout()

    handles1, labels1 = [], [] # Initialize for legend

    for row_idx, group_size in enumerate(group_sizes):
        for col_idx, benchmark in enumerate(benchmarks):
            ax1 = axes[row_idx, col_idx]
            
            # Filter data for this specific subplot
            benchmark_data = combined_df[
                (combined_df["Benchmark"] == benchmark) &
                (combined_df["group_size"] == group_size)
            ]
            assert len(benchmark_data['nheads'].unique()) == 1, benchmark_data['nheads'].unique()
            assert len(benchmark_data['nheads_kv'].unique()) == 1, benchmark_data['nheads_kv'].unique()
            hq = benchmark_data['nheads'].unique()[0]
            hkv = benchmark_data['nheads_kv'].unique()[0]

            assert not benchmark_data.empty
            
            flashinfer_data = benchmark_data[benchmark_data["Implementation"] == "FlashInfer"]
            has_flashinfer = not flashinfer_data.empty
            n_groups = 3 if has_flashinfer else 2
            width = 0.8 / n_groups

            flashlight_data = benchmark_data[benchmark_data["Implementation"] == "Flashlight"]
            avg_times = []
            runs = 20
            categories = benchmark_data['batch_seqlen'].unique()
            ind = np.array(list(range(len(categories))))
            std = []
            for i in range(0, len(flashlight_data.FW_Time_ms), runs):
                avg_times += [sum(list(flashlight_data.FW_Time_ms)[i:i+runs])/runs]
                std += [float(np.std(list(flashlight_data.FW_Time_ms)[i:i+runs]))]
            flash_bar = ax1.bar(ind, avg_times, width, label="Flashlight")
            ax1.errorbar(ind, avg_times, yerr=std, fmt='none', ecolor='black',capsize=2)

            flex_cache_hit = benchmark_data[benchmark_data["Implementation"] == "Flex (Cache Hit)"]
            flex_cache_miss = benchmark_data[benchmark_data["Implementation"] == "Flex (Cache Miss)"]
            hit_avg_times = []
            mask_avg_times = []
            miss_times = []
            std = []
            speedup = []
            runs = 20
            categories = benchmark_data['batch_seqlen'].unique()

            has_miss = not flex_cache_miss.empty

            for i in range(0, len(flex_cache_hit.FW_Time_ms), runs):
                hit_avg_times += [sum(list(flex_cache_hit.FW_Time_ms)[i:i+runs])/runs]
                if has_miss:
                    miss_times += [sum(list(flex_cache_miss.FW_Time_ms)[i:i+runs])/runs]
                    mask_avg_times += [miss_times[-1] - hit_avg_times[-1]]
                    std += [float(np.std(list(flex_cache_miss.FW_Time_ms)[i:i+runs]))]
                else:
                    miss_times += [hit_avg_times[-1]]
                    mask_avg_times += [0.0]
                    std += [float(np.std(list(flex_cache_hit.FW_Time_ms)[i:i+runs]))]

            flex_kernel_bars = ax1.bar(ind+width, hit_avg_times, width, label="FlexAttention (Kernel)")
            flex_mask_bars = ax1.bar(ind+width, mask_avg_times, width, bottom=hit_avg_times, yerr=std, ecolor='black',capsize=2, label="FlexAttention (Block Mask)")

            speedups = []
            for flash,flex in zip(avg_times, miss_times):
                speedups += [flex/flash]
            
            speedup_text_kwargs = dict(
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                color="black",
            )
            speedup_text_offset = 0.25

            for speedup, kernel_bar, mask_bar in zip(speedups, flex_kernel_bars, flex_mask_bars):
                ax1.text(kernel_bar.get_x() + kernel_bar.get_width() / 2.0,
                         kernel_bar.get_height() + mask_bar.get_height() + speedup_text_offset, f'{speedup:.2f}x', 
                         **speedup_text_kwargs)

            # Optional: FlashInfer bars
            if has_flashinfer:
                fi_avg_times = []
                fi_std = []
                for i in range(0, len(flashinfer_data.FW_Time_ms), runs):
                    fi_avg_times += [sum(list(flashinfer_data.FW_Time_ms)[i:i+runs])/runs]
                    fi_std += [float(np.std(list(flashinfer_data.FW_Time_ms)[i:i+runs]))]
                ax1.bar(ind+2*width, fi_avg_times, width, label="FlashInfer", color="purple")
                ax1.errorbar(ind+2*width, fi_avg_times, yerr=fi_std, fmt='none', ecolor='black', capsize=2)

                # Add speedup labels (FlashInfer / Flashlight)
                for j, (flash_t, fi_t) in enumerate(zip(avg_times, fi_avg_times)):
                    fi_speedup = fi_t / flash_t
                    ax1.text(ind[j] + 2*width, fi_t + speedup_text_offset, f'{fi_speedup:.2f}x',
                             **speedup_text_kwargs)

            ax1.set_xticks(ind + width * (n_groups - 1) / 2)
            ax1.set_xticklabels(categories)
            
            # ax1.bar(ind + width, avg_times, label="Flex")
            # Plot FW_Time_ms bars
            #sns.barplot(
            #    data=benchmark_data,
            #    x="batch_seqlen",
            #    y="FW_Time_ms",
            #    hue="impl_config", # Use combined hue
            #    ax=ax1,style='impl_config',markers=True
            #)
            ax1.set_xlabel("(Batch Size, Sequence Length)")
            
            # Add row label
            if col_idx == 0:
                if hq == 16 and hkv == 16:
                    label = "MHA"
                else:
                    label = "GQA"
                ax1.set_ylabel(f"Time (ms) for {label}", fontsize=12)
            else:
                ax1.yaxis.get_label().set_visible(False)
            
            title = _format_benchmark_title(benchmark)
            ax1.set_title(f"{title}", fontsize=12)
            ax1.tick_params(
                axis='x',
                rotation=90,
                labelsize=12)
            
            # Get legend from first plot
            if row_idx == 0 and col_idx == 0:
                handles1, labels1 = ax1.get_legend_handles_labels()
            
            if ax1.get_legend():
                ax1.get_legend().remove()

            # --- Add Speedup Text Labels ---
            
            # Create a lookup map for speedup values
            bench_speedup_df = speedup_df[
                (speedup_df["Benchmark"] == benchmark) &
                (speedup_df["group_size"] == group_size)
            ]

            speedup_map = {}
            
            # Get x-tick labels
            xtick_labels = [label.get_text() for label in ax1.get_xticklabels()]
            
            # Get legend labels for containers
            h, l = ax1.get_legend_handles_labels()

            # # Iterate over bar containers
            # for container, label in zip(ax1.containers, l):
            #     # Only add labels to "Flashlight" bars
            #     if "Flashlight" in label:
            #         # Iterate over bars in this container
            #         for bar_index, bar in enumerate(container):
            #             # Find the corresponding x-label
            #             x_label = xtick_labels[bar_index]
                        
            #             # Find the speedup value
            #             key = (x_label, label)
            #             if key in speedup_map:
            #                 speedup_val = speedup_map[key]
            #                 height = bar.get_height()
                            
            #                 # Add text label
            #                 ax1.text(bar.get_x() + bar.get_width() / 2.0, 
            #                          height, 
            #                          f'{speedup_val:.2f}x', 
            #                          ha='center', 
            #                          va='bottom', 
            #                          fontsize=12, 
            #                         #  rotation=90,
            #                          color='black')

    # Create a single legend
    if handles1:
         for x,l in enumerate(labels1):
             labels1[x] = l.replace("\n(16, 64)", "").strip()
             if "Flex" in labels1[x] and "FlexAttention" not in labels1[x]:
                 labels1[x] = labels1[x].replace("Flex", "FlexAttention")
         fig.legend(handles1, labels1, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(labels1))


    #plt.suptitle("Flex-able Benchmarks (Bar Plot with Speedup)", y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15 if handles1 else 0.1) # Adjust bottom for legend

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Bar plot saved to '{output_path}'")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot benchmark results.")
    parser.add_argument(
        '--plot',
        type=str,
        choices=['line', 'bar'],
        default='bar',
        help="Type of plot to generate ('line' or 'bar'). Default is 'bar'."
    )
    parser.add_argument(
        '--flashlight',
        type=str,
        default='results/all.csv',
        help="CSV file containing all Flashlight results. Default is 'results/all.csv'."
    )
    parser.add_argument(
        '--flex',
        type=str,
        default='results/all_flex.csv',
        help="CSV file containing all FlexAttention with a mask cache results. Default is 'results/all_flex.csv'."
    )
    parser.add_argument(
        '--flex-no-cache',
        type=str,
        default='results/all_flexnocache.csv',
        help="CSV file containing all FlexAttention without a mask cache results. Default is 'results/all_flexnocache.csv'."
    )
    parser.add_argument(
        '--torch-compile',
        type=str,
        default='results/all_torchcompile.csv',
        help="CSV file containing all torch.compile results. Default is 'results/all_torchcompile.csv'."
    )
    parser.add_argument(
        '--flashinfer',
        type=str,
        default='results/all_flashinfer.csv',
        help="CSV file containing FlashInfer results. Default is 'results/all_flashinfer.csv'."
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default=None,
        metavar='PATH',
        help="Output PNG path. Default: flex-able-torch.png (line) or flex-able-bar.png (bar), "
        "relative to this script's directory unless PATH is absolute.",
    )
    args = parser.parse_args()

    # Generate the chosen plot
    if args.plot == 'line':
        combined_df, _, benchmarks = load_and_prepare_data(
            args.flashlight, args.flex, args.flex_no_cache,
            torchcompile_csv=args.torch_compile,
            flashinfer_csv=args.flashinfer,
        )
        plot_line_charts(combined_df, benchmarks, args.output or 'flex-able-torch.png')
    elif args.plot == 'bar':
        combined_df, speedup_df, benchmarks = load_and_prepare_data(
            args.flashlight, args.flex, args.flex_no_cache,
            torchcompile_csv=None,
            flashinfer_csv=args.flashinfer,
        )
        plot_bar_charts(combined_df, speedup_df, benchmarks, args.output or 'flex-able-bar.png')

if __name__ == "__main__":
    main()
