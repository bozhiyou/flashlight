import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np

def load_and_prepare_data(ours_csv, flex_csv, flexnocache_csv, torchcompile_csv):
    """
    Loads and preprocesses the benchmark data from CSV files.
    """
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    ours_csv = os.path.join(bench_dir, ours_csv) if not os.path.isabs(ours_csv) else ours_csv
    flex_csv = os.path.join(bench_dir, flex_csv) if not os.path.isabs(flex_csv) else flex_csv
    flexnocache_csv = os.path.join(bench_dir, flexnocache_csv) if not os.path.isabs(flexnocache_csv) else flexnocache_csv
    torchcompile_csv = os.path.join(bench_dir, torchcompile_csv) if not os.path.isabs(torchcompile_csv) else torchcompile_csv
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

    try:
        torchcompile_df = pd.read_csv(torchcompile_csv)
    except FileNotFoundError as e:
        print(f"INFO: torch.compile data not found; skipping.")
        torchcompile_df = pd.DataFrame(columns=ours_df.columns)

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

    if not torchcompile_df.empty:
        torchcompile_df = torchcompile_df.rename(columns={"Implementation": "Benchmark"})
        torchcompile_df["Implementation"] = "torch.compile"
        dfs_to_concat = [ours_df, flex_df, flexnocache_df, torchcompile_df]
    else:
        dfs_to_concat = [ours_df, flex_df, flexnocache_df]

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


def plot_line_charts(combined_df, speedup_df, benchmarks):
    """
    Generates and saves the line charts as in the original script.
    """
    print("Generating line charts...")
    # Combine the plots with a secondary y-axis for speedup
    # This is a bit more complex with FacetGrid, so we'll iterate through the benchmarks
    # fig, axes = plt.subplots(nrows=(len(benchmarks) + 2) // 3, ncols=3, figsize=(15, 5 * ((len(benchmarks) + 2) // 3)))
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
    axes = axes.flatten()

    dashes_nhead_headdim = {"(32, 64)": (3, 2), "(16, 128)": (1, 1), "(16, 64)": ()}

    ax2_shared = None # Initialize a shared y-axis for speedup
    legend_handles, legend_labels = [], []
    for i, benchmark in enumerate(benchmarks):
        ax1 = axes[i]

        benchmark_data = combined_df[combined_df["Benchmark"] == benchmark]
        if benchmark_data.empty:
            ax1.set_title(benchmark)
            ax1.text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, color='gray')
            ax1.set_xticks([])
            ax1.set_yticks([])
            continue

        # Plot FW_Time_ms on the left y-axis
        sns.lineplot(
            data=benchmark_data,
            x="batch_seqlen",
            y="FW_Time_ms",
            hue="Implementation",
            style="nheads_headdim",
            markers=True,
            ax=ax1,
            palette={"Flashlight": "blue", "Flex (Cache Hit)": "green", "Flex (Cache Miss)": "orange", "torch.compile": "red"},
            dashes=dashes_nhead_headdim,
        )
        ax1.set_xlabel("(Batch Size, Sequence Length)")
        ax1.set_ylabel("Time (ms)")
        ax1.set_yscale("log", base=2) # Set y-axis to log2 scale
        ax1.yaxis.set_major_formatter(ScalarFormatter())
        ax1.yaxis.set_minor_formatter(ScalarFormatter())
        ax1.set_title(benchmark)
        # ax1.tick_params(axis='x', rotation=30)


        # Create a secondary y-axis for Speedup, sharing the y-axis with the first one
        if ax2_shared is None:
            ax2_shared = ax1.twinx()
        ax2 = ax1.twinx() if i == 0 else ax1.twinx()
        ax2.sharey(ax2_shared)

        speedup_data = speedup_df[speedup_df["Benchmark"] == benchmark]
        if not speedup_data.empty:
            sns.lineplot(
                data=speedup_data,
                x="batch_seqlen",
                y="Speedup",
                style="nheads_headdim",
                dashes=dashes_nhead_headdim,
                ax=ax2,
                # color="red",
                # legend=False # No separate legend for this line
            )
        ax2.set_ylabel("Speedup")

        # Only create the custom legend once
        if i == 0:
            speedup_handle = Line2D([], [], linestyle='-', marker='', label='Speedup (right axis)')
            handles1, labels1 = ax1.get_legend_handles_labels()
            legend_handles = handles1[1:3] + [speedup_handle] + handles1[3:]
            legend_labels = labels1[1:3] + [speedup_handle.get_label()] + ["(#heads, headdim)"] + labels1[4:]

        # Remove individual legends
        ax1.get_legend().remove()
        ax2.get_legend().remove()

    # Hide unused axes
    for j in range(len(benchmarks), len(axes)):
        axes[j].set_visible(False)

    assert legend_handles # Only if we actually plotted something
    # Create a single legend for all subplots
    fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, .05), ncol=2)

    plt.suptitle("Flex-able Benchmarks (Line Plot)", y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20 if legend_handles else 0.1) # Make more room for legend

    output_filename = 'flex-able.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Line plot saved to '{output_filename}'")


def plot_bar_charts(combined_df, speedup_df, benchmarks):
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
            
            flashlight_data = benchmark_data[benchmark_data["Implementation"] == "Flashlight"]
            avg_times = []
            runs = 20
            categories = benchmark_data['batch_seqlen'].unique()
            ind = np.array(list(range(len(categories))))
            width = 0.3
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
            # ax1.errorbar(ind+width, mask_avg_times, bottom=hit_avg_times, yerr=std, fmt='none', ecolor='black',capsize=2)

            speedups = []
            for flash,flex in zip(avg_times, miss_times):
                speedups += [flex/flash]
            
            for speedup, kernel_bar, mask_bar in zip(speedups, flex_kernel_bars, flex_mask_bars):
                ax1.text(kernel_bar.get_x() + kernel_bar.get_width() / 2.0, 
                         kernel_bar.get_height() + mask_bar.get_height()+0.2, f'{speedup:.2f}x',
                         ha='center', 
                         va='bottom', 
                         fontsize=12, 
                         rotation=90,
                         color='black')
                                 
            ax1.set_xticks(ind)
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
            
            title = benchmark[0].upper() + benchmark[1:].lower()
            title = " ".join([t[0].upper() + t[1:].lower() for t in title.split(' ')])
            if "Prefix Lm" == title:
                title = "PrefixLM"
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
         fig.legend(handles1, labels1, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5)


    #plt.suptitle("Flex-able Benchmarks (Bar Plot with Speedup)", y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15 if handles1 else 0.1) # Adjust bottom for legend

    output_filename = 'flex-able-bar.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Bar plot saved to '{output_filename}'")

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
    args = parser.parse_args()

    # Load and process data
    combined_df, speedup_df, benchmarks = load_and_prepare_data(args.flashlight, args.flex, args.flex_no_cache, args.torch_compile)

    # Generate the chosen plot
    if args.plot == 'line':
        plot_line_charts(combined_df, speedup_df, benchmarks)
    elif args.plot == 'bar':
        plot_bar_charts(combined_df, speedup_df, benchmarks)

if __name__ == "__main__":
    main()
