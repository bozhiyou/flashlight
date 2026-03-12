import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REFERENCE = "results/reference"

def load_and_prepare_data(diff_csvs, evo_csvs):
    """
    Loads and preprocesses benchmark data for DiffAttn and Evoformer.

    Each list of CSVs can contain results for one or both GPUs. If a GPU
    (A100/H100) is missing for a given benchmark, the corresponding reference
    CSV from ``results/reference/`` is loaded and used instead.
    """
    reference_sources = {
        "DiffAttn": {"A100": "diff_attn_a100.csv", "H100": "diff_attn_h100.csv"},
        "Evoformer": {"A100": "evo_attn_a100.csv", "H100": "evo_attn_h100.csv"},
    }
    input_csvs = {"DiffAttn": diff_csvs, "Evoformer": evo_csvs}


    all_dfs = []
    for benchmark_name, csv_paths in input_csvs.items():
        # Drop reference entries for GPUs that appear in input CSVs.
        missing_refs = reference_sources[benchmark_name].copy()

        for path in csv_paths:
            try:
                df = pd.read_csv(path)
                df["GPU"] = df["GPU"].astype(str).str.upper()
                df["Benchmark"] = benchmark_name
                all_dfs.append(df)

                for gpu in df["GPU"].unique():
                    missing_refs.pop(gpu, None)
            except FileNotFoundError:
                pass  # Ignore missing input CSVs; they will be filled from reference.

        # Load reference CSVs for any missing GPUs.
        for gpu, ref_file in missing_refs.items():
            ref_path = os.path.join(os.path.dirname(__file__), REFERENCE, ref_file)
            ref_df = pd.read_csv(ref_path)
            ref_df["GPU"] = ref_df["GPU"].astype(str).str.upper()
            ref_df["Benchmark"] = benchmark_name
            all_dfs.append(ref_df)

    raw_df = pd.concat(all_dfs, ignore_index=True)

    ours_df = raw_df[raw_df["Implementation"].str.contains("_compiled")].copy()
    ours_df["Implementation"] = "Flashlight"
    
    base_df = raw_df[~raw_df["Implementation"].str.contains("_compiled")].copy()
    base_df["Implementation"] = "torch"
    base_df["FW_Time_ms"] = base_df["FW_Time_ms"].replace(-1, float('nan'))

    combined_df = pd.concat([ours_df, base_df], ignore_index=True)
    combined_df = combined_df[["GPU", "Benchmark", "Implementation", "FW_Time_ms", "batch_size", "seqlen", "nheads", "headdim"]]

    speedup_df = combined_df.pivot_table(index=["GPU", "Benchmark", "batch_size", "seqlen", "nheads", "headdim"],
                                         columns="Implementation",
                                         values="FW_Time_ms").reset_index()
    speedup_df["Speedup"] = speedup_df["torch"] / speedup_df["Flashlight"]

    target_configs = ['DiffAttn-(32, 64)', 'DiffAttn-(16, 128)', 'Evoformer-(4, 64)', 'Evoformer-(4, 128)']

    for df in [combined_df, speedup_df]:
        df["nheads_headdim"] = df.apply(lambda row: f"({row['nheads']}, {row['headdim']})", axis=1)
        df["batch_seqlen"] = df.apply(lambda row: f"({row['batch_size']}, {row['seqlen']})", axis=1)
        df['benchmark_config'] = df['Benchmark'] + '-' + df['nheads_headdim']
    
    combined_df = combined_df[combined_df['benchmark_config'].isin(target_configs)]
    speedup_df = speedup_df[speedup_df['benchmark_config'].isin(target_configs)]
    
    # MODIFICATION: Reverse the sort order to put H100 first.
    gpus = sorted(combined_df["GPU"].unique(), reverse=True)

    return combined_df, speedup_df, target_configs, gpus

def plot_bar_charts(combined_df, speedup_df, benchmark_configs, gpus):
    """
    Generates a single row of bar charts with GPU subtitles and a legend in the last subplot.
    """
    print("Generating bar charts... 📊")
    
    sorted_configs = sorted(benchmark_configs, key=lambda x: ('Diff' not in x, x))
    num_configs = len(sorted_configs)
    total_plots = len(gpus) * num_configs
    
    fig, axes = plt.subplots(
        nrows=1, 
        ncols=total_plots, 
        figsize=(24, 5), 
        squeeze=False, 
        sharey=True
    )
    axes = axes.flatten()
    
    palette = {"Flashlight": sns.color_palette()[0], "torch": sns.color_palette()[3]}
    
    ax_idx = 0
    for gpu in gpus:
        for config in sorted_configs:
            ax = axes[ax_idx]
            
            plot_data = combined_df[(combined_df['GPU'] == gpu) & (combined_df['benchmark_config'] == config)]
            
            sns.barplot(
                data=plot_data, 
                x="batch_seqlen", 
                y="FW_Time_ms", 
                hue="Implementation", 
                ax=ax, 
                palette=palette
            )
            
            try:
                parts = config.split('-')
                name = parts[0]
                dims = parts[1].strip('()').split(',')
                nhead = dims[0].strip()
                headdim = dims[1].strip()
                new_title = f"{name}\n(nhead={nhead}, headdim={headdim})"
            except (IndexError, ValueError):
                new_title = config
            
            ax.set_title(new_title, fontsize=14)
            ax.tick_params(axis='x', labelsize=12, rotation=90)
            ax.set_xlabel(None)
            
            ax.set_ylabel("Time (ms)" if ax_idx == 0 else "", size=14)

            speedup_data_filtered = speedup_df[(speedup_df['GPU'] == gpu) & (speedup_df['benchmark_config'] == config)]
            speedup_map = speedup_data_filtered.set_index('batch_seqlen')['Speedup'].to_dict()

            xtick_labels = [label.get_text() for label in ax.get_xticklabels()]
            
            handles, labels = ax.get_legend_handles_labels()
            torch_container = None
            if ax.containers:
                for container, label in zip(ax.containers, labels):
                    if label == 'torch':
                        torch_container = container
                        break

            if torch_container:
                for bar_index, bar in enumerate(torch_container):
                    x_label = xtick_labels[bar_index]
                    if x_label in speedup_map and pd.notna(speedup_map[x_label]):
                        speedup_val = speedup_map[x_label]
                        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height()+0.2, f'{speedup_val:.2f}x',
                                ha='center', va='bottom', fontsize=14, color='black', rotation=90)
            
            if ax_idx == total_plots - 1:
                ax.legend(handles=handles, labels=labels, title=None, loc='upper right')
            else:
                if ax.get_legend():
                    ax.get_legend().remove()
            
            ax_idx += 1

    fig.supxlabel("(Batch Size, Sequence Length)", y=0.12, fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.32, top=0.80)

    # Add GPU subtitles
    pos1 = axes[0].get_position()
    pos2 = axes[num_configs - 1].get_position()
    mid_x1 = (pos1.x0 + pos2.x1) / 2
    
    pos3 = axes[num_configs].get_position()
    pos4 = axes[total_plots - 1].get_position()
    mid_x2 = (pos3.x0 + pos4.x1) / 2
    
    # The gpus list is now [H100, A100], so this will work correctly
    fig.text(mid_x1, 0.95, gpus[0], ha='center', va='center', fontsize=16, fontweight='bold')
    fig.text(mid_x2, 0.95, gpus[1], ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Add a vertical line between the two GPU groups
    last_left_gpu_pos = axes[num_configs - 1].get_position()
    first_right_gpu_pos = axes[num_configs].get_position()
    
    line_x_pos = (last_left_gpu_pos.x1 + first_right_gpu_pos.x0) / 2
    
    separator_line = Line2D(
        [line_x_pos, line_x_pos], [0.15, 0.92],
        transform=fig.transFigure,
        color='black',
        linestyle='--',
        linewidth=1.5
    )
    fig.add_artist(separator_line)
    
    output_filename = 'flex-unable-bar.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Bar plot saved to '{output_filename}'")

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results for attention mechanisms across GPUs.")
    parser.add_argument('--plot', type=str, choices=['bar'], default='bar', help="Plot type. Default is 'bar'.")
    parser.add_argument(
        '--diff-attn',
        type=str,
        nargs='+',
        default=[os.path.join(REFERENCE, "diff_attn_a100.csv"),
                 os.path.join(REFERENCE, "diff_attn_h100.csv")],
        help=(
            "One or more CSVs for DiffAttn (with a GPU column). "
            "If either A100 or H100 is missing, the corresponding reference "
            "CSV from results/reference/ is used."
        ),
    )
    parser.add_argument(
        '--evo-attn',
        type=str,
        nargs='+',
        default=[os.path.join(REFERENCE, "evo_attn_a100.csv"),
                 os.path.join(REFERENCE, "evo_attn_h100.csv")],
        help=(
            "One or more CSVs for Evoformer (with a GPU column). "
            "If either A100 or H100 is missing, the corresponding reference "
            "CSV from results/reference/ is used."
        ),
    )
    args = parser.parse_args()

    combined_df, speedup_df, benchmark_configs, gpus = load_and_prepare_data(
        args.diff_attn,
        args.evo_attn,
    )

    plot_bar_charts(combined_df, speedup_df, benchmark_configs, gpus)

if __name__ == "__main__":
    main()