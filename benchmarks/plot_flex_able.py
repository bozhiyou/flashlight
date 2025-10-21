import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

benchmark_names = {
    "full": "vanilla (Vaswani et al 2017)",
    "full_with_alibi": "ALiBi",
    "full_with_softcap": "softcap",
    "full_with_sliding_window": "sliding window (wsz=256)",
    "full_with_prefix_lm": "prefix LM (causal prefix sz=256)",
}

ours_df = pd.read_csv("all.csv")
flex_df = pd.read_csv("all_flex.csv")

ours_df = ours_df.rename(columns={"Implementation": "Benchmark"})
ours_df["Implementation"] = "Ours"
# FIXME Drop rows where 'Benchmark' is 'full_with_causal'
ours_df = ours_df[ours_df['Benchmark'] != 'full_with_causal']
ours_df['Benchmark'] = ours_df['Benchmark'].replace(benchmark_names)

flex_df = flex_df.rename(columns={"Implementation": "Benchmark"})
flex_df["Benchmark"] = flex_df["Benchmark"].str.replace("flex_", "")
flex_df["Implementation"] = "Flex"
flex_df["FW_Time_ms"] = flex_df["FW_Time_ms"].replace(-1, float('nan'))
# FIXME Drop rows where 'Benchmark' is 'full_with_causal'
flex_df = flex_df[flex_df['Benchmark'] != 'full_with_causal']
flex_df['Benchmark'] = flex_df['Benchmark'].replace(benchmark_names)

combined_df = pd.concat([ours_df, flex_df], ignore_index=True)

# Calculate speedup
speedup_df = combined_df.pivot_table(index=["Benchmark", "batch_size", "seqlen", "nheads", "headdim", "causal", "dropout_p"],
                                     columns="Implementation",
                                     values="FW_Time_ms").reset_index()
# result NaN if "Flex" is NaN
speedup_df["Speedup"] = speedup_df["Flex"] / speedup_df["Ours"]

# Combine the plots with a secondary y-axis for speedup
# This is a bit more complex with FacetGrid, so we'll iterate through the benchmarks
benchmarks = combined_df["Benchmark"].unique()
# fig, axes = plt.subplots(nrows=(len(benchmarks) + 2) // 3, ncols=3, figsize=(15, 5 * ((len(benchmarks) + 2) // 3)))
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
axes = axes.flatten()
# Prepare data for plotting
combined_df["nheads_headdim"] = combined_df.apply(lambda row: f"({row['nheads']}, {row['headdim']})", axis=1)
combined_df["batch_seqlen"] = combined_df.apply(lambda row: f"({row['batch_size']}, {row['seqlen']})", axis=1)
speedup_df["batch_seqlen"] = speedup_df.apply(lambda row: f"({row['batch_size']}, {row['seqlen']})", axis=1)
speedup_df["nheads_headdim"] = speedup_df.apply(lambda row: f"({row['nheads']}, {row['headdim']})", axis=1)

ax2_shared = None # Initialize a shared y-axis for speedup
legend_handles, legend_labels = [], []
for i, benchmark in enumerate(benchmarks):
    ax1 = axes[i]
    
    # Plot FW_Time_ms on the left y-axis
    sns.lineplot(
        data=combined_df[combined_df["Benchmark"] == benchmark],
        x="batch_seqlen",
        y="FW_Time_ms",
        hue="Implementation",
        style="nheads_headdim",
        markers=True,
        ax=ax1,
        palette={"Ours": "blue", "Flex": "green"},
        # linestyle="-",
        dashes={"(32, 64)": (3, 2), "(16, 128)": ""},
    )
    ax1.set_xlabel("(Batch Size, Sequence Length)")
    ax1.set_ylabel("Time (ms)")
    ax1.set_yscale("log", base=2) # Set y-axis to log2 scale
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.yaxis.set_minor_formatter(ScalarFormatter())
    ax1.set_title(benchmark)

    # Create a secondary y-axis for Speedup, sharing the y-axis with the first one
    if ax2_shared is None:
        ax2_shared = ax1.twinx()
    ax2 = ax1.twinx() if i == 0 else ax1.twinx()
    ax2.sharey(ax2_shared)

    sns.lineplot(
        data=speedup_df[speedup_df["Benchmark"] == benchmark],
        x="batch_seqlen",
        y="Speedup",
        style="nheads_headdim",
        # linestyle=":",
        # color="red",
        dashes={"(32, 64)": (3, 2), "(16, 128)": ()},
        ax=ax2)
    ax2.set_ylabel("Speedup")

    if i == 0:
        speedup_handle = Line2D([], [], linestyle='-', marker='', label='Speedup (right axis)')
        handles1, labels1 = ax1.get_legend_handles_labels()
        legend_handles = handles1[1:3] + [speedup_handle] + handles1[3:]
        legend_labels = labels1[1:3] + [speedup_handle.get_label()] + ["(#heads, headdim)"] + labels1[4:]

    # Remove individual legends
    ax1.get_legend().remove()
    ax2.get_legend().remove()

# Create a single legend for all subplots
fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, .05), ncol=2)

plt.suptitle("Flex-able Benchmarks", y=1.02)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

output_filename = 'flex-able.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved to '{output_filename}'")