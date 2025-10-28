import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


diffattn_df = pd.read_csv("diff_attn.csv")
ipa_df = pd.read_csv("ipa.csv")
ours_df = pd.concat(df[df["Implementation"].str.endswith("_compiled")] for df in (diffattn_df, ipa_df))
base_df = pd.concat(df[~df["Implementation"].str.endswith("_compiled")] for df in (diffattn_df, ipa_df))

ours_df = ours_df.rename(columns={"Implementation": "Benchmark"})
ours_df["Benchmark"] = ours_df["Benchmark"].str.replace("_compiled", "")
ours_df["Implementation"] = "Ours"
ours_df = ours_df[["Implementation", "FW_Time_ms", "Benchmark", "batch_size", "seqlen", "nheads", "headdim"]]

base_df = base_df.rename(columns={"Implementation": "Benchmark"})
base_df["Implementation"] = "Base"
base_df["FW_Time_ms"] = base_df["FW_Time_ms"].replace(-1, float('nan'))
base_df = base_df[["Implementation", "FW_Time_ms", "Benchmark", "batch_size", "seqlen", "nheads", "headdim"]]

combined_df = pd.concat([ours_df, base_df], ignore_index=True)

# Calculate speedup
speedup_df = combined_df.pivot_table(index=["Benchmark", "batch_size", "seqlen", "nheads", "headdim"],
                                     columns="Implementation",
                                     values="FW_Time_ms").reset_index()
# result NaN if "Base" is NaN
speedup_df["Speedup"] = speedup_df["Base"] / speedup_df["Ours"]

# Combine the plots with a secondary y-axis for speedup
# This is a bit more complex with FacetGrid, so we'll iterate through the benchmarks
benchmarks = combined_df["Benchmark"].unique()
# fig, axes = plt.subplots(nrows=(len(benchmarks) + 2) // 3, ncols=3, figsize=(15, 5 * ((len(benchmarks) + 2) // 3)))
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
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
        data=combined_df[combined_df["Benchmark"] == benchmark].sort_values(by=['seqlen', 'batch_size'], ascending=[True, True]),
        x="batch_seqlen",
        y="FW_Time_ms",
        hue="Implementation",
        style="nheads_headdim",
        markers=True,
        ax=ax1,
        palette={"Ours": "blue", "Base": "green"},
        # linestyle="-",
        dashes={"(32, 64)": (3, 2), "(16, 128)": (), '(4, 32)': (), '(4, 64)': (), '(4, 128)': ()},
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
        dashes={"(32, 64)": (3, 2), "(16, 128)": (), '(4, 32)': (), '(4, 64)': (), '(4, 128)': ()},
        ax=ax2)
    ax2.set_ylabel("Speedup")

    handles1, labels1 = ax1.get_legend_handles_labels()
    if i == 0:
        speedup_handle = Line2D([], [], linestyle='-', marker='', label='Speedup (right axis)')
        legend_handles = handles1[1:3] + [speedup_handle]
        legend_labels = labels1[1:3] + [speedup_handle.get_label()]

    # Remove individual legends
    ax1.get_legend().remove()
    ax1.legend(handles1[3:], ["(#heads, headdim)"] + labels1[4:], title="")
    ax2.get_legend().remove()

# Create a single legend for all subplots
fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, .05), ncol=3)

plt.suptitle("Un-Flex-able Benchmarks", y=1.02)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

output_filename = 'flex-unable.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved to '{output_filename}'")