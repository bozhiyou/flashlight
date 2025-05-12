import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # Import Patch for legend
import seaborn as sns
import numpy as np
import os
import math

# --- Configuration ---
# Use the uploaded file path directly
input_csv_path = # input path
output_plot_path = # output path
# Define colors and markers for plotting consistency
color_map = {
    'ours_time': 'skyblue',     # Renamed from base_time
    'flex_time': 'deepskyblue',
    'ours_tflops': 'lightcoral', # Renamed from base_tflops
    'flex_tflops': 'firebrick'
}
marker_map = {
    'ours_time': 'o',           # Renamed from base_time
    'flex_time': 'o',
}

# Check if the file exists
if not os.path.exists(input_csv_path):
    raise FileNotFoundError(f"Error: Input file not found at {input_csv_path}")

# Load the dataframe
df = pd.read_csv(input_csv_path, skipinitialspace=True)

# Data Cleaning/Validation
required_columns = ['Implementation', 'batch_size', 'seqlen', 'headdim', 'FW_Time_ms', 'FW_TFLOPS']
if not all(col in df.columns for col in required_columns):
    missing = [col for col in required_columns if col not in df.columns]
    raise ValueError(f"Error: Missing required columns: {missing}")

# Convert relevant columns to numeric first, coercing errors
for col in ['batch_size', 'seqlen', 'headdim', 'FW_Time_ms', 'FW_TFLOPS']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Identify original -1 values before converting them to NaN for plotting
df['is_oom_time'] = df['FW_Time_ms'] == -1
df['is_oom_tflops'] = df['FW_TFLOPS'] == -1
# Combine OOM flags for easier checking later
df['is_oom'] = df['is_oom_time'] | df['is_oom_tflops']

# Replace -1 with NaN for plotting purposes AFTER checking
df['FW_Time_ms'] = df['FW_Time_ms'].replace(-1, np.nan)
df['FW_TFLOPS'] = df['FW_TFLOPS'].replace(-1, np.nan)

# Drop rows where essential config columns are NaN after coercion
df.dropna(subset=['Implementation', 'batch_size', 'seqlen', 'headdim'], inplace=True)
# Drop rows only if BOTH time and TFLOPS are NaN (often due to failed runs not marked as -1)
# Keep rows where only one is NaN if the other is valid (or was -1/OOM)
# Also drop rows where config generation might fail
df.dropna(subset=['FW_Time_ms', 'FW_TFLOPS', 'batch_size', 'seqlen'], how='all', inplace=True)


# Convert config columns to integers AFTER potential NaNs are handled/dropped
for col in ['batch_size', 'seqlen', 'headdim']:
    df[col] = df[col].astype('Int64') # Use Int64 to handle potential remaining NaNs if any

# Drop rows where key identifiers are still somehow NaN after conversion attempts
df.dropna(subset=['Implementation', 'batch_size', 'seqlen', 'headdim'], inplace=True)


# Create a unique configuration string for the X-axis
# Ensure types are standard int before string formatting if Int64 caused issues downstream (unlikely here)
df['config_str'] = df.apply(lambda row: f"({int(row['batch_size'])}, {int(row['seqlen'])})", axis=1)

# --- Handle Potential Duplicates by Averaging ---
# Before identifying pairs, group by all identifying columns and average metrics
# This ensures that subsequent filtering operates on unique configurations per implementation
aggregation_funcs = {
    'FW_Time_ms': 'mean',
    'FW_TFLOPS': 'mean',
    'is_oom': 'any' # If any run for this config was OOM, mark it as OOM
}
df = df.groupby(['Implementation', 'batch_size', 'seqlen', 'headdim', 'config_str'], as_index=False).agg(aggregation_funcs)


# --- Identify Implementation Pairs ---
implementations = df['Implementation'].unique()
# Find base implementations (those NOT starting with 'flex_')
base_implementations = sorted([impl for impl in implementations if not impl.startswith('flex_')])
pairs = []
for base_impl_name in base_implementations: # This is the name like 'flash'
    flex_impl_name = f"flex_{base_impl_name}"
    if flex_impl_name in implementations:
        # Store both the original base name and the flex name
        pairs.append({'base_name': base_impl_name, 'flex_name': flex_impl_name})
    else:
        print(f"Warning: No corresponding 'flex_' implementation found for '{base_impl_name}'. Skipping.")

if not pairs:
    raise ValueError("Error: No implementation pairs (e.g., 'flash' and 'flex_flash') found in the data.")

# Get all implementation names involved in the valid pairs
paired_impl_names = [p['base_name'] for p in pairs] + [p['flex_name'] for p in pairs]
# Ensure headdim is treated as numeric for sorting, handling potential NaNs introduced by grouping if keys were missing
df['headdim'] = pd.to_numeric(df['headdim'], errors='coerce')
df.dropna(subset=['headdim'], inplace=True)
df['headdim'] = df['headdim'].astype(int) # Convert back to int after potential coercion/dropna

unique_head_dims = sorted(df[df['Implementation'].isin(paired_impl_names)]['headdim'].unique())
# unique_head_dims = [hd for hd in unique_head_dims if pd.notna(hd)] # Already handled by dropna


if not unique_head_dims:
    raise ValueError("Error: No valid data found for the identified implementation pairs after cleaning and aggregation.")

# --- Plotting Setup ---
n_pairs = len(pairs)
n_head_dims = len(unique_head_dims)
n_plots = n_pairs * n_head_dims
n_cols = n_pairs
n_rows = n_head_dims

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 5.5), squeeze=False) # Slightly taller
axes_flat = axes.flatten()
plot_idx = 0

# --- Generate Plots for Each Pair and Head Dimension ---
for r_idx, headdim in enumerate(unique_head_dims):
    for c_idx, pair in enumerate(pairs):
        # Use the stored names
        base_impl = pair['base_name'] # e.g., 'flash'
        flex_impl = pair['flex_name'] # e.g., 'flex_flash'
        ours_label = 'ours' # The new label for the base implementation

        current_plot_abs_idx = r_idx * n_cols + c_idx
        if current_plot_abs_idx >= len(axes_flat):
            print("Warning: More plots needed than grid allows. Some combinations might be skipped.")
            break

        ax = axes[r_idx, c_idx] # Use 2D indexing

        # Filter *aggregated* data for the current pair and head dimension
        pair_df = df[
            df['Implementation'].isin([base_impl, flex_impl]) &
            (df['headdim'] == headdim)
        ].copy()

        # Get ALL unique configs for this head dimension across ALL implementations from aggregated df
        all_configs_for_headdim_df = df[df['headdim'] == headdim].copy()
         # Sort by batch_size, seqlen extracted from config_str to ensure correct order
        all_configs_for_headdim_df[['temp_batch', 'temp_seqlen']] = all_configs_for_headdim_df['config_str'].str.strip('()').str.split(',', expand=True).astype(int)
        all_configs_for_headdim_df = all_configs_for_headdim_df.sort_values(by=['temp_batch', 'temp_seqlen'])
        config_order = all_configs_for_headdim_df['config_str'].unique().tolist()

        if pair_df.empty and not any(c in pair_df['config_str'].unique() for c in config_order):
             ax.set_title(f"{ours_label} ({base_impl}) vs {flex_impl}\nhead_dim={headdim}\n(No Data)")
             ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
             plot_idx += 1 # Increment the flat index counter
             continue

        # Create a mapping for renaming implementations in the plotting dataframe
        rename_map = {base_impl: ours_label, flex_impl: flex_impl}
        pair_df['Plot_Implementation'] = pair_df['Implementation'].map(rename_map)

        # Use Categorical type AFTER filtering pair_df, ensuring all possible configs are categories
        pair_df['config_str'] = pd.Categorical(pair_df['config_str'], categories=config_order, ordered=True)
        pair_df.dropna(subset=['config_str'], inplace=True) # Drop if config wasn't in order (shouldn't happen now)
        pair_df = pair_df.sort_values('config_str')

        # --- Prepare data for merge ---
        # Data is already aggregated, just filter and set index
        base_data_subset = pair_df[pair_df['Implementation'] == base_impl]
        flex_data_subset = pair_df[pair_df['Implementation'] == flex_impl]

        # Check for duplicates *after* filtering, before setting index (should not happen after grouping)
        if base_data_subset['config_str'].duplicated().any():
             print(f"WARNING: Duplicates found in base data for {base_impl}, {headdim} AFTER grouping. Check logic.")
             # Optional: Drop duplicates again as a fallback
             # base_data_subset = base_data_subset.drop_duplicates(subset=['config_str'], keep='first')
        if flex_data_subset['config_str'].duplicated().any():
             print(f"WARNING: Duplicates found in flex data for {flex_impl}, {headdim} AFTER grouping. Check logic.")
             # flex_data_subset = flex_data_subset.drop_duplicates(subset=['config_str'], keep='first')


        base_data = base_data_subset[['config_str', 'FW_Time_ms', 'FW_TFLOPS', 'is_oom']].set_index('config_str')
        flex_data = flex_data_subset[['config_str', 'FW_Time_ms', 'FW_TFLOPS', 'is_oom']].set_index('config_str')

        # Merge should now work with unique indices
        merged_data = pd.merge(base_data, flex_data, left_index=True, right_index=True, how='outer', suffixes=('_base', '_flex'))

        # Calculate ratio as ours (base) / flex
        merged_data['TFLOPS_ratio'] = (merged_data['FW_TFLOPS_base'] / merged_data['FW_TFLOPS_flex']).replace([np.inf, -np.inf], np.nan).fillna(np.nan)

        # Reindex merged_data based on config_order - THIS should now work
        merged_data = merged_data.reindex(config_order)
        # Fill NaN in OOM flags with False for easier checking AFTER reindexing
        merged_data['is_oom_base'] = merged_data['is_oom_base'].fillna(False)
        merged_data['is_oom_flex'] = merged_data['is_oom_flex'].fillna(False)

        # === Plotting ===
        ax.set_title(f"{ours_label} ({base_impl}) vs {flex_impl} (headdim={headdim})")
        x_indices = range(len(config_order))
        ax.set_xticks(x_indices)
        ax.set_xticklabels(config_order)

        # --- Secondary Y-axis: TFLOPs (Bars) ---
        ax2 = ax.twinx()
        current_hue_order = [ours_label, flex_impl]
        bar_plot = sns.barplot(data=pair_df, x='config_str', y='FW_TFLOPS', hue='Plot_Implementation',
                                palette={ours_label: color_map['ours_tflops'], flex_impl: color_map['flex_tflops']},
                                hue_order=current_hue_order,
                                ax=ax2, legend=False, zorder=1)
        ax2.set_ylabel("TFLOPs", color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.grid(False)

        # --- Primary Y-axis: Time (ms) ---
        config_to_x = {config: i for i, config in enumerate(config_order)}
        pair_df['x_coord'] = pair_df['config_str'].map(config_to_x)
        line_plot = sns.lineplot(data=pair_df.dropna(subset=['x_coord']), x='x_coord', y='FW_Time_ms', hue='Plot_Implementation', # Drop rows where mapping failed
                        palette={ours_label: color_map['ours_time'], flex_impl: color_map['flex_time']},
                        hue_order=current_hue_order,
                        marker='o',
                        markers={ours_label: marker_map['ours_time'], flex_impl: marker_map['flex_time']},
                        ax=ax, legend=False, zorder=1000)
        ax.set_ylabel("Time (ms)", color='black')
        ax.tick_params(axis='y', labelcolor='black')
        ax.set_xlabel("Config (batch_size, seqlen)")
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

        if r_idx == n_rows - 1:
             ax.tick_params(axis='x', rotation=45, labelsize=8, which='major')
             ax.set_xticklabels(config_order, rotation=45, ha='right')
        else:
             ax.tick_params(axis='x', labelbottom=False)
             ax.set_xlabel("")

        ax.set_xlim(-0.5, len(config_order) - 0.5)
        ax2.set_xlim(-0.5, len(config_order) - 0.5)

        # --- Annotate TFLOPS Ratio (ours/flex) ---
        if len(ax2.containers) > 0:
            try:
                ours_bars_container = ax2.containers[0]
                ours_tflops_values = merged_data['FW_TFLOPS_base'].values
                ratios = merged_data['TFLOPS_ratio'].values
                num_bars_in_container = len(ours_bars_container)

                if num_bars_in_container > 0: # Proceed only if bars exist
                    bar_x_centers = [bar.get_x() + bar.get_width() / 2 for bar in ours_bars_container]
                    # Estimate indices, handle potential floating point inaccuracies
                    est_indices = np.round(np.array(bar_x_centers) - (-0.2)) # Base bars are often at x - 0.2 relative to tick center
                    container_indices = np.clip(est_indices, 0, len(config_order) - 1).astype(int)

                    # Verify mapping if lengths differ significantly
                    if num_bars_in_container != len(config_order) and len(container_indices)>0:
                         # Check if the mapping seems reasonable, e.g. are indices unique?
                         if len(set(container_indices)) != len(container_indices):
                              print(f"Warning: Bar mapping to config index is ambiguous for plot {current_plot_abs_idx}. Annotations might be misplaced.")
                              # Fallback or skip annotation might be needed here depending on severity

                    plotted_indices = set() # Track plotted indices to avoid duplicate annotations if mapping is bad
                    for bar_idx, bar in enumerate(ours_bars_container):
                         if bar_idx >= len(container_indices): continue # Index out of bounds
                         config_idx = container_indices[bar_idx]
                         if config_idx in plotted_indices: continue # Skip if already annotated
                         plotted_indices.add(config_idx)

                         if config_idx < len(ratios) and pd.notna(ratios[config_idx]) and pd.notna(ours_tflops_values[config_idx]) and ours_tflops_values[config_idx] > 0:
                              height = bar.get_height()
                              if pd.notna(height) and height > 0:
                                   ax2.annotate(f"{ratios[config_idx]:.2f}x",
                                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                                xytext=(0, 3), textcoords="offset points",
                                                ha='center', va='bottom', fontsize=7, color='black',
                                                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec='none'),
                                                zorder=20)
            except IndexError:
                 print(f"Warning: IndexError during TFLOPS ratio annotation for plot {current_plot_abs_idx} ({ours_label}, headdim={headdim}). Container count: {len(ax2.containers)}")
            except Exception as e:
                 print(f"Error during ratio annotation for plot {current_plot_abs_idx}: {e}")


        # --- Annotate OOM ---
        for idx, config_str in enumerate(config_order):
             # Use the reindexed merged_data which has flags for all configs
            if config_str in merged_data.index and \
               (merged_data.loc[config_str, 'is_oom_base'] or merged_data.loc[config_str, 'is_oom_flex']):
                y_min, y_max = ax.get_ylim()
                # Ensure y_max > y_min to avoid division by zero or weirdness if plot is empty/flat
                if y_max > y_min:
                    y_pos = y_min + 0.05 * (y_max - y_min)
                else: # Fallback if ylim is invalid
                    y_pos = 0.05
                ax.text(idx, y_pos, 'OOM',
                        horizontalalignment='center', verticalalignment='bottom',
                        fontsize=7, color='red', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7, ec='none'),
                        zorder=25)

        # --- Create Legend (using 'ours' label) ---
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=color_map['ours_time'], lw=2, marker=marker_map['ours_time'], label=f'{ours_label} Time'),
            Line2D([0], [0], color=color_map['flex_time'], lw=2, marker=marker_map['flex_time'], label=f'{flex_impl} Time'),
            mpatches.Patch(color=color_map['ours_tflops'], label=f'{ours_label} TFLOPs'),
            mpatches.Patch(color=color_map['flex_tflops'], label=f'{flex_impl} TFLOPs')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize='small')
        if ax2.get_legend():
            ax2.get_legend().remove()

        plot_idx += 1

    if current_plot_abs_idx >= len(axes_flat) -1:
            break

# --- Final Touches ---
for i in range(plot_idx, len(axes_flat)):
    axes_flat[i].set_visible(False)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# --- Save and Show ---
plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_plot_path}")
plt.show()