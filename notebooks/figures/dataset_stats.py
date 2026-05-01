#%%
import os, sys
sys.path.append(os.path.dirname('../../'))

import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm

from notebooks.figures.helpers import save_figure
from utils.setup import setup_environment
os.chdir('/data/Zeitler/code/SIDE')
setup_environment(skip_cuda=True)

#%%
# --- Configuration ---
BASE_DATA_DIR = Path("/data/Zeitler/SIDED/EndoVis17/processed")
TRAIN_DIR = BASE_DATA_DIR / "train"
VAL_DIR = BASE_DATA_DIR / "val"
TEST_DIR = BASE_DATA_DIR / "test"
MAPPINGS_PATH = BASE_DATA_DIR / "mapping_8.json"

#%%
def get_pixel_counts(dataset_dir: Path, instrument_mappings: dict) -> pd.DataFrame:
    """
    Counts pixels for each instrument class in the processed dataset.
    Applies a 1024x1024 center crop to match training augmentation.

    Args:
        dataset_dir (Path): Path to the dataset split (e.g., '.../processed/train').
        instrument_mappings (dict): Dictionary mapping instrument names to class IDs.

    Returns:
        pd.DataFrame: A DataFrame with columns ['sequence', 'instrument', 'pixel_count', 'frame_count'].
    """
    records = []
    value_to_name = {v: k for k, v in instrument_mappings.items()}
    
    sequence_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])

    for seq_dir in tqdm(sequence_dirs, desc=f"Processing {dataset_dir.name}"):
        seg_dir = seq_dir / "target" / "segmentation_8"
        if not seg_dir.is_dir():
            continue

        sequence_pixel_counts = {name: 0 for name in instrument_mappings.keys()}
        sequence_frame_sets = {name: set() for name in instrument_mappings.keys()}

        mask_files = list(seg_dir.glob('*.png'))
        for mask_path in mask_files:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Replicate 1024x1024 centercrop augmentation
                h, w = mask.shape
                crop_size = 1024
                start_x = w // 2 - crop_size // 2
                start_y = h // 2 - crop_size // 2
                mask = mask[max(0, start_y):start_y+crop_size, max(0, start_x):start_x+crop_size]

                unique, counts = np.unique(mask, return_counts=True)
                for val, count in zip(unique, counts):
                    if val in value_to_name:
                        instr_name = value_to_name[val]
                        sequence_pixel_counts[instr_name] += count
                        if count > 0:
                            sequence_frame_sets[instr_name].add(mask_path.name)

        sequence_frame_counts = {name: len(frames) for name, frames in sequence_frame_sets.items()}
        
        # Background is always present in the frame
        sequence_frame_counts["background"] = len(mask_files) 

        for instrument_name, total_pixels in sequence_pixel_counts.items():
            records.append({
                "sequence": seq_dir.name,
                "instrument": instrument_name,
                "pixel_count": total_pixels,
                "frame_count": sequence_frame_counts[instrument_name]
            })
            
    return pd.DataFrame(records)


def extract_seq_id(seq_name):
    """
    Extract the numeric sequence ID from a sequence folder name.
    """
    digits = ''.join(filter(str.isdigit, str(seq_name)))
    return int(digits) if digits else -1


def aggregate_split_pixel_counts(
    counts_df: pd.DataFrame,
    split_config: dict[str, list[int]],
    split_name: str
) -> pd.DataFrame:
    """
    Aggregate class-wise pixel counts for a named train/val/test split.
    """
    records = []

    for split_label, sequence_ids in split_config.items():
        split_df = counts_df[counts_df["seq_id"].isin(sequence_ids)]
        split_agg = split_df.groupby("instrument", as_index=False)[["pixel_count", "frame_count"]].sum()
        split_agg["subset"] = split_label
        split_agg["split_name"] = split_name
        records.append(split_agg)

    return pd.concat(records, ignore_index=True)

# Load instrument mappings
try:
    with open(MAPPINGS_PATH) as f:
        instrument_mappings = json.load(f)
except FileNotFoundError:
    print(f"Error: Mapping file not found at {MAPPINGS_PATH}")
    instrument_mappings = {}

# Process train, val, and test sets
train_counts_df = pd.DataFrame(columns=["sequence", "instrument", "pixel_count", "frame_count"])
val_counts_df = pd.DataFrame(columns=["sequence", "instrument", "pixel_count", "frame_count"])
test_counts_df = pd.DataFrame(columns=["sequence", "instrument", "pixel_count", "frame_count"])

if instrument_mappings:
    train_counts_df = get_pixel_counts(TRAIN_DIR, instrument_mappings)
    val_counts_df = get_pixel_counts(VAL_DIR, instrument_mappings)
    test_counts_df = get_pixel_counts(TEST_DIR, instrument_mappings)
else:
    raise FileNotFoundError(
        f"Instrument mapping file not found or empty: {MAPPINGS_PATH}"
    )
# Combine summaries for a comparative plot
train_counts_df['split'] = 'Train'
val_counts_df['split'] = 'Val'
test_counts_df['split'] = 'Test'
combined_counts_df = pd.concat([train_counts_df, val_counts_df, test_counts_df])

combined_counts_df['seq_id'] = combined_counts_df['sequence'].apply(extract_seq_id)

custom_split_configs = {
    "Split A": {
        "Train": [7, 8, 1, 2, 3, 4],
        "Val": [5, 6],
        "Test": [9, 10],
    },
}

custom_split_records = []

for split_name, split_config in custom_split_configs.items():
    custom_split_records.append(
        aggregate_split_pixel_counts(combined_counts_df, split_config, split_name)
    )

custom_splits_df = pd.concat(custom_split_records, ignore_index=True)

# Co-occurence matrix
import itertools

# We will collect frame-level instrument presence to compute the co-occurrence matrix
co_occurrence_records = []
base_instrument_names = list(instrument_mappings.keys())

train_sequence_dirs = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])

for seq_dir in tqdm(train_sequence_dirs, desc="Processing Co-occurrences in Train"):
    seg_dir = seq_dir / "target" / "segmentation_8"
    if not seg_dir.is_dir():
        continue
    
    value_to_name = {v: k for k, v in instrument_mappings.items()}
    mask_files = list(seg_dir.glob('*.png'))
    for mask_path in mask_files:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # Replicate 1024x1024 centercrop augmentation
            h, w = mask.shape
            crop_size = 1024
            start_x = w // 2 - crop_size // 2
            start_y = h // 2 - crop_size // 2
            mask = mask[max(0, start_y):start_y+crop_size, max(0, start_x):start_x+crop_size]

            unique = np.unique(mask)
            instruments_in_frame = set()
            for val in unique:
                if val in value_to_name:
                    instr_name = value_to_name[val]
                    if instr_name != 'background':
                        instruments_in_frame.add(instr_name)
            
            if instruments_in_frame:
                co_occurrence_records.append(list(instruments_in_frame))


# %%
### PLOTS
# --- Plot Instrument Percentage Distribution per Sequence (All Sequences) ---

# Calculate total pixels across the entire dataset to compute percentages relative to the whole dataset
total_dataset_pixels = combined_counts_df['pixel_count'].sum()

# Create a new dataframe with percentages
percent_seq_df = combined_counts_df.copy()
percent_seq_df['percentage'] = (percent_seq_df['pixel_count'] / total_dataset_pixels) * 100

# Filter out entries with zero pixels (and exclude background) for clean display
percent_seq_df = percent_seq_df[(percent_seq_df['pixel_count'] > 0) & (percent_seq_df['instrument'] != 'background')]

# Create a new formatted sequence column for the y-axis (2 digits for alignment)
percent_seq_df['formatted_sequence'] = "" + percent_seq_df['seq_id'].apply(lambda x: f"{x:02d}")

# Calculate frame percentage relative to total frames in the dataset
# Denominator is the sum of sequence frame counts (stored under 'background')
total_dataset_frames = combined_counts_df[combined_counts_df['instrument'] == 'background']['frame_count'].sum()
percent_seq_df['frame_percentage'] = (percent_seq_df['frame_count'] / total_dataset_frames) * 100

# Melt the dataframe to have multiple rows per sequence/instrument pair
melted_seq_df = percent_seq_df.melt(
    id_vars=['formatted_sequence', 'instrument', 'seq_id', 'pixel_count'],
    value_vars=['percentage', 'frame_count'], # Use raw frame_count here
    var_name='metric',
    value_name='value'
)

# Rename the metric column to readable names for faceting
metric_mapping = {
    'percentage': 'Relative Pixel Count', 
    'frame_count': 'Cumulative Occurrences'
}
melted_seq_df['metric'] = melted_seq_df['metric'].map(metric_mapping)

# Abbreviate instruments for the first graph to save space
abbr_map = {
    "bipolar_forceps": "BF",
    "prograsp_forceps": "PF",
    "large_needle_driver": "LND",
    "vessel_sealer": "VS",
    "grasping_retractor": "GR",
    "monopolar_curved_scissors": "MCS",
    "other": "Other"
}

legend_map = {k: f"{k.replace('_', ' ').title()} [{v}]" if k != "other" else "Other" for k, v in abbr_map.items()}
melted_seq_df['instrument'] = melted_seq_df['instrument'].map(legend_map)

def get_seq_text_label(row):
    is_pixel = row['metric'] == 'Relative Pixel Count'
    # Show labels if pixel percentage > 0.01%, or if frame count > 50
    threshold = 0.01 if is_pixel else 50 
    
    if row['value'] < threshold:
        raw_val = row['pixel_count'] if is_pixel else row['value']
        unit = "px" if is_pixel else "frames"
        instr_full = row['instrument']
        abbr = instr_full.split('[')[-1].split(']')[0] if '[' in instr_full else instr_full
        return f"+ {int(raw_val)} {unit} {abbr}"
    return ""

# Add text labels for very small bars (< 0.01% / 0.09%) to remain legible
melted_seq_df['text_label'] = melted_seq_df.apply(get_seq_text_label, axis=1)

# Sort sequences for y-axis representation
unique_seq_ids = sorted(melted_seq_df['seq_id'].unique(), reverse=True)
all_sequences_sorted_desc = [f"{sid:02d}" for sid in unique_seq_ids]

# Fixed original names for instruments order (top to bottom)
instrument_order = [
    "bipolar_forceps",
    "prograsp_forceps",
    "large_needle_driver",
    "vessel_sealer",
    "grasping_retractor",
    "monopolar_curved_scissors",
    "other"
]

legend_instrument_order = [legend_map[instr] for instr in instrument_order]

# --- 1) Create a horizontal bar chart displaying percentages with a split/facet vertical axis ---
fig_seq_percent = px.bar(
    melted_seq_df,
    y='formatted_sequence',
    x='value',
    color='instrument',
    facet_col='metric',
    text='text_label',
    #title='Instrument Distribution per Sequence',
    labels={'formatted_sequence': 'Sequence', 'value': 'Percentage of Total Dataset (%)', 'instrument': 'Instrument'},
    category_orders={
        "formatted_sequence": all_sequences_sorted_desc,
        "instrument": legend_instrument_order,
        "metric": ["Relative Pixel Count", "Cumulative Occurrences"]
    },
    color_discrete_sequence=px.colors.qualitative.Set2,
    orientation='h'
)

fig_seq_percent.update_traces(textposition='outside', cliponaxis=False)

# Clean up facet titles by hiding the "metric=" prefix
fig_seq_percent.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

# Allow each facet to have its own independent X-axis range
fig_seq_percent.update_xaxes(matches=None, showticklabels=True)

# Manually set ranges to prevent text cutoff
fig_seq_percent.update_xaxes(range=[0, 2.5], col=1, title_text="Percentage of Dataset (%)")
fig_seq_percent.update_xaxes(range=[0, 1300], col=2, title_text="Frames in Dataset (Count)")

# Configure dual-axis scaling and formatting
fig_seq_percent.update_layout(
    legend=dict(
        title_font_size=15.5,
        font_size=15.5,
        orientation="h",
        y=-0.2,
    ),
)

# Reversing axis so IDs count starting from 01 at the top
fig_seq_percent.update_yaxes(autorange='reversed')
fig_seq_percent.layout.yaxis.title.text = "Sequence"
fig_seq_percent.layout.yaxis2.title.text = ""

save_figure(fig_seq_percent, height=400, name='instrument_dataset_distribution', margin=(40, 40, 20, 0))

# %%
# --- Plot Instrument Percentage Distribution for Custom Split ---

# Map subset names to full written-out names
subset_mapping = {"Train": "Training Set", "Val": "Validation Set", "Test": "Test Set"}
custom_splits_df['subset_full'] = custom_splits_df['subset'].map(subset_mapping)

# Calculate percentage relative to the whole dataset
# (Reusing total_dataset_pixels computed earlier)
custom_splits_df['percentage'] = (custom_splits_df['pixel_count'] / total_dataset_pixels) * 100

# Filter out entries with zero pixels (and exclude background)
plot_split_df = custom_splits_df[(custom_splits_df['pixel_count'] > 0) & (custom_splits_df['instrument'] != 'background')].copy()

# Melt the dataframe using raw frame_count to match the first graph
melted_split_df = plot_split_df.melt(
    id_vars=['instrument', 'subset_full', 'pixel_count'],
    value_vars=['percentage', 'frame_count'], 
    var_name='metric',
    value_name='value'
)

metric_mapping = {
    'percentage': 'Relative Pixel Count', 
    'frame_count': 'Occurrences'
}
melted_split_df['metric'] = melted_split_df['metric'].map(metric_mapping)

display_map = {k: k.replace('_', ' ').title() for k in abbr_map.keys()}
melted_split_df['instrument'] = melted_split_df['instrument'].map(display_map)
display_instrument_order = [display_map[instr] for instr in instrument_order]

def get_text_label(row):
    is_pixel = row['metric'] == 'Relative Pixel Count'
    # Updated threshold: 0.01 for percentage, 50 for raw frame counts
    threshold = 0.01 if is_pixel else 50
    
    if row['value'] < threshold:
        raw_val = row['pixel_count'] if is_pixel else row['value']
        unit = "px" if is_pixel else "frames"
        return f"{int(raw_val)} {unit}"
    return ""

# Add text labels for very small bars to remain legible
melted_split_df['text_label'] = melted_split_df.apply(get_text_label, axis=1)

# Generate a descriptive title specifying the current active split's mapping
split_name = "Split A"
split_details = " | ".join([f"{k} {v}" for k, v in custom_split_configs[split_name].items()])
title_str = f"Instrument Distribution for the Split: {split_details}"

# Create a horizontal bar chart displaying percentages with a facet vertical axis
fig_split_percent = px.bar(
    melted_split_df,
    y='instrument',
    x='value',
    color='subset_full',
    facet_col='metric',
    barmode='group',
    text='text_label',
    #title=title_str,
    labels={'subset_full': 'Dataset Split', 'value': '', 'instrument': 'Instrument'},
    category_orders={
        "subset_full": ["Test Set", "Validation Set", "Training Set"], # Reversed so Training draws at the top
        "metric": ["Relative Pixel Count", "Occurrences"] # Updated to match new mapping
    },
    color_discrete_map={
        "Training Set": px.colors.qualitative.Plotly[0], # Standard Plotly Blue
        "Validation Set": px.colors.qualitative.Plotly[1], # Standard Plotly Red/Orange
        "Test Set": px.colors.qualitative.Plotly[2], # Standard Plotly Green
    },
    orientation='h'
)

fig_split_percent.update_traces(textposition='outside', cliponaxis=False)

# Clean up facet titles by hiding the "metric=" prefix
fig_split_percent.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

# Allow each facet to have its own independent X-axis range
fig_split_percent.update_xaxes(matches=None, showticklabels=True)

fig_split_percent.update_layout(
    legend=dict(
        traceorder='reversed',
        title_font_size=15.5,
        font_size=15.5,
        orientation="h",
        y=-0.2,
        ), 
    yaxis={'categoryorder': 'array', 'categoryarray': display_instrument_order[::-1]} 
)

# Prevent doubled y-axis titles
if hasattr(fig_split_percent.layout, 'yaxis'):
    fig_split_percent.layout.yaxis.title.text = "Instrument"
if hasattr(fig_split_percent.layout, 'yaxis2'):
    fig_split_percent.layout.yaxis2.title.text = ""

# Set independent x-axis titles so the right side doesn't say "Percentage"
fig_split_percent.update_xaxes(title_text="Percentage of Dataset (%)", col=1)
fig_split_percent.update_xaxes(title_text="Frames in Dataset (Count)", col=2)

save_figure(fig_split_percent, height=400, name='instrument_dataset_distribution_split', margin=(0, 0, 20, 0))


# %%
# --- Co-occurrence Matrix for Training Set ---
# Initialize co-occurrence matrix
co_matrix = pd.DataFrame(0, index=base_instrument_names, columns=base_instrument_names)

# Populate matrix
for instruments in co_occurrence_records:
    # Occurrences (diagonal)
    for inst in instruments:
        co_matrix.loc[inst, inst] += 1
    # Co-occurrences (off-diagonal)
    for inst1, inst2 in itertools.combinations(instruments, 2):
        co_matrix.loc[inst1, inst2] += 1
        co_matrix.loc[inst2, inst1] += 1

# Filter out instruments with zero occurrences in the training set
co_matrix = co_matrix.loc[(co_matrix.sum(axis=1) > 0), (co_matrix.sum(axis=0) > 0)]

# Format instrument names for display
co_matrix.index = [display_map.get(idx, idx.replace('_', ' ').title()) for idx in co_matrix.index]
co_matrix.columns = [display_map.get(col, col.replace('_', ' ').title()) for col in co_matrix.columns]



# Plot Heatmap
fig_co = px.imshow(
    co_matrix, 
    labels=dict(x="Instrument", y="Instrument", color="Co-occurrences"),
    x=co_matrix.columns, 
    y=co_matrix.index,
    #title="Instrument Co-occurrence Matrix of Training Set",
    color_continuous_scale="Viridis",
    text_auto=True
)

fig_co.update_layout(
    coloraxis_colorbar=dict(
        len=0.84,
        y=0.535,
        yanchor="middle",
        title_font_size=15.5,
        tickfont_size=15.5,
    ),
)

save_figure(fig_co, height=450, name='instrument_cooccurrence_matrix')

# %%
