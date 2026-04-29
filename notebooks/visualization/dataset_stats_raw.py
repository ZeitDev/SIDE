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

from utils.setup import setup_environment
os.chdir('/data/Zeitler/code/SIDE')
setup_environment(skip_cuda=True)

#%%
# --- Configuration ---
BASE_DATA_DIR = Path("/data/Zeitler/SIDED/EndoVis17/raw")
TRAIN_DIR = BASE_DATA_DIR / "train"
TEST_DIR = BASE_DATA_DIR / "test"
MAPPINGS_PATH = BASE_DATA_DIR / "mapping_8.json"

#%%
def get_pixel_counts(dataset_dir: Path, instrument_mappings: dict) -> pd.DataFrame:
    """
    Counts non-zero pixels for each instrument class in a given dataset directory.
    It handles variations in folder names (e.g., 'Left_'/'Right_' prefixes).

    Args:
        dataset_dir (Path): Path to the dataset split (e.g., '.../raw/train').
        instrument_mappings (dict): Dictionary mapping instrument names to class IDs.

    Returns:
        pd.DataFrame: A DataFrame with columns ['sequence', 'instrument', 'pixel_count'].
    """
    records = []
    base_instrument_names = list(instrument_mappings.keys())
    
    sequence_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])

    for seq_dir in tqdm(sequence_dirs, desc=f"Processing {dataset_dir.name}"):
        gt_dir = seq_dir / "ground_truth"
        if not gt_dir.is_dir():
            continue

        # Create a temporary dictionary to store pixel counts for this sequence
        # to aggregate left/right versions before adding to records.
        sequence_pixel_counts = {name: 0 for name in base_instrument_names}

        # Iterate over actual instrument folders in the target directory
        instrument_folders = [d for d in gt_dir.iterdir() if d.is_dir()]
        for instrument_dir in instrument_folders:
            folder_name_str = instrument_dir.name.replace('_', ' ')
            
            # Find the corresponding base instrument name
            matched_instrument = None
            for base_name in base_instrument_names:
                if base_name in folder_name_str:
                    matched_instrument = base_name
                    break
            
            if matched_instrument:
                total_pixels = 0
                mask_files = list(instrument_dir.glob('*.png'))
                for mask_path in mask_files:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        total_pixels += np.count_nonzero(mask)
                
                # Add pixels to the matched base instrument
                sequence_pixel_counts[matched_instrument] += total_pixels

        # Append aggregated counts for the sequence to the main records list
        
        # Calculate background if it is one of the trackable base instruments
        # Always calculate background and name it 'background'
        sample_mask_path = next(gt_dir.rglob('*.png'), None)
        if sample_mask_path:
            img_shape = cv2.imread(str(sample_mask_path), cv2.IMREAD_GRAYSCALE).shape
            area = img_shape[0] * img_shape[1]
            max_frames = max([len(list(p.glob('*.png'))) for p in gt_dir.iterdir() if p.is_dir()], default=0)
            total_sequence_pixels = max_frames * area
            sum_instruments = sum(sequence_pixel_counts.values())
            sequence_pixel_counts["background"] = max(0, total_sequence_pixels - sum_instruments)

        for instrument_name, total_pixels in sequence_pixel_counts.items():
            records.append({
                "sequence": seq_dir.name,
                "instrument": instrument_name,
                "pixel_count": total_pixels
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
        split_agg = split_df.groupby("instrument", as_index=False)["pixel_count"].sum()
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

# Process both train and test sets
train_counts_df = pd.DataFrame(columns=["sequence", "instrument", "pixel_count"])
test_counts_df = pd.DataFrame(columns=["sequence", "instrument", "pixel_count"])

if instrument_mappings:
    train_counts_df = get_pixel_counts(TRAIN_DIR, instrument_mappings)
    test_counts_df = get_pixel_counts(TEST_DIR, instrument_mappings)
else:
    raise FileNotFoundError(
        f"Instrument mapping file not found or empty: {MAPPINGS_PATH}"
    )

#%%
# --- Aggregate and Display Results ---
# Group by instrument and sum the pixel counts for the entire dataset split.
train_summary = train_counts_df.groupby("instrument")["pixel_count"].sum().sort_values(ascending=False)
test_summary = test_counts_df.groupby("instrument")["pixel_count"].sum().sort_values(ascending=False)

print("--- Training Set Pixel Counts per Class ---")
print(train_summary)
print("\n" + "="*40 + "\n")
print("--- Test Set Pixel Counts per Class ---")
print(test_summary)

#%%
# --- Plot Training Set Imbalance ---
fig_train = px.bar(
    train_summary,
    title='Total Pixel Count per Instrument in Training Set',
    labels={'index': 'Instrument Type', 'value': 'Total Pixels'},
    log_y=True
)
fig_train.update_layout(xaxis_tickangle=-45, yaxis_title='Total Pixels (log scale)')
#fig_train.show()


#%%
# --- Plot Test Set Imbalance ---
fig_test = px.bar(
    test_summary,
    title='Total Pixel Count per Instrument in Test Set',
    labels={'index': 'Instrument Type', 'value': 'Total Pixels'},
    log_y=True
)
fig_test.update_layout(xaxis_tickangle=-45, yaxis_title='Total Pixels (log scale)')
#fig_test.show()

#%%
# --- Compare Train vs. Test Distribution ---
# Combine summaries for a comparative plot
train_summary_df = train_summary.reset_index()
train_summary_df['split'] = 'Train'
test_summary_df = test_summary.reset_index()
test_summary_df['split'] = 'Test'

combined_summary = pd.concat([train_summary_df, test_summary_df])

# Plot
fig_combined = px.bar(
    combined_summary,
    x='instrument',
    y='pixel_count',
    color='split',
    barmode='group',
    title='Comparison of Pixel Counts (Train vs. Test)',
    labels={'instrument': 'Instrument Type', 'pixel_count': 'Total Pixels'},
    log_y=False
)
fig_combined.update_layout(
    xaxis_tickangle=-45,
    yaxis_title='Total Pixels',
    legend_title='Dataset Split'
)
#fig_combined.show()
# %%

# --- Plot Instrument Distribution per Sequence (Combined Train & Test) ---

# Add a 'split' column to each dataframe before combining
train_counts_df['split'] = 'Train'
test_counts_df['split'] = 'Test'
combined_counts_df = pd.concat([train_counts_df, test_counts_df])

# Filter out entries with zero pixels for a cleaner plot
combined_seq_df = combined_counts_df[combined_counts_df['pixel_count'] > 0]

# Get a sorted list of all unique sequences to maintain order
all_sequences_sorted = sorted(combined_counts_df['sequence'].unique())

# Create a faceted bar chart
fig_seq_combined = px.bar(
    combined_seq_df,
    x='sequence',
    y='pixel_count',
    color='instrument',
    facet_col='split',  # Create separate columns for 'Train' and 'Test'
    title='Instrument Pixel Counts per Sequence (Train vs. Test)',
    labels={'sequence': 'Sequence', 'pixel_count': 'Total Pixels', 'instrument': 'Instrument'},
    category_orders={"sequence": all_sequences_sorted}
)

# Update layout for better readability
fig_seq_combined.update_xaxes(tickangle=-90)
fig_seq_combined.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])) # Clean up facet titles
fig_seq_combined.update_layout(yaxis_title='Total Pixels')
#fig_seq_combined.show()

# %%
# --- Cross Validation Folds Analysis ---

# Define the 5 folds based on sequence IDs
# Fold definitions:
# Fold 0: Test {1, 2} | Train {3..10}
# Fold 1: Test {3, 4} | Train {1, 2, 5..10}
# Fold 2: Test {5, 6} | Train {1..4, 7..10}
# Fold 3: Test {7, 8} | Train {1..6, 9, 10}
# Fold 4: Test {9, 10}| Train {1..8} (Official Split)

folds_config = {
    "Fold 0": [1, 2],
    "Fold 1": [3, 4],
    "Fold 2": [5, 6],
    "Fold 3": [7, 8],
    "Fold 4": [9, 10]
}
all_seq_ids = set(range(1, 11))

# Create a column for sequence ID to facilitate filtering
combined_counts_df['seq_id'] = combined_counts_df['sequence'].apply(extract_seq_id)

fold_records = []

for fold_name, test_ids_list in folds_config.items():
    test_ids = set(test_ids_list)
    train_ids = all_seq_ids - test_ids
    
    # Aggregate Train for this fold
    # Filter by seq_id being in the calculated train_ids set
    train_mask = combined_counts_df['seq_id'].isin(train_ids)
    train_fold_data = combined_counts_df[train_mask]
    train_agg = train_fold_data.groupby("instrument")["pixel_count"].sum().reset_index()
    train_agg["split"] = "Train"
    train_agg["fold"] = fold_name
    fold_records.append(train_agg)
    
    # Aggregate Test for this fold
    test_mask = combined_counts_df['seq_id'].isin(test_ids)
    test_fold_data = combined_counts_df[test_mask]
    test_agg = test_fold_data.groupby("instrument")["pixel_count"].sum().reset_index()
    test_agg["split"] = "Test"
    test_agg["fold"] = fold_name
    fold_records.append(test_agg)

cv_df = pd.concat(fold_records)

# Plot comparison
fig_cv = px.bar(
    cv_df,
    x="instrument",
    y="pixel_count",
    color="split",
    barmode="group",
    facet_col="fold",
    facet_col_wrap=3, # Arranges plots in a grid (3 columns)
    title="Class Balance across 5 Cross-Validation Folds",
    labels={'instrument': 'Instrument', 'pixel_count': 'Total Pixels'},
    category_orders={"fold": sorted(folds_config.keys())}
)

fig_cv.update_xaxes(tickangle=-45)
fig_cv.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])) # Clean facet labels
fig_cv.update_layout(yaxis_title="Total Pixels")
fig_cv.show()

# %%
# --- Total Pixel Count Comparison (Train vs Test per Fold) ---
total_pixels_per_fold = cv_df.groupby(["fold", "split"])["pixel_count"].sum().reset_index()

fig_total_cv = px.bar(
    total_pixels_per_fold,
    x="fold",
    y="pixel_count",
    color="split",
    barmode="group",
    text_auto='.2s',
    title="Total Pixel Count: Train vs Test per Fold",
    labels={'fold': 'Fold', 'pixel_count': 'Total Pixels', 'split': 'Split'}
)
fig_total_cv.update_layout(yaxis_title="Total Pixels")
#fig_total_cv.show()

# %%
# --- Custom Train / Val / Test Split Comparison ---

# Add more named split definitions here to compare alternative train/val/test setups.
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

fig_custom_splits = px.bar(
    custom_splits_df,
    x="instrument",
    y="pixel_count",
    color="subset",
    barmode="group",
    facet_col="split_name",
    title="Class Pixel Counts for Custom Train / Val / Test Splits",
    labels={
        "instrument": "Instrument",
        "pixel_count": "Total Pixels",
        "subset": "Subset",
        "split_name": "Split Definition",
    },
    category_orders={"subset": ["Train", "Val", "Test"]},
)

fig_custom_splits.update_xaxes(tickangle=-45)
fig_custom_splits.for_each_annotation(
    lambda a: a.update(text=a.text.split("=")[-1])
)
fig_custom_splits.update_layout(yaxis_title="Total Pixels")
#fig_custom_splits.show()

# %%
### THESIS

# %%
# --- Plot Instrument Percentage Distribution per Sequence (All Sequences) ---

# Calculate total pixels across the entire dataset to compute percentages relative to the whole dataset
total_dataset_pixels = combined_counts_df['pixel_count'].sum()

# Create a new dataframe with percentages
percent_seq_df = combined_counts_df.copy()
percent_seq_df['percentage'] = (percent_seq_df['pixel_count'] / total_dataset_pixels) * 100

# Filter out entries with zero pixels (and exclude background) for clean display
percent_seq_df = percent_seq_df[(percent_seq_df['pixel_count'] > 0) & (percent_seq_df['instrument'] != 'background')]

# Create a new formatted sequence column for the y-axis (2 digits for alignment)
percent_seq_df['formatted_sequence'] = "Instrument Dataset " + percent_seq_df['seq_id'].apply(lambda x: f"{x:02d}")

# Sort sequences for y-axis representation
unique_seq_ids = sorted(percent_seq_df['seq_id'].unique(), reverse=True)
all_sequences_sorted_desc = [f"Instrument Dataset {sid:02d}" for sid in unique_seq_ids]

# Fixed original names for instruments order (top to bottom)
instrument_order = [
    "Bipolar Forceps",
    "Prograsp Forceps",
    "Large Needle Driver",
    "Vessel Sealer",
    "Grasping Retractor",
    "Monopolar Curved Scissors",
    "Other"
]

# Create a horizontal bar chart displaying percentages
fig_seq_percent = px.bar(
    percent_seq_df,
    y='formatted_sequence',
    x='percentage',
    color='instrument',
    title='Instrument Distribution per Sequence (Relative to Total Dataset)',
    labels={'formatted_sequence': 'Sequence', 'percentage': 'Percentage of Total Dataset Pixels (%)', 'instrument': 'Instrument'},
    category_orders={
        "formatted_sequence": all_sequences_sorted_desc,
        "instrument": instrument_order
    },
    orientation='h'
)

fig_seq_percent.update_layout(
    xaxis_title='Percentage of Total Dataset Pixels (%)', 
    yaxis_title='Sequence',
    width=900,   # Optimal width for thesis PDFs (fits nicely across the page)
    height=500,  # Allows enough spacing for 10 bars without feeling cramped
    margin=dict(l=20, r=20, t=50, b=20), # Tighten margins for clean export
    font=dict(
        family="Latin Modern Roman, Computer Modern Roman, serif",
        size=14
    ),
    yaxis={'autorange': 'reversed'} # Ensures sequences render 01 to 10 top-to-bottom
)

# Export the downloaded file as crisp scalable vector graphics (SVG) 
fig_seq_percent.show(config={
    'toImageButtonOptions': {
        'format': 'svg', 
        'filename': 'notebooks/output/instrument_distribution_seq'
    }
})

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

# Generate a descriptive title specifying the current active split's mapping
split_name = "Split A"
split_details = " | ".join([f"{k} {v}" for k, v in custom_split_configs[split_name].items()])
title_str = f"Instrument Distribution for Custom Split ({split_details})"

# Create a horizontal bar chart displaying percentages
fig_split_percent = px.bar(
    plot_split_df,
    y='instrument',
    x='percentage',
    color='subset_full',
    barmode='group',
    title=title_str,
    labels={'subset_full': 'Dataset Split', 'percentage': 'Percentage of Total Dataset Pixels (%)', 'instrument': 'Instrument'},
    category_orders={
        "subset_full": ["Training Set", "Validation Set", "Test Set"],
    },
    orientation='h'
)

fig_split_percent.update_layout(
    xaxis_title='Percentage of Total Dataset Pixels (%)', 
    yaxis_title='Instrument',
    width=900,   
    height=500,  # Adjusted height for instrument bars
    margin=dict(l=20, r=20, t=50, b=20), 
    font=dict(
        family="Latin Modern Roman, Computer Modern Roman, serif",
        size=14
    ),
    yaxis={'categoryorder': 'array', 'categoryarray': instrument_order[::-1]} # Reverses instrument order explicitly so Bipolar Forceps is on top in this barmode
)

# Export configuration for SVG
fig_split_percent.show(config={
    'toImageButtonOptions': {
        'format': 'svg', 
        'filename': 'notebooks/output/instrument_distribution_custom_split'
    }
})

# %%
