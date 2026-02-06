#%%
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm

from setup import setup_environment
setup_environment()

#%%
# --- Configuration ---
# Adjust these paths if your directory structure is different.
BASE_DATA_DIR = Path("/data/Zeitler/SIDED/EndoVis17/raw")
TRAIN_DIR = BASE_DATA_DIR / "train"
TEST_DIR = BASE_DATA_DIR / "test"
# The mapping file is expected in the training directory.
MAPPINGS_PATH = TRAIN_DIR / "instrument_type_mapping.json"

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

        # Iterate over actual instrument folders in the ground_truth directory
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
        for instrument_name, total_pixels in sequence_pixel_counts.items():
            records.append({
                "sequence": seq_dir.name,
                "instrument": instrument_name,
                "pixel_count": total_pixels
            })
            
    return pd.DataFrame(records)

# Load instrument mappings
try:
    with open(MAPPINGS_PATH) as f:
        instrument_mappings = json.load(f)
except FileNotFoundError:
    print(f"Error: Mapping file not found at {MAPPINGS_PATH}")
    instrument_mappings = {}

# Process both train and test sets
if instrument_mappings:
    train_counts_df = get_pixel_counts(TRAIN_DIR, instrument_mappings)
    test_counts_df = get_pixel_counts(TEST_DIR, instrument_mappings)

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
fig_train.show()


#%%
# --- Plot Test Set Imbalance ---
fig_test = px.bar(
    test_summary,
    title='Total Pixel Count per Instrument in Test Set',
    labels={'index': 'Instrument Type', 'value': 'Total Pixels'},
    log_y=True
)
fig_test.update_layout(xaxis_tickangle=-45, yaxis_title='Total Pixels (log scale)')
fig_test.show()

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
fig_combined.show()
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
fig_seq_combined.show()

# %%
