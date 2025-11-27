"""
A utility script to prepare a public metabolomics dataset (Human Cachexia) from
MetaboAnalyst for ingestion into the Metabo-MVP application.

This script will:
1. Define the source URL for the raw data.
2. Load the raw data, which is in a "wide" format (samples in columns).
3. Transform the data into a "long" feature list for a single sample.
4. Parse combined peak names (e.g., "mz_rt") into separate columns.
5. Save the cleaned data to a new CSV file, ready for our application.
"""

import pandas as pd
import urllib.request
import os

# --- Configuration ---

# Get the project's root directory (where this script is located)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Source URL for the MetaboAnalyst Human Cachexia dataset (this can be unstable)
SOURCE_URL = "https://rest.xialab.ca/api/download/metaboanalyst/human_cachexia.csv"
LOCAL_RAW_PATH = os.path.join(DATA_DIR, "human_cachexia.csv")

# Path to save the final, cleaned CSV file
OUTPUT_PATH = os.path.join(DATA_DIR, "prepared_metaboanalyst_cachexia.csv")


def main():
    """Main function to run the data preparation pipeline."""
    print("Starting dataset preparation...")

    # --- Robust Data Loading ---
    # 1. Check if the raw data file already exists locally.
    if not os.path.exists(LOCAL_RAW_PATH):
        print(f"Raw data not found locally. Attempting to download from {SOURCE_URL}")
        try:
            # 2. If not, try to download it.
            urllib.request.urlretrieve(SOURCE_URL, LOCAL_RAW_PATH)
            print(f"✅ Download successful. Saved to {LOCAL_RAW_PATH}")
        except Exception as e:
            # 3. If download fails, give the user clear instructions.
            print(f"\n❌ ERROR: Could not download the data file: {e}")
            print("\n--- MANUAL ACTION REQUIRED ---")
            print(f"1. Open this URL in your browser: {SOURCE_URL}")
            print("2. Save the file as 'human_cachexia.csv'")
            print(f"3. Place the file inside this directory: {DATA_DIR}")
            print("4. Run this script again.")
            return # Stop execution

    # Load the raw data, treating the first column as the index (peak names)
    df = pd.read_csv(LOCAL_RAW_PATH)

    print(f"Raw data loaded. Shape: {df.shape}")

    # We will transform the data for the first sample (first row) into a "long" feature list.
    first_sample_row = df.iloc[0]
    sample_id = first_sample_row.iloc[0]
    print(f"Extracting feature list for the first sample: '{sample_id}'")

    # The first two columns are metadata ('Patient ID', 'Muscle loss'). Features start from the 3rd column.
    feature_data = first_sample_row.iloc[2:]

    # Create a new DataFrame from this series
    final_df = pd.DataFrame({
        'feature_id': feature_data.index,
        'intensity': feature_data.values
    })

    # This dataset does not contain m/z or rt, so we will create placeholder values.
    final_df['mz'] = final_df['feature_id'].apply(lambda x: 100 + (hash(x) % 90000) / 100.0)
    final_df['rt_sec'] = 0.0 # No retention time provided

    print(f"Saving cleaned data with {len(final_df)} rows to {OUTPUT_PATH}")
    final_df.to_csv(OUTPUT_PATH, index=False)

    print("✅ Dataset preparation complete!")

if __name__ == "__main__":
    # Call the main function to execute the script's logic
    main()
