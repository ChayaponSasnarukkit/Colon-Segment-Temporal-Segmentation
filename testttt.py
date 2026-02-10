import pandas as pd
import os

# ==========================================
#   CONFIGURATION
# ==========================================
# 1. Path to your current Master CSV (The one with labels/metadata)
MASTER_CSV_PATH = "/scratch/lt200353-pcllm/location/cas_colon/Video_Label.csv" 

# 2. Path to the split files
SPLIT_ROOT = "/scratch/lt200353-pcllm/location/cas_colon/splits"
TRAIN_BUNDLE = os.path.join(SPLIT_ROOT, "train.split1.bundle")
TEST_BUNDLE = os.path.join(SPLIT_ROOT, "test.split1.bundle")

# 3. Output Paths (Where to save the new CSVs)
OUTPUT_DIR = "/scratch/lt200353-pcllm/location/cas_colon/"
TRAIN_CSV_OUT = os.path.join(OUTPUT_DIR, "train_split.csv")
TEST_CSV_OUT = os.path.join(OUTPUT_DIR, "test_split.csv")

# 4. The column name in your CSV that matches the bundle file content
#    (e.g., "video_name", "id", "video")
VIDEO_COL_NAME = "VideoID"  # <--- CHANGE THIS if your column is named differently

# ==========================================
#   EXECUTION
# ==========================================
def read_bundle(path):
    """Reads a bundle file into a clean set of video names."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    
    with open(path, 'r') as f:
        # Read lines, strip whitespace, and filter empty lines
        # We also strip file extensions (.mp4) just in case the CSV doesn't have them
        # but the split file does (or vice versa).
        return set(line.strip() for line in f if line.strip())

def main():
    print(f"📂 Loading Master CSV: {MASTER_CSV_PATH}")
    df = pd.read_csv(MASTER_CSV_PATH)
    print(f"   Total rows in master: {len(df)}")
    
    # Ensure the video column is string type and strip whitespace
    df[VIDEO_COL_NAME] = df[VIDEO_COL_NAME].astype(str).str.strip()

    print("📄 Reading Split Bundles...")
    train_videos = read_bundle(TRAIN_BUNDLE)
    test_videos = read_bundle(TEST_BUNDLE)
    
    print(f"   Train split count: {len(train_videos)}")
    print(f"   Test split count:  {len(test_videos)}")

    # --- Filter Dataframes ---
    # We check if the video name in the CSV exists in the bundle set
    train_df = df[df[VIDEO_COL_NAME].isin(train_videos)].copy()
    test_df = df[df[VIDEO_COL_NAME].isin(test_videos)].copy()

    # --- Save Separate CSVs ---
    print("\n💾 Saving separate CSVs...")
    
    train_df.to_csv(TRAIN_CSV_OUT, index=False)
    print(f"   ✅ Saved: {TRAIN_CSV_OUT} ({len(train_df)} rows)")
    
    test_df.to_csv(TEST_CSV_OUT, index=False)
    print(f"   ✅ Saved: {TEST_CSV_OUT} ({len(test_df)} rows)")

    # --- Validation Check ---
    if len(train_df) == 0:
        print("\n⚠️ WARNING: The train CSV is empty! Check if 'VIDEO_COL_NAME' matches your CSV header.")
    if len(test_df) == 0:
        print("\n⚠️ WARNING: The test CSV is empty!")

if __name__ == "__main__":
    main()
