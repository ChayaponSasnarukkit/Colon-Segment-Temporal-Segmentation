import csv
import os
import random

# --- Configuration ---
# Source file paths (from your cat command)
# Note: If running this locally on your machine, ensure you have access to these paths
# or update them to where the files are located.
SOURCE_TEST_FILE = "/scratch/lt200353-pcllm/location/cas_colon/updated_test_split.csv"
SOURCE_TRAIN_FILE = "/scratch/lt200353-pcllm/location/cas_colon/updated_train_split.csv"

# Output directory
OUTPUT_DIR = "cv_folds_generated"

# --- Functions ---

def read_csv_data(filepath):
    """Reads CSV and returns the header and a list of data lines."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Separate header from data
    # We assume the first line is the header
    header = lines[0].strip()
    data = [line.strip() for line in lines[1:] if line.strip()]
    return header, data

def write_csv(filepath, header, data_lines):
    """Writes a CSV file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(header + "\n")
        f.write("\n".join(data_lines))
        # Ensure the file ends with a newline
        f.write("\n")

# --- Main Logic ---

def main():
    # 1. Read existing files
    try:
        print(f"Reading Test Split from: {SOURCE_TEST_FILE}")
        header, current_test_lines = read_csv_data(SOURCE_TEST_FILE)
        
        print(f"Reading Train Split from: {SOURCE_TRAIN_FILE}")
        _, current_train_lines = read_csv_data(SOURCE_TRAIN_FILE)
    except FileNotFoundError as e:
        print(f"Error: Could not find file. {e}")
        return

    # 2. Setup Partitioning
    # Partition 1 is the original test set (16 videos)
    partition_1 = current_test_lines
    
    # Partitions 2-5 will come from the original training set (62 videos)
    # We shuffle them first to ensure random distribution
    training_pool = current_train_lines[:]
    random.seed(42) # Fixed seed for reproducibility
    random.shuffle(training_pool)
    
    # Calculate split sizes automatically
    # We need to split 'training_pool' into 4 parts
    total_pool = len(training_pool)
    chunk_size = total_pool // 4
    remainder = total_pool % 4
    
    partitions = [partition_1] # Start with the first partition
    
    start = 0
    for i in range(4):
        # Distribute the remainder (extra items) one by one to the first few chunks
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        partitions.append(training_pool[start:end])
        start = end

    # 3. Generate the 5 Folds
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nGenerating 5 folds in '{OUTPUT_DIR}/'...")

    for i in range(5):
        fold_num = i + 1
        
        # CURRENT FOLD TEST SET = Partition[i]
        test_data = partitions[i]
        
        # CURRENT FOLD TRAIN SET = All other partitions combined
        train_data = []
        for j in range(5):
            if i != j:
                train_data.extend(partitions[j])
        
        # Write files
        test_filename = os.path.join(OUTPUT_DIR, f"fold{fold_num}_test.csv")
        train_filename = os.path.join(OUTPUT_DIR, f"fold{fold_num}_train.csv")
        
        write_csv(test_filename, header, test_data)
        write_csv(train_filename, header, train_data)
        
        print(f"Fold {fold_num}: Test={len(test_data)} videos, Train={len(train_data)} videos")

    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()
