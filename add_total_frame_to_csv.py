import os
import torch
import pandas as pd
from tqdm import tqdm

def add_total_frames_to_csv(input_csv, output_csv, emb_dir):
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Initialize the new column
    total_frames_list = []
    
    print("Extracting frame counts from embeddings...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        vid_id = row['VideoID']
        emb_path = os.path.join(emb_dir, f"{vid_id}.pt")
        
        if os.path.exists(emb_path):
            try:
                # We only need the shape, but torch.load reads the whole tensor.
                # Loading to CPU to prevent GPU memory spikes.
                emb = torch.load(emb_path, map_location='cpu')
                total_frames_list.append(emb.shape[0])
            except Exception as e:
                print(f"Error loading {emb_path}: {e}")
                total_frames_list.append(0) # Or a default value
        else:
            print(f"Warning: Embedding not found for {vid_id} at {emb_path}")
            total_frames_list.append(0) # Default if missing
            
    df['TotalFrames'] = total_frames_list
    
    df.to_csv(output_csv, index=False)
    print(f"Done! Saved updated CSV to {output_csv}")

# --- Run the script ---
if __name__ == "__main__":
    INPUT_CSV = "/scratch/lt200353-pcllm/location/cas_colon/Video_Label.csv"   # <-- UPDATE THIS
    OUTPUT_CSV = "/scratch/lt200353-pcllm/location/cas_colon/updated_Video_Label.csv"  # <-- UPDATE THIS
    EMB_DIR = "/scratch/lt200353-pcllm/location/cas_colon/features_dinov3"
    
    add_total_frames_to_csv(INPUT_CSV, OUTPUT_CSV, EMB_DIR)
