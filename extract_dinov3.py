import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
import numpy as np
import os
import glob
from tqdm import tqdm
from decord import VideoReader, cpu

# --- Configuration ---
VIDEO_DIR = "/scratch/lt200353-pcllm/location/cas_colon/"
OUTPUT_DIR = "/scratch/lt200353-pcllm/location/cas_colon/features_dinov3"
BATCH_SIZE = 512                  
NUM_WORKERS = 8
TARGET_FPS = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Choose Model Variant:
# - Base:  "facebook/dinov3-vitb16-pretrain-lvd1689m"
# - Large: "facebook/dinov3-vitl16-pretrain-lvd1689m" (Recommended)
# - Huge:  "facebook/dinov3-vith16-pretrain-lvd1689m"
MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

# --- 1. Define the Backbone (DINOv3) ---
class DINOv3FeatureExtractor(nn.Module):
    def __init__(self, model_id):
        super(DINOv3FeatureExtractor, self).__init__()
        print(f"Loading DINOv3 model: {model_id} ...")
        
        # Load Model from Hugging Face
        self.model = AutoModel.from_pretrained(model_id)
        
        # Load Processor (Handles Resize, Norm, etc.)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        
        # Freeze model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        # pixel_values: (B, 3, H, W)
        outputs = self.model(pixel_values=pixel_values)
        
        # DINOv3 Output: last_hidden_state is (B, SeqLen, HiddenDim)
        # Index 0 is the CLS token (Global Representation)
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :] # (B, HiddenDim)
        
        return cls_token

# --- 2. Main Extraction Loop ---
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Setup Model
    extractor = DINOv3FeatureExtractor(MODEL_ID).to(DEVICE)
    extractor.eval()
    
    # Get the processor for manual usage in the loop
    processor = extractor.processor

    # 2. Find Videos
    video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    print(f"Found {len(video_files)} videos. Starting DINOv3 extraction...")

    # 3. Process Each Video
    for vid_path in tqdm(video_files):
        video_name = os.path.splitext(os.path.basename(vid_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{video_name}.npy")
        
        if os.path.exists(out_path):
            continue

        try:
            # A. Read Video
            vr = VideoReader(vid_path, ctx=cpu(0))
            total_frames = len(vr)
            fps = vr.get_avg_fps()

            # B. Calculate Indices
            step = max(1, int(round(fps / TARGET_FPS)))
            indices = list(range(0, total_frames, step))

            video_feats = []

            # C. Batch Processing
            for i in tqdm(range(0, len(indices), BATCH_SIZE)):
                batch_indices = indices[i : i + BATCH_SIZE]
                
                # Get raw frames (B, H, W, C) - uint8
                raw_frames = vr.get_batch(batch_indices).asnumpy()
                
                # Convert to list of arrays for the processor
                list_frames = [raw_frames[j] for j in range(len(raw_frames))]

                # D. Preprocess with Hugging Face Processor
                # DINOv3 uses 224x224 by default usually, but we enforce it here for consistency.
                inputs = processor(
                    images=list_frames, 
                    return_tensors="pt", 
                    do_resize=True, 
                    size={"height": 224, "width": 224} 
                )
                
                pixel_values = inputs["pixel_values"].to(DEVICE)

                # E. Inference
                with torch.no_grad():
                    # (B, HiddenDim)
                    feats = extractor(pixel_values)
                    video_feats.append(feats.cpu().numpy())

            # F. Save
            if len(video_feats) > 0:
                full_video_feats = np.concatenate(video_feats, axis=0)
                # Save as (T, Feature_Dim) -> e.g. (T, 1024)
                np.save(out_path, full_video_feats) 

        except Exception as e:
            print(f"Error on {video_name}: {e}")

if __name__ == "__main__":
    main()
