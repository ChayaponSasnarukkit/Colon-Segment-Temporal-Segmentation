import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from tqdm import tqdm
from decord import VideoReader, cpu

# --- Configuration ---
VIDEO_DIR = "/scratch/lt200353-pcllm/location/cas_colon/"          # Folder containing your .mp4 files
OUTPUT_DIR = "/scratch/lt200353-pcllm/location/cas_colon/features"       # Folder to save .npy files
BATCH_SIZE = 128                     # Adjust based on GPU VRAM (64-128 is usually safe)
NUM_WORKERS = 8                      # Number of CPU threads for loading
TARGET_FPS = 1                       # Downsample to 1 fps (Standard for Cholec80/AutoLaparo)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Define the Backbone (ResNet50) ---
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, weight_path="./resnet50-0676ba61.pth"):
        super(ResNetFeatureExtractor, self).__init__()
        # Load standard ResNet-50
        resnet = models.resnet50(weights=None)

        # 2. Load local weights
        if os.path.exists(weight_path):
            print(f"Loading local weights from {weight_path}")
            state_dict = torch.load(weight_path, map_location="cpu")
            resnet.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Weight file not found at {weight_path}")
        
        # Remove the final classification layer (fc)
        # ResNet structure: (conv1 -> bn1 -> ... -> avgpool -> fc)
        # We take everything up to avgpool.
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.features(x)         # Output: (B, 2048, 1, 1)
        x = x.view(x.size(0), -1)    # Flatten: (B, 2048)
        return x

# --- 2. Video Dataset Helper ---
class SingleVideoDataset(Dataset):
    """Reads a single video file frame-by-frame for the DataLoader"""
    def __init__(self, video_path, target_fps=1, transform=None):
        self.vr = VideoReader(video_path, ctx=cpu(0))
        self.original_fps = self.vr.get_avg_fps()
        self.transform = transform
        
        # Calculate indices to sample at target_fps
        step = max(1, int(round(self.original_fps / target_fps)))
        self.indices = list(range(0, len(self.vr), step))
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        # 1. Read Frame
        frame_idx = self.indices[idx]
        frame = self.vr[frame_idx].asnumpy() # (H, W, C)
        
        # 2. Convert to Torch & Transform
        # Decord gives (H, W, C) uint8. Transform expects (C, H, W) float/tensor.
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        
        if self.transform:
            frame = self.transform(frame)
            
        return frame

# --- 3. Main Extraction Loop ---
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Setup Model
    print(f"Loading ResNet-50 backbone on {DEVICE}...")
    model = ResNetFeatureExtractor().to(DEVICE)
    model.eval()

    # 2. Setup Transforms (Standard ImageNet Normalization)
    transform = T.Compose([
        T.Resize((224, 224), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Find Videos
    video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    print(f"Found {len(video_files)} videos. Starting extraction...")

    # 4. Process Each Video
    for vid_path in tqdm(video_files):
        video_name = os.path.splitext(os.path.basename(vid_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{video_name}.npy")
        
        # Skip if already exists
        if os.path.exists(out_path):
            continue

        try:
            # 1. Open Video (Fast Header Read)
            vr = VideoReader(vid_path, ctx=cpu(0))
            total_frames = len(vr)
            fps = vr.get_avg_fps()

            # 2. Calculate indices
            step = max(1, int(round(fps / TARGET_FPS)))
            indices = list(range(0, total_frames, step))

            video_feats = []

            # 3. Manual Batch Loop (Replaces DataLoader)
            # This runs in the MAIN thread. It cannot deadlock.
            for i in tqdm(range(0, len(indices), BATCH_SIZE)):
                batch_indices = indices[i : i + BATCH_SIZE]

                # Load raw frames
                frames = vr.get_batch(batch_indices).asnumpy() # (B, H, W, C)

                # Preprocess
                batch_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
                batch_tensor = transform(batch_tensor)

                # Inference
                batch_tensor = batch_tensor.to(DEVICE)
                with torch.no_grad():
                    feats = model(batch_tensor)
                    video_feats.append(feats.cpu().numpy())

            # 4. Save
            if len(video_feats) > 0:
                full_video_feats = np.concatenate(video_feats, axis=0)
                np.save(out_path, full_video_feats.T) # Save as (2048, T)

        except Exception as e:
            print(f"Error on {video_name}: {e}")
if __name__ == "__main__":
    main()
