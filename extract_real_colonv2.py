import torch
import torch.nn as nn
from transformers import AutoModel
import numpy as np
import os
import glob
import csv
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# --- Configuration ---
DATA_ROOT = "/scratch/lt200353-pcllm/location/real_colon/dataset"
LABEL_DIR = os.path.join(DATA_ROOT, "labels") # Fixed based on your 'ls' output
VIDEO_INFO_PATH = os.path.join(DATA_ROOT, "video_info.csv") 
OUTPUT_DIR = os.path.join(DATA_ROOT, "features_dinov3")

BATCH_SIZE = 256
NUM_WORKERS = 8  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m" 

# --- 1. Define Dataset ---
class VideoFrameDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# --- 2. Define Backbone ---
class DINOFeatureExtractor(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        print(f"Loading model: {model_id} ...")
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        # Take CLS token
        return outputs.last_hidden_state[:, 0, :]

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# --- 3. Main Loop ---
def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # --- Load Video FPS Mapping ---
    video_fps_dict = {}
    if os.path.exists(VIDEO_INFO_PATH):
        print(f"Loading video metadata from: {VIDEO_INFO_PATH}")
        with open(VIDEO_INFO_PATH, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_fps_dict[row['unique_video_name']] = float(row['fps'])
    else:
        print(f"Warning: Could not find {VIDEO_INFO_PATH}. Defaulting to 29.97 fps fallback.")

    extractor = DINOFeatureExtractor(MODEL_ID).to(DEVICE)
    transform = get_transform()

    label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.csv")))

    for label_path in tqdm(label_files, desc="Processing Videos"):
        video_name = os.path.basename(label_path).replace(".csv", "")
        video_dir = os.path.join(DATA_ROOT, f"{video_name}_frames")
        
        out_path_pt = os.path.join(OUTPUT_DIR, f"{video_name}.pt")
        out_path_labels = os.path.join(OUTPUT_DIR, f"{video_name}_labels.npy")

        if os.path.exists(out_path_pt) and os.path.exists(out_path_labels):
            continue

        # --- Calculate Step Size for 5 FPS ---
        original_fps = video_fps_dict.get(video_name, 29.97)
        step_size = max(1, round(original_fps / 5.0)) 
        
        # --- Gather Images & Labels ---
        image_files = []
        valid_labels = [] 
        print("...extracted valid files")
        with open(label_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            frame_idx = 0
            for row in reader:
                # Handle potential empty rows or headers if they exist
                if not row or row[0] == "frame_filename": 
                    continue
                
                # Apply 5 fps sampling
                if frame_idx % step_size == 0:
                    frame_name = row[0]
                    label = row[1]
                    img_path = os.path.join(video_dir, frame_name)
                    
                    if os.path.exists(img_path):
                        image_files.append(img_path)
                        valid_labels.append(label)
                
                frame_idx += 1

        if len(image_files) == 0:
            print(f"No valid frames found for {video_name}. Skipping.")
            continue

        # --- Loader and Inference ---
        dataset = VideoFrameDataset(image_files, transform=transform)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
        
        video_feats = []
        with torch.no_grad():
            for batch_imgs in tqdm(loader, desc=f"Extracting {video_name}", leave=False):
                feats = extractor(batch_imgs.to(DEVICE))
                video_feats.append(feats.cpu())

        # --- Save Synchronized Results ---
        if len(video_feats) > 0:
            # 1. Save Features
            full_video_feats = torch.cat(video_feats, dim=0).float()
            torch.save(full_video_feats, out_path_pt)
            
            # 2. Save Label Numpy Array
            labels_array = np.array(valid_labels)
            np.save(out_path_labels, labels_array)

            # Safety check
            assert len(full_video_feats) == len(labels_array), \
                f"Mismatch in {video_name}: {len(full_video_feats)} feats vs {len(labels_array)} labels"

        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
