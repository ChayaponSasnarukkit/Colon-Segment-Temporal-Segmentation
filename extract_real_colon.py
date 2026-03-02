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
# Update to your real_colon directory structure
DATA_ROOT = "/scratch/lt200353-pcllm/location/real_colon"
LABEL_DIR = os.path.join(DATA_ROOT, "label")
OUTPUT_DIR = os.path.join(DATA_ROOT, "features_dinov3")

BATCH_SIZE = 256
NUM_WORKERS = 8  # This is the key to speedup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
CLEANED_LABEL_DIR = os.path.join(DATA_ROOT, "labels_cleaned")

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
class DINOv3FeatureExtractor(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        print(f"Loading DINOv3 model: {model_id} ...")
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        # Take CLS token
        return outputs.last_hidden_state[:, 0, :]

def get_transform():
    # Replicates DINOv3 preprocessing using fast Torchvision transforms
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# --- 3. Main Loop ---
def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if not os.path.exists(CLEANED_LABEL_DIR): os.makedirs(CLEANED_LABEL_DIR)

    extractor = DINOv3FeatureExtractor(MODEL_ID).to(DEVICE)
    transform = get_transform()

    label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.csv")))

    for label_path in tqdm(label_files, desc="Processing Videos"):
        video_name = os.path.basename(label_path).replace(".csv", "")
        video_dir = os.path.join(DATA_ROOT, f"{video_name}_frames")
        out_path_pt = os.path.join(OUTPUT_DIR, f"{video_name}.pt")
        cleaned_csv_path = os.path.join(CLEANED_LABEL_DIR, f"{video_name}.csv")

        if os.path.exists(out_path_pt) and os.path.exists(cleaned_csv_path):
            continue

        # --- C. Gather Images & Filter CSV Rows ---
        image_files = []
        valid_rows = [] # To store CSV metadata for existing images
        
        with open(label_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames # Keep original headers
            for row in reader:
                frame_name = row['frame_filename']
                img_path = os.path.join(video_dir, frame_name)
                
                if os.path.exists(img_path):
                    image_files.append(img_path)
                    valid_rows.append(row)
                else:
                    # Log the specific missing frame for debugging
                    print(f"\n[Skipping] Missing: {img_path}")

        if len(image_files) == 0:
            continue

        # --- D & E. Loader and Inference (Same as your original) ---
        dataset = VideoFrameDataset(image_files, transform=transform)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
        
        video_feats = []
        with torch.no_grad():
            for batch_imgs in tqdm(loader):
                feats = extractor(batch_imgs.to(DEVICE))
                video_feats.append(feats.cpu())

        # --- F. Save Synchronized Results ---
        if len(video_feats) > 0:
            # 1. Save Features
            full_video_feats = torch.cat(video_feats, dim=0).float()
            torch.save(full_video_feats, out_path_pt)
            
            # 2. Save Cleaned CSV (Guaranteed to match .pt length)
            with open(cleaned_csv_path, mode='w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(valid_rows)

            # Safety check
            assert len(full_video_feats) == len(valid_rows), \
                f"Mismatch in {video_name}: {len(full_video_feats)} feats vs {len(valid_rows)} labels"

        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
