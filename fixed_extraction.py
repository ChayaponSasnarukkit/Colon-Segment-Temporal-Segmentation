import torch
import torch.nn as nn
from transformers import AutoModel
import numpy as np
import os
import csv
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# --- Configuration ---
DATA_ROOT = "/scratch/lt200353-pcllm/location/real_colon/dataset"
LABEL_DIR = os.path.join(DATA_ROOT, "labels") 
OUTPUT_DIR = os.path.join(DATA_ROOT, "features_dinov3")

TARGET_VIDEO = "001-012"
ORIGINAL_FPS = 60.0 # We know this is 60 fps from your video_info.csv

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
        return outputs.last_hidden_state[:, 0, :]

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# --- 3. Main Logic ---
def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    extractor = DINOFeatureExtractor(MODEL_ID).to(DEVICE)
    transform = get_transform()

    video_dir = os.path.join(DATA_ROOT, f"{TARGET_VIDEO}_frames")
    label_path = os.path.join(LABEL_DIR, f"{TARGET_VIDEO}.csv")
    out_path_pt = os.path.join(OUTPUT_DIR, f"{TARGET_VIDEO}.pt")
    out_path_labels = os.path.join(OUTPUT_DIR, f"{TARGET_VIDEO}_labels.npy")

    print(f"Processing target video: {TARGET_VIDEO}")

    # Calculate Step Size for 5 FPS
    step_size = max(1, round(ORIGINAL_FPS / 5.0)) 
    print(f"Original FPS: {ORIGINAL_FPS} -> Step size: {step_size}")
    
    image_files = []
    valid_labels = [] 
    
    with open(label_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        frame_idx = 0
        for row in reader:
            if not row or row[0] == "frame_filename": 
                continue
            
            # Apply 5 fps sampling
            if frame_idx % step_size == 0:
                frame_name = row[0]
                label = row[1]
                img_path = os.path.join(video_dir, frame_name)
                
                # --- THE FIX: Handle .0.jpg mismatch ---
                if not os.path.exists(img_path) and ".jpg" in frame_name:
                    alt_frame_name = frame_name.replace(".jpg", ".0.jpg")
                    alt_img_path = os.path.join(video_dir, alt_frame_name)
                    if os.path.exists(alt_img_path):
                        img_path = alt_img_path # Update to the actual file path

                if os.path.exists(img_path):
                    image_files.append(img_path)
                    valid_labels.append(label)
                else:
                    # If it STILL doesn't exist, we print it out to debug
                    pass
            
            frame_idx += 1

    if len(image_files) == 0:
        print(f"❌ Still no valid frames found for {TARGET_VIDEO}. Check filenames.")
        return

    print(f"Found {len(image_files)} valid frames for {TARGET_VIDEO}. Extracting features...")

    # --- Loader and Inference ---
    dataset = VideoFrameDataset(image_files, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    
    video_feats = []
    with torch.no_grad():
        for batch_imgs in tqdm(loader, desc=f"Extracting {TARGET_VIDEO}"):
            feats = extractor(batch_imgs.to(DEVICE))
            video_feats.append(feats.cpu())

    # --- Save Synchronized Results ---
    if len(video_feats) > 0:
        full_video_feats = torch.cat(video_feats, dim=0).float()
        torch.save(full_video_feats, out_path_pt)
        
        labels_array = np.array(valid_labels)
        np.save(out_path_labels, labels_array)

        assert len(full_video_feats) == len(labels_array), \
            f"Mismatch: {len(full_video_feats)} feats vs {len(labels_array)} labels"

        print(f"✅ Successfully extracted {TARGET_VIDEO} ({len(full_video_feats)} frames)")

if __name__ == "__main__":
    main()
