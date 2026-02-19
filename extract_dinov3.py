import torch
import torch.nn as nn
from transformers import AutoModel
import numpy as np
import os
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# --- Configuration ---
# Point this to the parent directory containing the folders of images
# e.g., if images are in .../cas_colon/1/*.jpg, point to .../cas_colon/
IMAGE_ROOT_DIR = "/scratch/lt200353-pcllm/location/cas_colon/" 
OUTPUT_DIR = "/scratch/lt200353-pcllm/location/cas_colon/features_dinov3"

BATCH_SIZE = 256
NUM_WORKERS = 8  # This is the key to speedup
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
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image in case of error to keep batch consistent
            return torch.zeros((3, 224, 224))

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
    # Mean/Std for ImageNet (standard for ViTs)
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# --- 3. Main Loop ---
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # A. Setup Model
    extractor = DINOv3FeatureExtractor(MODEL_ID).to(DEVICE)
    
    # Use torchvision transforms (faster than HF processor)
    transform = get_transform()

    # B. Find Image Directories
    # Assuming structure: root_dir/video_name/*.jpg
    # We look for subdirectories in the root folder
    subdirs = [
        d for d in os.listdir(IMAGE_ROOT_DIR) 
        if os.path.isdir(os.path.join(IMAGE_ROOT_DIR, d)) and d.isdigit()
    ]
    
    # Sort numerically (key=int) so "2" comes before "10"
    subdirs.sort(key=int)
    
    print(f"Found {len(subdirs)} integer-named folders. Starting extraction...")
    for video_name in tqdm(subdirs):
        video_dir = os.path.join(IMAGE_ROOT_DIR, video_name)
        out_path = os.path.join(OUTPUT_DIR, f"{video_name}.npy")

        if os.path.exists(out_path):
            continue

        # C. Gather and Sort Images
        # Filter for jpg/png and SORT them to ensure temporal order
        image_files = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
        
        if len(image_files) == 0:
            continue

        # D. Create Loader with Workers
        # pin_memory=True speeds up transfer from CPU RAM to GPU RAM
        dataset = VideoFrameDataset(image_files, transform=transform)
        loader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            num_workers=NUM_WORKERS, 
            shuffle=False, 
            pin_memory=True,
            prefetch_factor=2 # Pre-loads 2 batches per worker
        )

        video_feats = []

        # E. Inference
        with torch.no_grad():
            for batch_imgs in tqdm(loader):
                batch_imgs = batch_imgs.to(DEVICE)
                
                # (B, HiddenDim)
                feats = extractor(batch_imgs)
                video_feats.append(feats.cpu())

        # F. Save Results
        if len(video_feats) > 0:
            full_video_feats = torch.cat(video_feats, dim=0).float().cpu()
            out_path_pt = os.path.join(OUTPUT_DIR, f"{video_name}.pt")
            torch.save(full_video_feats, out_path_pt)
        del video_feats
        del full_video_feats
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
