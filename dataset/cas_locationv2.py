import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
from tqdm import tqdm
import cv2
import glob

# --- IMPORTS FROM SURGFORMER LIB ---
# Ensure the 'datasets' folder is in your Python path
from dataset.transforms.random_erasing import RandomErasing
import dataset.transforms.video_transforms as video_transforms
import dataset.transforms.volume_transforms as volume_transforms

def tensor_normalize(tensor, mean, std):
    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    if isinstance(mean, list): mean = torch.tensor(mean)
    if isinstance(std, list): std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor

def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames

# --- MAIN DATASET ---
class CasColonDataset(Dataset):
    def __init__(
        self,
        csv_path,
        video_root,        # Path to folder containing .mp4 files
        cache_root,        # Path where frames will be extracted (JPGs)
        clip_len=20,       # Number of frames per clip
        sampling_rate=6,   # Dilation: 16 frames cover ~1.0 second (60/4 = 15Hz effective)
        stride=60,         # Step size: Generate 1 sample every 1s (30 frames)
        mode="train",
        extracted_fps=60,  # The FPS of your source video
        crop_size=224,
        short_side_size=256,
        transition_factor = 2,
        oversampling=True,
        args=None 
    ):
        self.csv_path = csv_path
        self.video_root = video_root
        self.cache_root = cache_root
        
        # Configuration
        self.clip_len = clip_len
        self.sampling_rate = sampling_rate
        self.stride = stride
        self.mode = mode
        self.extracted_fps = extracted_fps
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.transition_factor = transition_factor
        self.oversampling = oversampling
        
        # Default Args wrapper
        class DefaultArgs:
            aa = "rand-m7-n4-mstd0.5-inc1"
            train_interpolation = "bicubic"
            reprob = 0.25
            remode = "pixel"
            recount = 1
        
        self.args = args if args else DefaultArgs()
        self.rand_erase = True if (self.mode == "train" and self.args.reprob > 0) else False

        # Classes
        self.classes = [
            'Terminal_Ileum', 'Cecum', 'Ascending_Colon', 'Hepatic_Flexure',
            'Transverse_Colon', 'Splenic_Flexure', 'Descending_Colon', 
            'Sigmoid_Colon', 'Rectum', 'Anal_Canal', 
        ]
        self.cls_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # 1. Prepare Cache (Extract frames if needed)
        self._prepare_cache()

        # 2. Build Index
        self.samples = self._make_dataset()

        # 3. Validation Transforms
        if self.mode != "train":
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(
                    size=(crop_size, crop_size),
                    interpolation="bilinear",
                )
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ])

    def _prepare_cache(self):
        """
        Scans CSV for video IDs. Checks if frames exist in cache_root.
        If not, extracts them using OpenCV.
        """
        df = pd.read_csv(self.csv_path)
        unique_videos = df['VideoID'].unique()
        
        print(f"Checking cache for {len(unique_videos)} videos in {self.cache_root}...")
        
        for vid_id in tqdm(unique_videos, desc="Caching Videos"):
            vid_id = str(vid_id)
            save_dir = os.path.join(self.cache_root, vid_id)
            
            # Simple check: Does folder exist and have content?
            if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 100:
                continue 
            
            # Attempt to find video file (try mp4, avi, etc.)
            video_path = os.path.join(self.video_root, f"{vid_id}.mp4")
            if not os.path.exists(video_path):
                # Fallback search
                candidates = glob.glob(os.path.join(self.video_root, f"{vid_id}.*"))
                if candidates: video_path = candidates[0]
                else:
                    print(f"  [Warning] Video {vid_id} not found in {self.video_root}. Skipping.")
                    continue

            # Extract
            print(f"  -> Extracting {vid_id}...")
            os.makedirs(save_dir, exist_ok=True)
            cap = cv2.VideoCapture(video_path)
            count = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # Save as 000000.jpg
                filename = os.path.join(save_dir, f"{count:06d}.jpg")
                
                # Optional: Resize to save space (e.g., 360p)
                # h, w = frame.shape[:2]
                # if h > 360:
                #    scale = 360 / h
                #    frame = cv2.resize(frame, (int(w*scale), 360))
                
                cv2.imwrite(filename, frame)
                count += 1
            cap.release()

    def _build_label_map(self, row, num_frames):
        """
        Creates a dense array of length `num_frames` where array[i] = class_index.
        Returns -1 for frames that have no label (gaps in CSV).
        """
        # Initialize with -1 (meaning "Background" or "Unlabeled")
        label_map = np.full(num_frames, -1, dtype=int)
        
        for cls_name in self.classes:
            if cls_name not in row: continue
            
            cls_idx = self.cls_to_idx[cls_name]
            intervals = self._parse_intervals(row[cls_name])
            
            for (s, e) in intervals:
                start_f = int(s * self.extracted_fps)
                end_f = int(e * self.extracted_fps)
                
                # Clip to video duration
                start_f = max(0, min(start_f, num_frames))
                end_f = max(0, min(end_f, num_frames))
                
                # Fill the map
                label_map[start_f:end_f+1] = cls_idx
                
        return label_map
    
    """def _make_dataset(self):
        samples = []
        df = pd.read_csv(self.csv_path)
        clip_span = self.clip_len * self.sampling_rate

        for _, row in df.iterrows():
            vid_id = str(row['VideoID'])
            vid_dir = os.path.join(self.cache_root, vid_id)
            if not os.path.exists(vid_dir): continue
            
            # Count frames
            frames_on_disk = len([n for n in os.listdir(vid_dir) if n.endswith('.jpg')])
            
            # --- NEW: Build the lookup map for this video ---
            label_map = self._build_label_map(row, frames_on_disk)

            for cls_name in self.classes:
                if cls_name not in row: continue
                intervals = self._parse_intervals(row[cls_name])
                
                for (s, e) in intervals:
                    start_f = int(s * self.extracted_fps)
                    end_f = int(e * self.extracted_fps)
                    
                    actual_start = max(start_f, clip_span)
                    actual_end = min(end_f, frames_on_disk)

                    if actual_end > actual_start:
                        for f in range(actual_start, actual_end, self.stride):
                            
                            # --- NEW: Look up previous label ---
                            # We look back exactly 1 stride step
                            prev_frame_idx = f - self.stride
                            
                            # Handle edge case: if start of video, use current label or -1
                            if prev_frame_idx < 0:
                                prev_label = self.cls_to_idx[cls_name]
                            else:
                                prev_label = label_map[prev_frame_idx]

                            samples.append({
                                'video_id': vid_id,
                                'frame_idx': f,
                                'label': self.cls_to_idx[cls_name],
                                'prev_label': int(prev_label), # Store it here
                                'max_frame': frames_on_disk
                            })
        cnt=0
        for x in samples:
            if x['prev_label'] == -1: 
                cnt +=1
                print(x['prev_label'])
        print(f"there are {cnt} unlabled prev frames")
        return samples"""
    
    def _make_dataset(self):
        samples = []
        df = pd.read_csv(self.csv_path)
        # clip_span is the index of the first frame we can legally sample
        clip_span = self.clip_len * self.sampling_rate

        for _, row in df.iterrows():
            vid_id = str(row['VideoID'])
            vid_dir = os.path.join(self.cache_root, vid_id)
            if not os.path.exists(vid_dir): continue

            # 1. Count Frames
            frames_on_disk = len([n for n in os.listdir(vid_dir) if n.endswith('.jpg')])

            # 2. Build Label Map (Must be continuous!)
            # Ensure _build_label_map fills gaps with a 'Background' class if needed
            label_map = self._build_label_map(row, frames_on_disk)

            # 3. Calculate Expected Count (For debugging)
            if frames_on_disk > clip_span:
                expected_range = range(clip_span, frames_on_disk, self.stride)
                expected_count = len(expected_range)
            else:
                expected_count = 0

            current_vid_samples = []
            new_expected_count = expected_count

            # 4. Sliding Window Loop (Timeline Driven)
            # strictly follows range(start, end, stride)
            for f in range(clip_span, frames_on_disk, self.stride):

                # Get label (defaults to -1 if unannotated)
                label = label_map[f]

                # Look back
                prev_idx = f - self.stride
                prev_label = label_map[prev_idx] if prev_idx >= 0 else label

                # Handle unlabeled gaps
                # If your task requires ignoring background, you might skip here,
                # BUT that will cause the assertion to fail.
                # Ideally, you assign a BACKGROUND_CLASS_ID (e.g., 0) instead of skipping.
                if label == -1: print("wrong Label") # Example: Set to background
                if prev_label == -1: print("wrong prev Label")

                sample = {
                    'video_id': vid_id,
                    'frame_idx': f,
                    'label': int(label),
                    'prev_label': int(prev_label),
                    'max_frame': frames_on_disk
                }
                current_vid_samples.append(sample)

                if self.mode == "train" and self.oversampling==True and label != prev_label:
                    dynamic_count = int(expected_count // (len(self.classes) * self.transition_factor))
                    oversample_count = max(dynamic_count, 5)
                    new_expected_count += oversample_count
                    for _ in range(oversample_count):
                        current_vid_samples.append(sample)

            # 5. Verify
            if len(current_vid_samples) != new_expected_count:
                print(f"⚠️ Video {vid_id}: Expected {new_expected_count}, Got {len(current_vid_samples)}")

            samples.extend(current_vid_samples)
        print("FINISH MAKING DATASET")

        return samples

    def _load_clip(self, video_id, target_frame_idx, max_frames):
        """
        Loads a clip of length `self.clip_len` ending at `target_frame_idx`.
        Uses `self.sampling_rate` as dilation.
        """
        frames = []
        indices = []
        
        # Calculate indices backwards from target
        for i in range(self.clip_len):
            offset = (self.clip_len - 1 - i) * self.sampling_rate
            idx = target_frame_idx - offset
            
            # Clamp to valid range
            idx = max(0, min(idx, max_frames - 1))
            indices.append(idx)
            
        for idx in indices:
            path = os.path.join(self.cache_root, video_id, f"{idx:06d}.jpg")
            try:
                # Loading implies I/O - optimization here is crucial
                with Image.open(path) as img:
                    frames.append(img.convert('RGB'))
            except Exception as e:
                # Robust fallback: Duplicate last valid frame or black frame
                if frames: frames.append(frames[-1])
                else: frames.append(Image.new('RGB', (self.short_side_size, self.short_side_size)))
        return frames

    def _aug_frame(self, buffer):
        """
        Applies AutoAugment, Normalization, Spatial Sampling, and Random Erasing.
        """
        # 1. AutoAugment (List[PIL] -> List[PIL])
        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=self.args.aa,
            interpolation=self.args.train_interpolation,
        )
        buffer = aug_transform(buffer)

        # 2. ToTensor & Stack
        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C

        # 3. Normalize
        buffer = tensor_normalize(buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # 4. Permute (C T H W) required for Spatial Sampling
        buffer = buffer.permute(3, 0, 1, 2)

        # 5. Spatial Sampling (Random Scale/Crop/Flip)
        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=True,
            inverse_uniform_sampling=False,
        )

        # 6. Random Erasing
        if self.rand_erase:
            erase_transform = RandomErasing(
                self.args.reprob,
                mode=self.args.remode,
                max_count=self.args.recount,
                num_splits=self.args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3) # (T, C, H, W)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3) # Back to (C, T, H, W)

        return buffer

    def __getitem__(self, index):
        sample = self.samples[index]
        
        # 1. Load Clip from Disk Cache
        buffer = self._load_clip(sample['video_id'], sample['frame_idx'], sample['max_frame'])
        
        # 2. Apply Transforms
        if self.mode == "train":
            buffer = self._aug_frame(buffer)
        else:
            # Validation pipeline (Deterministic)
            buffer = self.data_resize(buffer)
            buffer = self.data_transform(buffer)

        return buffer, sample['label'], sample['prev_label'], sample['video_id']

    def _parse_intervals(self, time_str):
        intervals = []
        if pd.isna(time_str) or str(time_str).strip() in ['', '-', 'nan']: return []
        segments = str(time_str).split('/')
        for seg in segments:
            try:
                if '-' not in seg: continue
                s, e = seg.strip().split('-')
                def to_s(x): return int(x.split(':')[0])*60 + int(x.split(':')[1])
                intervals.append((to_s(s), to_s(e)))
            except: pass
        return intervals
    
    def __len__(self):
        return len(self.samples)

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # Define paths
    csv_file = "/scratch/lt200353-pcllm/location/cas_colon/Video_Label.csv"
    video_folder = "/scratch/lt200353-pcllm/location/cas_colon/" 
    cache_folder = "/scratch/lt200353-pcllm/location/cas_colon/cached_images"  # Frames will be saved here automatically

    # Initialize Dataset
    dataset = CasColonDataset(
        csv_path=csv_file,
        video_root=video_folder,
        cache_root=cache_folder,
        clip_len=16,
        sampling_rate=4,  # Skip 3 frames inside clip
        stride=30,        # Skip 30 frames between samples (0.5s at 60fps)
        mode="train"
    )

    # Test Loader
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    print("Testing batch loading...")
    cnt = 0
    for imgs, labels in loader:
        if cnt ==0:
            print(f"Batch Shape: {imgs.shape}") # Should be [4, 3, 16, 224, 224]
            print(f"Labels: {labels}")
        cnt+=1
    print(f"Dataset ok: end with {cnt} batches")
