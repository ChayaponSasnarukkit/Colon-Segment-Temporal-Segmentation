import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as L
from sklearn.model_selection import train_test_split
from decord import VideoReader, cpu
import torchvision.transforms.v2 as T

class RawVideoColonDataset(Dataset):
    def __init__(self, master_csv_path, video_root, split_video_ids=None, 
                 context_length=8, downsample_factor=60, transform=None):
        """
        Args:
            downsample_factor (int): Stride for sampling. 
                                     If FPS=60 and factor=60, we sample 1 FPS.
                                     If FPS=60 and factor=6, we sample 10 FPS.
        """
        self.video_root = video_root
        self.context_length = context_length
        self.downsample_factor = downsample_factor
        self.transform = transform
        
        self.classes = [
            'Terminal_Ileum', 'Cecum', 'Ascending_Colon', 'Hepatic_Flexure',
            'Transverse_Colon', 'Splenic_Flexure', 'Descending_Colon', 
            'Sigmoid_Colon', 'Rectum', 'Anal_Canal'
        ]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = [] 
        self._load_data(master_csv_path, split_video_ids)

    def _parse_time_intervals(self, time_str, fps):
        if pd.isna(time_str) or str(time_str).strip() in ['-', 'nan', '']:
            return []

        intervals = []
        segments = str(time_str).split('/')
        
        for seg in segments:
            seg = seg.strip()
            if not seg: continue
            try:
                start_str, end_str = seg.split('-')
                
                def to_sec(t_str):
                    parts = t_str.strip().split(':')
                    return int(parts[0]) * 60 + int(parts[1])
                
                start_frame = int(to_sec(start_str) * fps)
                end_frame = int(to_sec(end_str) * fps)
                
                # Validation: End must be > Start
                if end_frame > start_frame:
                    intervals.append((start_frame, end_frame))
            except Exception:
                continue 
        return intervals

    def _load_data(self, csv_path, split_ids):
        df = pd.read_csv(csv_path)
        
        if split_ids is not None:
            split_ids = [str(i) for i in split_ids]
            df['VideoID'] = df['VideoID'].astype(str)
            df = df[df['VideoID'].isin(split_ids)]
        
        print(f"🔄 Indexing {len(df)} videos...")

        for _, row in df.iterrows():
            vid_id = row['VideoID']
            vid_path = os.path.join(self.video_root, f"{vid_id}.mp4")
            
            if not os.path.exists(vid_path):
                continue

            # 1. Quick FPS Detection (Only read header)
            try:
                # decord reads metadata very fast
                vr = VideoReader(vid_path, ctx=cpu(0))
                fps = vr.get_avg_fps()
                if fps <= 0 or np.isnan(fps): fps = 60.0
            except:
                print(f"⚠️ Error reading {vid_path}, skipping.")
                continue

            # 2. Generate Samples
            for label_name in self.classes:
                if label_name not in row: continue
                
                intervals = self._parse_time_intervals(row[label_name], fps=fps)
                label_idx = self.class_to_idx[label_name]
                
                for (start_f, end_f) in intervals:
                    # FIX: Handle short segments that are smaller than downsample_factor
                    segment_len = end_f - start_f
                    
                    if segment_len < self.downsample_factor:
                        # If segment is too short, take the center frame
                        center_frame = start_f + (segment_len // 2)
                        self.samples.append((vid_path, center_frame, label_idx))
                    else:
                        # Standard strided sampling
                        for frame_idx in range(start_f, end_f, self.downsample_factor):
                            self.samples.append((vid_path, frame_idx, label_idx))

        print(f"✅ Generated {len(self.samples)} training samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, current_idx, label_idx = self.samples[idx]
        
        # 1. Calculate Frame Indices for Context Window
        # This creates a window ending at current_idx
        # e.g., if current is 100, factor 60, len 3 -> [20, 80, 100] (previous seconds)
        indices = [
            current_idx - (self.context_length - 1 - i) * self.downsample_factor 
            for i in range(self.context_length)
        ]
        
        try:
            # 2. Open Video
            # We open fresh every time. This is IO heavy but safe for large datasets.
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            max_len = len(vr)
            
            # 3. Handle Boundary Conditions (Padding/Clamping)
            # If index < 0, clamp to 0 (repeat first frame)
            # If index > max, clamp to max (repeat last frame)
            valid_indices = [max(0, min(x, max_len - 1)) for x in indices]
            
            # 4. Fetch Frames
            buffer = vr.get_batch(valid_indices).asnumpy() # (T, H, W, C)
            
            # 5. Transform
            # Convert numpy (T, H, W, C) -> torch (T, C, H, W)
            buffer = torch.from_numpy(buffer).permute(0, 3, 1, 2).float()
            
            if self.transform:
                # Apply transform. V2 transforms expect (C, H, W) or (T, C, H, W)
                buffer = self.transform(buffer)

            # Ensure output is (C, T, H, W) for standard 3D CNNs
            # If your model expects (T, C, H, W), remove this permute
            buffer = buffer.permute(1, 0, 2, 3)
            
            return buffer, label_idx

        except Exception as e:
            print(f"❌ Error loading {video_path} frame {current_idx}: {e}")
            # Robust fallback: return zeros of expected shape
            c, h, w = 3, 224, 224 # Assume 224 default
            return torch.zeros(c, self.context_length, h, w), label_idx
        
class RawVideoDataModule(L.LightningDataModule):
    def __init__(self, master_csv_path, video_root, batch_size=8, num_workers=4, 
                 seed=42, context_length=8, downsample_factor=60):
        super().__init__()
        self.save_hyperparameters()
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        # Define transforms here or pass them in
        # Using V2 transforms is faster than PIL
        self.train_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=self.input_mean, std=self.input_std)
        ])
        
        self.val_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=self.input_mean, std=self.input_std)
        ])

    def setup(self, stage=None):
        df = pd.read_csv(self.hparams.master_csv_path)
        all_ids = df['VideoID'].unique().tolist()
        
        train_ids, temp_ids = train_test_split(all_ids, train_size=0.7, random_state=self.hparams.seed)
        val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=self.hparams.seed)
        
        if stage == 'fit' or stage is None:
            self.train_ds = RawVideoColonDataset(
                self.hparams.master_csv_path, self.hparams.video_root, 
                split_video_ids=train_ids,
                context_length=self.hparams.context_length,
                downsample_factor=self.hparams.downsample_factor,
                transform=self.train_transform
            )
            self.val_ds = RawVideoColonDataset(
                self.hparams.master_csv_path, self.hparams.video_root, 
                split_video_ids=val_ids,
                context_length=self.hparams.context_length,
                downsample_factor=self.hparams.downsample_factor,
                transform=self.val_transform
            )

        if stage == 'test':
            self.test_ds = RawVideoColonDataset(
                self.hparams.master_csv_path, self.hparams.video_root, 
                split_video_ids=test_ids,
                context_length=self.hparams.context_length,
                downsample_factor=self.hparams.downsample_factor,
                transform=self.val_transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, 
                          shuffle=True, num_workers=self.hparams.num_workers, 
                          persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, 
                          shuffle=False, num_workers=self.hparams.num_workers, 
                          persistent_workers=True, pin_memory=True)