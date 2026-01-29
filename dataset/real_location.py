import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import re

class RealColonCausalDataset(Dataset):
    def __init__(self, root_dir, label_dir=None, context_length=5, downsample_factor=1, transform=None):
        """
        Args:
            root_dir (str): Dataset root.
            context_length (int): Number of frames in the context window.
            downsample_factor (int): 
                - 1 = Use original FPS (every frame is a target, context is consecutive).
                - 60 = Simulate 1 FPS (if orig is 60fps). 
                       Only every 60th frame is used as a target.
                       Context frames are also spaced by 60 (t, t-60, t-120...).
            transform (callable): Transform to apply to individual frames.
        """
        self.root_dir = root_dir
        self.label_dir = label_dir if label_dir else os.path.join(root_dir, 'label')
        self.context_length = context_length
        self.downsample_factor = downsample_factor
        self.transform = transform

        self.video_data = {} 
        self.samples = []    
        self.class_to_idx = {}
        
        self._load_data()

    def _natural_key(self, text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

    def _load_data(self):
        """
        NOTE: The label csv files need to be sorted by frame number before using this dataset
        Register the following object:
            (1) self.video_data[video_id]: Contain ALL video frame's path{
                    'paths': paths, # list of path follow the order of row in csv file 
                    'labels': labels, # list of label follow the order of row in csv file (image at paths[i] has label of labels[i])
                    'length': len(paths) # length of the video
                }
            (2) self.samples: a list of tuple (video_id, index) where index is index in paths 
                                (register wrt. downsample_factor, which mean for video i that have L frame only L/downsample_factor frame of it will present it self.samples)
            ** self.samples is the final produce that getitem will itered from. The look back will be implemented in getitem**
        """
        csv_files = glob.glob(os.path.join(self.label_dir, "*.csv"))
        all_labels = set()
        print(f"Found {len(csv_files)} annotation files. Parsing with downsample_factor={self.downsample_factor}...")

        for csv_path in csv_files:
            video_id = os.path.splitext(os.path.basename(csv_path))[0]
            df = pd.read_csv(csv_path)
            
            # Sort naturally
            df['sort_key'] = df['frame_filename'].apply(self._natural_key)
            df = df.sort_values('sort_key').drop(columns=['sort_key'])
            
            paths = []
            labels = []
            
            frame_folder = os.path.join(self.root_dir, f"{video_id}_frames")
            if not os.path.isdir(frame_folder):
                continue

            for _, row in df.iterrows():
                paths.append(os.path.join(frame_folder, row['frame_filename']))
                labels.append(row['GT'])
                all_labels.add(row['GT'])
            
            self.video_data[video_id] = {
                'paths': paths,
                'labels': labels,
                'length': len(paths)
            }
            
            # --- Sparse Sampling ---
            # Instead of adding every index, we skip by downsample_factor.
            # This drastically reduces the dataset size (len(self.samples)).
            for i in range(0, len(paths), self.downsample_factor):
                self.samples.append((video_id, i))

        self.classes = sorted(list(all_labels))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        print(f"Classes: {self.class_to_idx}")
        print(f"Total samples after downsampling: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        NOTE: The label csv files need to be sorted by frame number before using this dataset
        """
        video_id, current_idx = self.samples[idx]
        video_info = self.video_data[video_id]
        seq_len = video_info['length']
        
        # --- Strided Context Window ---
        # We need 'context_length' frames.
        # The stride between frames is 'downsample_factor'.
        # Example: L=3, factor=10, current=100.
        # Indices needed: [80, 90, 100]
        
        # Since the video_info contain whole video's frame, we need to build the list of indices manually to handle the stride.
        indices = []
        for i in range(self.context_length):
            # We look back: (L-1) steps, (L-2) steps ... 0 steps
            step_back = (self.context_length - 1 - i) * self.downsample_factor
            frame_idx = current_idx - step_back
            
            # Padding by repetition/Clamping: If index < 0, use 0. If > seq_len, use last.
            clamped_idx = max(0, min(frame_idx, seq_len - 1))
            indices.append(clamped_idx)
            
        frames = []
        for i in indices:
            img_path = video_info['paths'][i]
            try:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                frames.append(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                frames.append(torch.zeros(3, 224, 224)) 

        # Stack: (T, C, H, W)
        img_tensor = torch.stack(frames)
        
        # --- Channel First Output ---
        # Permute to (C, T, H, W) for direct model compatibility
        img_tensor = img_tensor.permute(1, 0, 2, 3)
        
        label_indices = []
        for i in indices:
            # indices contains [t-4, t-3, ... t]
            lbl_str = video_info['labels'][i]
            lbl_idx = self.class_to_idx[lbl_str]
            label_indices.append(lbl_idx)
            
        # Convert to tensor: Shape (T,)
        labels_tensor = torch.tensor(label_indices, dtype=torch.long)
        
        return img_tensor, labels_tensor