import os
import torch
import pandas as pd
import numpy as np
import math
from torch.utils.data import IterableDataset, get_worker_info
from PIL import Image

# Updated RealColon Class Map
LABEL_MAP = {
    "outside": 0,
    "insertion": 1,
    "ceacum": 2,
    "ileum": 3,
    "ascending": 4,
    "transverse": 5,
    "descending": 6,
    "sigmoid": 7,
    "rectum": 8,
    # "uncertain": -100,
}
NUM_CLASSES = len(LABEL_MAP)
import os.path as osp
class RealColonStreamingDataset(IterableDataset):
    def __init__(self, 
                 video_root, 
                 batch_size_per_worker, 
                 
                 # --- NEW: Split Configuration ---
                 split_dir=None,
                 fold=1,
                 phase='train',
                 
                 chunk_size=1024, 
                 fps=5,             
                 target_fps=5,      
                 
                 use_memory_bank=False,
                 context_seconds=600, 
                 context_fps=5,     
                 shuffle=False,
                 num_future_seconds=3,

                 use_emb=True,
                 emb_dim=768,
                 transform=None):
        
        self.video_root = video_root
        self.batch_size = batch_size_per_worker
        self.chunk_size = chunk_size
        self.transform = transform

        self.use_emb = use_emb
        self.emb_dim = emb_dim
        
        # --- NEW: Split Logic ---
        self.split_dir = split_dir
        self.fold = fold
        self.phase = phase
        # --- Read Text Files for Splits (Integrated Logic) ---
        self.sessions = []
        if self.phase == 'train':
            # Combine train and valid for training, ignoring validation phase entirely
            split_files = [f'fold{self.fold}_train.txt', f'fold{self.fold}_valid.txt']
        else:
            # Strictly use test set for evaluation
            split_files = [f'fold{self.fold}_test.txt']

        if self.split_dir:
            for file_name in split_files:
                file_path = osp.join(self.split_dir, file_name)
                if osp.exists(file_path):
                    with open(file_path, 'r') as f:
                        # Read lines, strip newline characters, and ignore empty lines
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                        self.sessions.extend(lines)
                else:
                    print(f"Warning: Split file not found: {file_path}")

        # --- Auto-discover dataset ---
        self.df = self._build_dataset_dataframe()
        
        # --- FPS / Stride Logic ---
        self.fps = fps
        self.target_fps = target_fps
        self.step = int(fps / target_fps)  
        
        if self.step < 1:
            raise ValueError("Target FPS cannot be higher than Source FPS")

        # Memory Bank Setup
        self.use_memory_bank = use_memory_bank
        self.context_seconds = context_seconds
        self.context_stride = max(1, int(fps / context_fps)) 
        self.context_len_frames = context_seconds * fps 
        self.context_num_samples = context_seconds * context_fps
        self.epoch = 0 
        self.shuffle = shuffle

        self.num_future = num_future_seconds
        self.future_offsets = torch.arange(1, self.num_future + 1) * fps

    def _build_dataset_dataframe(self):
        data = []
        files = os.listdir(self.video_root)

        pt_files = [f for f in files if f.endswith('.pt')]
        pt_files.sort()

        for pt_file in pt_files:
            vid_id = pt_file.replace('.pt', '')

            # --- Filter by Split ---
            # If self.sessions is populated, skip videos not in the target split
            if self.sessions and vid_id not in self.sessions:
                continue

            lbl_file = f"{vid_id}_labels.npy"
            lbl_path = osp.join(self.video_root, lbl_file)

            if osp.exists(lbl_path):
                lbl_array = np.load(lbl_path, mmap_mode='r')
                total_frames = lbl_array.shape[0]
                data.append({'VideoID': vid_id, 'TotalFrames': total_frames})

        if not data:
            raise ValueError(f"No valid .pt and _labels.npy pairs found for fold {self.fold} ({self.phase}) in {self.video_root}")

        return pd.DataFrame(data)

    def _load_images(self, video_id, frame_indices):
        """ Reads a list of frame indices from disk (Images). """
        batch_images = []
        video_dir = os.path.join(self.video_root, str(video_id))
        
        for idx in frame_indices:
            if idx < 0:
                batch_images.append(torch.zeros(3, 224, 224))
                continue
                
            img_path = os.path.join(video_dir, f"{idx}.jpg")
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    else:
                        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                    batch_images.append(img)
            except (FileNotFoundError, OSError):
                batch_images.append(torch.zeros(3, 224, 224))

        if len(batch_images) == 0:
            return torch.zeros(len(frame_indices), 3, 224, 224)
            
        return torch.stack(batch_images)

    def _get_embeddings(self, full_emb_tensor, indices):
        """ Extracts embeddings from the pre-loaded video tensor. """
        out = torch.zeros(len(indices), self.emb_dim)
        video_len = full_emb_tensor.shape[0]
        
        for i, frame_idx in enumerate(indices):
            if 0 <= frame_idx < video_len:
                out[i] = full_emb_tensor[frame_idx]
        return out

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        span_needed = self.chunk_size * self.step
        all_indices = list(range(len(self.df)))

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices_perm = torch.randperm(len(all_indices), generator=g).tolist()
        else:
            indices_perm = all_indices

        idx_queue = indices_perm.copy()
        active_chunks_left = [0] * self.batch_size

        def get_chunks(row_idx):
            frames = self.df.iloc[row_idx]['TotalFrames']
            if frames <= 0: return 0
            return math.ceil(frames / span_needed)

        for i in range(self.batch_size):
            if idx_queue:
                active_chunks_left[i] = get_chunks(idx_queue.pop(0))

        total_batches = 0
        while any(chunks > 0 for chunks in active_chunks_left):
            total_batches += 1
            for i in range(self.batch_size):
                if active_chunks_left[i] > 0:
                    active_chunks_left[i] -= 1
                    if active_chunks_left[i] == 0 and idx_queue:
                        active_chunks_left[i] = get_chunks(idx_queue.pop(0))

        return total_batches

    def __iter__(self):
        worker_info = get_worker_info()
        all_indices = list(range(len(self.df)))
        
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices_perm = torch.randperm(len(all_indices), generator=g).tolist()
        else:
            indices_perm = all_indices
        
        if worker_info is None:
            my_indices = indices_perm
            worker_id = 0
        else:
            per_worker = int(math.ceil(len(all_indices) / float(worker_info.num_workers)))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(all_indices))
            my_indices = indices_perm[start:end]
            worker_id = worker_info.id

        def start_stream(row_idx):
            row = self.df.iloc[row_idx]
            vid_id = row['VideoID']
            total_frames = row['TotalFrames']
            
            # 1. Load Embeddings
            video_emb = None
            if self.use_emb:
                emb_path = os.path.join(self.video_root, f"{vid_id}.pt")
                try:
                    video_emb = torch.load(emb_path, map_location='cpu')
                    # Ensure frames match the embedding shape just in case
                    total_frames = min(total_frames, video_emb.shape[0])
                except FileNotFoundError:
                    video_emb = torch.zeros(total_frames, self.emb_dim)
            
            # 2. Load Labels
            lbl_path = os.path.join(self.video_root, f"{vid_id}_labels.npy")
            raw_labels = np.load(lbl_path)
            
            dense_labels = []
            for lbl in raw_labels:
                # Handle potential byte strings (e.g. b'ceacum')
                lbl_str = lbl.decode('utf-8') if isinstance(lbl, bytes) else str(lbl)
                lbl_str = lbl_str.strip()
                # Map to int, default to 9 ("uncertain") or -100 (ignore index) if not found
                dense_labels.append(LABEL_MAP.get(lbl_str, -100))
                
            dense_labels = torch.tensor(dense_labels[:total_frames], dtype=torch.long)
            
            return {
                'cursor': 0,
                'total': total_frames,
                'vid_id': vid_id,
                'labels': dense_labels,
                'row_idx': row_idx,
                'embeddings': video_emb
            }

        active_streams = [None] * self.batch_size 
        idx_queue = my_indices.copy()
        
        for i in range(self.batch_size):
            if idx_queue:
                active_streams[i] = start_stream(idx_queue.pop(0)) 

        while any(s is not None for s in active_streams):
            batch_curr, batch_ctx, batch_lbl, batch_future_lbl = [], [], [], []
            reset_mask, batch_ctx_mask = [], []

            for i in range(self.batch_size):
                stream = active_streams[i]
                
                # --- A. Handle Empty Slots ---
                if stream is None:
                    if self.use_emb:
                        batch_curr.append(torch.zeros(self.chunk_size, self.emb_dim))
                        if self.use_memory_bank:
                            batch_ctx.append(torch.zeros(self.context_num_samples, self.emb_dim))
                    else:
                        batch_curr.append(torch.zeros(self.chunk_size, 3, 224, 224))
                        if self.use_memory_bank:
                            batch_ctx.append(torch.zeros(self.context_num_samples, 3, 224, 224))
                            
                    batch_lbl.append(torch.full((self.chunk_size,), -100, dtype=torch.long))
                    batch_future_lbl.append(torch.full((self.chunk_size, self.num_future), -100, dtype=torch.long))
                    reset_mask.append(True)
                    if self.use_memory_bank:
                        batch_ctx_mask.append(torch.zeros(self.context_num_samples, dtype=torch.bool))
                    continue

                # --- B. Calculate Indices ---
                curr_start = stream['cursor']
                span_needed = self.chunk_size * self.step
                curr_end = min(curr_start + span_needed, stream['total'])
                
                curr_indices = list(range(curr_start, curr_end, self.step))
                
                ctx_stop = curr_start
                ctx_start_ideal = ctx_stop - (self.context_num_samples * self.context_stride)
                
                raw_ctx_indices = range(ctx_start_ideal, ctx_stop, self.context_stride)
                valid_ctx_indices = [idx for idx in raw_ctx_indices if idx >= 0]
                
                num_valid = len(valid_ctx_indices)
                num_pad = self.context_num_samples - num_valid
                
                # --- C. Load Data ---
                curr_tensor, ctx_tensor, ctx_mask = None, None, None 

                if self.use_emb:
                    curr_tensor = self._get_embeddings(stream['embeddings'], curr_indices)
                else:
                    curr_tensor = self._load_images(stream['vid_id'], curr_indices)

                # Context Padding Logic
                if self.use_memory_bank:
                    if self.use_emb:
                        valid_ctx_tensor = self._get_embeddings(stream['embeddings'], valid_ctx_indices)
                    else:
                        valid_ctx_tensor = self._load_images(stream['vid_id'], valid_ctx_indices)
                    
                    if num_pad > 0:
                        pad_shape = (num_pad, self.emb_dim) if self.use_emb else (num_pad, 3, 224, 224)
                        pad_tensor = torch.zeros(pad_shape)
                        
                        ctx_tensor = torch.cat([valid_ctx_tensor, pad_tensor], dim=0)
                        ctx_mask = torch.cat([
                            torch.ones(num_valid, dtype=torch.bool),
                            torch.zeros(num_pad, dtype=torch.bool)
                        ])
                    else:
                        ctx_tensor = valid_ctx_tensor
                        ctx_mask = torch.ones(num_valid, dtype=torch.bool)

                # Labels
                lbl_tensor = stream['labels'][curr_indices] 

                # Future Labels
                curr_idx_tensor = torch.tensor(curr_indices).unsqueeze(1) 
                offsets_tensor = self.future_offsets.unsqueeze(0)
                future_indices_tensor = curr_idx_tensor + offsets_tensor
                
                valid_future_mask = future_indices_tensor < stream['total']
                future_lbl_tensor = torch.full((len(curr_indices), self.num_future), -100, dtype=torch.long)
                
                valid_f_idx = future_indices_tensor[valid_future_mask]
                future_lbl_tensor[valid_future_mask] = stream['labels'][valid_f_idx]

                # --- D. Right Padding for main sequence ---
                actual_sampled_len = len(curr_indices)
                if actual_sampled_len < self.chunk_size:
                    pad_len = self.chunk_size - actual_sampled_len
                    
                    pad_shape = (pad_len, self.emb_dim) if self.use_emb else (pad_len, 3, 224, 224)
                    curr_pad = torch.zeros(pad_shape)
                    curr_tensor = torch.cat([curr_tensor, curr_pad], dim=0)
                    
                    lbl_pad = torch.full((pad_len,), -100, dtype=torch.long)
                    lbl_tensor = torch.cat([lbl_tensor, lbl_pad], dim=0)

                    lbl_pad_future = torch.full((pad_len, self.num_future), -100, dtype=torch.long)
                    future_lbl_tensor = torch.cat([future_lbl_tensor, lbl_pad_future], dim=0)

                batch_curr.append(curr_tensor)
                if self.use_memory_bank:
                    batch_ctx.append(ctx_tensor)
                    batch_ctx_mask.append(ctx_mask)
                batch_lbl.append(lbl_tensor)
                batch_future_lbl.append(future_lbl_tensor)
                
                # --- E. Advance Stream ---
                reset_mask.append(curr_start == 0)
                stream['cursor'] += span_needed
                
                if stream['cursor'] >= stream['total']:
                    if idx_queue:
                        active_streams[i] = start_stream(idx_queue.pop(0))
                    else:
                        active_streams[i] = None

            # --- Yield Batch ---
            final_curr = torch.stack(batch_curr)
            final_lbl = torch.stack(batch_lbl)
            final_mask = torch.tensor(reset_mask)
            final_ctx = torch.stack(batch_ctx) if self.use_memory_bank else None
            final_ctx_mask = torch.stack(batch_ctx_mask) if self.use_memory_bank else None
            final_future_lbl = torch.stack(batch_future_lbl)
            
            yield final_curr, final_ctx, final_lbl, final_future_lbl, final_mask, final_ctx_mask, worker_id
