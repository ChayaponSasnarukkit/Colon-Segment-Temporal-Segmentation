import os
import torch
import pandas as pd
import numpy as np
import math
from torch.utils.data import IterableDataset, get_worker_info
import os.path as osp

# 50salads Class Map (derived from mapping.txt)
LABEL_MAP = {
    "cut_tomato": 0,
    "place_tomato_into_bowl": 1,
    "cut_cheese": 2,
    "place_cheese_into_bowl": 3,
    "cut_lettuce": 4,
    "place_lettuce_into_bowl": 5,
    "add_salt": 6,
    "add_vinegar": 7,
    "add_oil": 8,
    "add_pepper": 9,
    "mix_dressing": 10,
    "peel_cucumber": 11,
    "cut_cucumber": 12,
    "place_cucumber_into_bowl": 13,
    "add_dressing": 14,
    "mix_ingredients": 15,
    "serve_salad_onto_plate": 16,
    "action_start": 17,
    "action_end": 18
}
NUM_CLASSES = len(LABEL_MAP)

class FiftySaladsStreamingDataset(IterableDataset):
    def __init__(self, 
                 data_root, 
                 batch_size_per_worker, 
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
                 emb_dim=2048): 
        
        self.data_root = data_root
        self.features_dir = osp.join(data_root, 'features')
        self.gt_dir = osp.join(data_root, 'groundTruth')
        
        self.batch_size = batch_size_per_worker
        self.chunk_size = chunk_size
        self.emb_dim = emb_dim
        
        # --- Split Logic ---
        self.split_dir = split_dir
        self.fold = fold
        self.phase = phase
        
        # --- Read Text Files for Splits (UPDATED FOR .bundle) ---
        self.sessions = []
        if self.phase == 'train':
            split_files = [f'train.split{self.fold}.bundle']
        else:
            split_files = [f'test.split{self.fold}.bundle']

        if self.split_dir:
            for file_name in split_files:
                file_path = osp.join(self.split_dir, file_name)
                if osp.exists(file_path):
                    with open(file_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                # The bundle files contain lines like 'rgb-01-1.txt'
                                # We remove the '.txt' to just store the raw video ID
                                vid_id = line.replace('.txt', '')
                                self.sessions.append(vid_id)
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
        if not osp.exists(self.features_dir):
            raise FileNotFoundError(f"Features directory not found: {self.features_dir}")
            
        npy_files = [f for f in os.listdir(self.features_dir) if f.endswith('.npy')]
        npy_files.sort()

        for npy_file in npy_files:
            vid_id = npy_file.replace('.npy', '')

            # --- Filter by Split ---
            if self.sessions and vid_id not in self.sessions:
                continue

            lbl_file = f"{vid_id}.txt"
            lbl_path = osp.join(self.gt_dir, lbl_file)
            npy_path = osp.join(self.features_dir, npy_file)

            if osp.exists(lbl_path):
                # Retrieve sequence length from the .npy feature matrix
                feat_array = np.load(npy_path, mmap_mode='r')
                
                # Check orientation: If shape is (C, T) which is standard for MS-TCN, grab dim 1.
                if feat_array.shape[0] == self.emb_dim:
                    total_frames = feat_array.shape[1]
                else:
                    total_frames = feat_array.shape[0]

                data.append({'VideoID': vid_id, 'TotalFrames': total_frames})

        if not data:
            raise ValueError(f"No valid .npy and .txt pairs found for fold {self.fold} ({self.phase}) in {self.data_root}")

        return pd.DataFrame(data)

    def _get_embeddings(self, full_emb_tensor, indices):
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
            
            span_needed = self.chunk_size * self.step
            
            # Temporal Jitter Logic
            start_cursor = 0
            if self.phase == 'train' and total_frames > span_needed:
                start_cursor = int(torch.randint(0, span_needed, (1,)).item())
            
            # 1. Load Embeddings (.npy)
            emb_path = osp.join(self.features_dir, f"{vid_id}.npy")
            try:
                feat = np.load(emb_path)
                # Ensure format is (Time, Features)
                if feat.shape[0] == self.emb_dim:
                    feat = feat.T
                    
                video_emb = torch.from_numpy(feat).float()
                total_frames = min(total_frames, video_emb.shape[0])
            except FileNotFoundError:
                video_emb = torch.zeros(total_frames, self.emb_dim)
            
            # 2. Load Labels (from .txt line-by-line)
            lbl_path = osp.join(self.gt_dir, f"{vid_id}.txt")
            dense_labels = []
            
            with open(lbl_path, 'r') as f:
                for line in f:
                    lbl_str = line.strip()
                    if lbl_str:
                        dense_labels.append(LABEL_MAP.get(lbl_str, -100))
                        
            # Truncate/pad to match features length
            dense_labels = torch.tensor(dense_labels[:total_frames], dtype=torch.long)
            
            return {
                'cursor': start_cursor,         
                'is_first_chunk': True,         
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
                    batch_curr.append(torch.zeros(self.chunk_size, self.emb_dim))
                    if self.use_memory_bank:
                        batch_ctx.append(torch.zeros(self.context_num_samples, self.emb_dim))
                            
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
                curr_tensor = self._get_embeddings(stream['embeddings'], curr_indices)
                ctx_tensor, ctx_mask = None, None 

                # Context Padding Logic
                if self.use_memory_bank:
                    valid_ctx_tensor = self._get_embeddings(stream['embeddings'], valid_ctx_indices)
                    
                    if num_pad > 0:
                        pad_shape = (num_pad, self.emb_dim)
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
                    
                    pad_shape = (pad_len, self.emb_dim)
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
                reset_mask.append(stream.get('is_first_chunk', False)) 
                stream['is_first_chunk'] = False 
                
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