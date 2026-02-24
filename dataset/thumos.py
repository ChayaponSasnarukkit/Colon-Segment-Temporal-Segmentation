import torch
import numpy as np
import os.path as osp
import math
from torch.utils.data import IterableDataset, get_worker_info

class ThumosStreamingDataset(IterableDataset):
    def __init__(self, 
                 data_root, 
                 json_data, 
                 batch_size_per_worker, 
                 
                 # Streaming Config
                 chunk_size=240,       
                 
                 # Feature Config
                 feature_folder_rgb="rgb",
                 feature_folder_flow="flow",
                 fps=4.0,              # Source FPS of the extracted features
                 target_fps=4.0,       # Desired Training FPS
                 
                 # --- Context Config (MATCHING MEDICAL DATASET) ---
                 use_memory_bank=True,
                 context_seconds=100,  # Physical time duration of context
                 context_fps=1,        # Sampling rate of context (1 fps = 1 frame every second)
                 
                 # Future Config
                 num_future=3,         # Number of predictions
                 future_step=4,        # Stride for future (4 frames @ 4fps = 1 second)

                 phase='train',
                 shuffle=True,
                 return_indices=True,
                 ):
        
        self.data_root = data_root
        self.sessions = json_data[phase + '_session_set']
        self.ignore_index = json_data.get('ignore_index', -100)
        self.phase = phase
        self.batch_size = batch_size_per_worker
        self.chunk_size = chunk_size 
        
        self.feature_folder_rgb = feature_folder_rgb
        self.feature_folder_flow = feature_folder_flow
        
        # --- FPS / Stride Logic ---
        self.fps = fps
        self.target_fps = target_fps
        self.step = int(fps / target_fps)
        if self.step < 1: self.step = 1

        # --- Context Logic (Replicated from MedicalStreamingDataset) ---
        self.use_memory_bank = use_memory_bank
        self.context_seconds = context_seconds
        self.context_fps = context_fps
        
        # Calculate Stride relative to Source Feature FPS
        # e.g., Features 4fps, Context 1fps -> Stride 4 (Skip 3 frames)
        self.context_stride = int(fps / context_fps) 
        if self.context_stride < 1: self.context_stride = 1
            
        # Total number of context vectors to return
        self.context_num_samples = int(context_seconds * context_fps)
        
        self.num_future = num_future
        self.future_step = future_step 
        
        self.shuffle = shuffle
        self.return_indices = return_indices
        self.epoch = 0
        
        # Detect Dimension strictly from first file to avoid hardcoding
        self.feature_dim = self._detect_feature_dim()

    def _detect_feature_dim(self):
        """Helper to safely get feature dim from the first available file"""
        if len(self.sessions) == 0: return 4096
        try:
            feats, _, _ = self._load_video_data(self.sessions[0])
            return feats.shape[1]
        except:
            return 4096

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _load_video_data(self, session_name):
        # (Same loading logic as before)
        rgb_path = osp.join(self.data_root, self.feature_folder_rgb, session_name + '.npy')
        flow_path = osp.join(self.data_root, self.feature_folder_flow, session_name + '.npy')
        
        try:
            rgb = np.load(rgb_path)
            flow = np.load(flow_path)
            features = np.concatenate([rgb, flow], axis=-1)
            features = torch.from_numpy(features).float()
        except (FileNotFoundError, ValueError):
            return torch.zeros(1, self.feature_dim), torch.zeros(1).long(), 1

        target_path = osp.join(self.data_root, 'targets', session_name + '.npy')
        if osp.exists(target_path):
            targets = np.load(target_path)
            if self.return_indices:
                if targets.ndim > 1:
                    row_sums = targets.sum(axis=-1)
                    target_indices = np.argmax(targets, axis=-1)
                    target_indices[row_sums == 0] = self.ignore_index
                    targets = torch.from_numpy(target_indices).long()
                else:
                    targets = torch.from_numpy(targets).long()
            else:
                targets = torch.from_numpy(targets).float()
        else:
            targets = torch.zeros(features.shape[0]).long() if self.return_indices else torch.zeros(features.shape[0], 22)

        # Sync lengths
        total_frames = features.shape[0]
        if targets.shape[0] > total_frames:
            targets = targets[:total_frames]
        elif targets.shape[0] < total_frames:
            pad_len = total_frames - targets.shape[0]
            if self.return_indices:
                pad = torch.full((pad_len,), self.ignore_index, dtype=torch.long)
                targets = torch.cat([targets, pad], dim=0)
            else:
                pad = torch.zeros((pad_len, targets.shape[1]), dtype=torch.float)
                targets = torch.cat([targets, pad], dim=0)

        return features, targets, total_frames

    def __len__(self):
        # Approximate length for tqdm
        # Assuming avg 2000 frames per video
        total_frames = len(self.sessions) * 2000 
        total_steps = total_frames // (self.chunk_size * self.step)
        return max(1, total_steps // self.batch_size)

    def __iter__(self):
        worker_info = get_worker_info()
        all_indices = list(range(len(self.sessions)))
        
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

        def start_stream(session_idx):
            session_name = self.sessions[session_idx]
            feats, lbls, total = self._load_video_data(session_name)
            return {'cursor': 0, 'total': total, 'features': feats, 'labels': lbls}

        active_streams = [None] * self.batch_size
        idx_queue = my_indices.copy()

        for i in range(self.batch_size):
            if idx_queue:
                active_streams[i] = start_stream(idx_queue.pop(0))

        while any(s is not None for s in active_streams):
            batch_curr, batch_ctx, batch_lbl, batch_fut = [], [], [], []
            reset_mask, batch_ctx_mask = [], []

            for i in range(self.batch_size):
                stream = active_streams[i]

                # --- Handle Empty Stream (Padding) ---
                if stream is None:
                    batch_curr.append(torch.zeros(self.chunk_size, self.feature_dim))
                    
                    if self.use_memory_bank:
                        batch_ctx.append(torch.zeros(self.context_num_samples, self.feature_dim))
                        batch_ctx_mask.append(torch.zeros(self.context_num_samples).bool())
                    
                    if self.return_indices:
                        batch_lbl.append(torch.full((self.chunk_size,), self.ignore_index, dtype=torch.long))
                        batch_fut.append(torch.full((self.chunk_size, self.num_future), self.ignore_index, dtype=torch.long))
                    else:
                        batch_lbl.append(torch.zeros(self.chunk_size, 22))
                        batch_fut.append(torch.zeros(self.chunk_size, self.num_future, 22))
                    
                    reset_mask.append(True)
                    continue

                # --- Indices ---
                curr_start = stream['cursor']
                span_needed = self.chunk_size * self.step
                curr_end = min(curr_start + span_needed, stream['total'])
                curr_indices = list(range(curr_start, curr_end, self.step))

                # --- 1. Current ---
                curr_tensor = stream['features'][curr_indices]
                lbl_tensor = stream['labels'][curr_indices]

                # --- 2. Context (EXACT MEDICAL LOGIC) ---
                if self.use_memory_bank:
                    # Logic: 
                    # 1. Determine "ideal" start based on time
                    # 2. Slice raw range
                    # 3. Filter for valid (>=0)
                    # 4. Right Pad to reach fixed length
                    
                    ctx_stop = curr_start
                    # "context_num_samples" is fixed length we promised the model
                    # "context_stride" is how many frames we skip
                    ctx_start_ideal = ctx_stop - (self.context_num_samples * self.context_stride)
                    
                    raw_ctx_indices = range(ctx_start_ideal, ctx_stop, self.context_stride)
                    valid_ctx_indices = [idx for idx in raw_ctx_indices if idx >= 0]
                    
                    # Fetch Valid Data
                    # Note: We can slice directly because features are already in memory
                    valid_ctx = stream['features'][valid_ctx_indices]
                    
                    # Calculate Padding
                    num_valid = valid_ctx.shape[0]
                    num_pad = self.context_num_samples - num_valid
                    
                    if num_pad > 0:
                        # Right Padding (zeros)
                        pad = torch.zeros(num_pad, self.feature_dim) 
                        ctx_tensor = torch.cat([valid_ctx, pad], dim=0)
                        
                        # Mask (1=Valid, 0=Pad)
                        ctx_mask = torch.cat([torch.ones(num_valid), torch.zeros(num_pad)]).bool()
                    else:
                        ctx_tensor = valid_ctx
                        ctx_mask = torch.ones(self.context_num_samples).bool()

                    batch_ctx.append(ctx_tensor)
                    batch_ctx_mask.append(ctx_mask)

                # --- 3. Future ---
                future_tensor_list = []
                for t in curr_indices:
                    f_indices = [t + (f * self.future_step) for f in range(1, self.num_future + 1)]
                    valid_f = [min(x, stream['total'] - 1) for x in f_indices]
                    future_tensor_list.append(stream['labels'][valid_f])
                
                batch_fut_tensor = torch.stack(future_tensor_list)

                # --- 4. End Padding (Chunk Alignment) ---
                actual_len = curr_tensor.shape[0]
                if actual_len < self.chunk_size:
                    pad_len = self.chunk_size - actual_len
                    p_feat = torch.zeros(pad_len, self.feature_dim)
                    curr_tensor = torch.cat([curr_tensor, p_feat], dim=0)
                    
                    if self.return_indices:
                        p_lbl = torch.full((pad_len,), self.ignore_index, dtype=torch.long)
                        p_fut = torch.full((pad_len, self.num_future), self.ignore_index, dtype=torch.long)
                    else:
                        p_lbl = torch.zeros((pad_len, lbl_tensor.shape[1]))
                        p_fut = torch.zeros((pad_len, self.num_future, lbl_tensor.shape[1]))
                        
                    lbl_tensor = torch.cat([lbl_tensor, p_lbl], dim=0)
                    batch_fut_tensor = torch.cat([batch_fut_tensor, p_fut], dim=0)

                batch_curr.append(curr_tensor)
                batch_lbl.append(lbl_tensor)
                batch_fut.append(batch_fut_tensor)
                reset_mask.append(curr_start == 0)

                # Advance
                stream['cursor'] += span_needed
                if stream['cursor'] >= stream['total']:
                    if idx_queue:
                        active_streams[i] = start_stream(idx_queue.pop(0))
                    else:
                        active_streams[i] = None

            final_curr = torch.stack(batch_curr)
            final_lbl = torch.stack(batch_lbl)
            final_fut = torch.stack(batch_fut)
            final_mask = torch.tensor(reset_mask)
            final_ctx = torch.stack(batch_ctx) if self.use_memory_bank else None
            final_ctx_mask = torch.stack(batch_ctx_mask) if self.use_memory_bank else None
            
            yield final_curr, final_ctx, final_lbl, final_fut, final_mask, final_ctx_mask, worker_id