import pandas as pd
import torch
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info
import math

# Define your class map based on the CSV columns
CLASS_MAP = {
    'Terminal_Ileum': 0,
    'Cecum': 1,
    'Ascending_Colon': 2,
    'Hepatic_Flexure': 3,
    'Transverse_Colon': 4,
    'Splenic_Flexure': 5,
    'Descending_Colon': 6,
    'Sigmoid_Colon': 7,
    'Rectum': 8,
    'Anal_Canal': 9
}
NUM_CLASSES = len(CLASS_MAP)

def parse_intervals(time_str):
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

def build_dense_label(row, num_frames, fps=60):
    """
    Creates a dense array of length `num_frames` where array[i] = class_index.
    Returns -1 for frames that have no label (gaps in CSV).
    """
    # Initialize with -1 (meaning "Background" or "Unlabeled")
    label_map = np.full(num_frames, -1, dtype=int)
    
    for cls_name in CLASS_MAP:
        if cls_name not in row: continue
        
        cls_idx = CLASS_MAP[cls_name]
        intervals = parse_intervals(row[cls_name])
        
        for (s, e) in intervals:
            start_f = int(s * fps)
            end_f = int(e * fps)
            
            # Clip to video duration
            start_f = max(0, min(start_f, num_frames))
            end_f = max(0, min(end_f, num_frames))
            
            # Fill the map
            label_map[start_f:end_f+1] = cls_idx
            
    return label_map

import os
import torch
import pandas as pd
import numpy as np
import math
from torch.utils.data import IterableDataset, get_worker_info
from PIL import Image

class MedicalStreamingDataset(IterableDataset):
    def __init__(self, 
                 csv_path, 
                 video_root, 
                 batch_size_per_worker, 
                 chunk_size=1024, 
                 
                 # FPS Configuration
                 fps=60,            # Source Video FPS
                 target_fps=30,     # Desired Training FPS (New Argument)
                 
                 # Context / Memory Bank Config
                 use_memory_bank=False,
                 context_seconds=100, 
                 context_fps=5,
                 shuffle = False,

                 use_emb=True,
                 emb_dim=768,
                 transform=None):
        
        self.df = pd.read_csv(csv_path)
        self.video_root = video_root
        self.batch_size = batch_size_per_worker
        self.chunk_size = chunk_size
        self.transform = transform

        self.use_emb = use_emb
        self.emb_dim = emb_dim
        
        # --- FPS / Stride Logic ---
        self.fps = fps
        self.target_fps = target_fps
        self.step = int(fps / target_fps)  # e.g., 60/30 = 2 (Skip every other frame)
        
        if self.step < 1:
            raise ValueError("Target FPS cannot be higher than Source FPS")

        # Memory Bank Setup
        self.use_memory_bank = use_memory_bank
        self.context_seconds = context_seconds
        
        # Context stride is relative to Source FPS
        # e.g. Source 60fps, Context 5fps -> Stride 12
        self.context_stride = int(fps / context_fps) 
        self.context_len_frames = context_seconds * fps 
        self.context_num_samples = context_seconds * context_fps
        self.epoch = 0 # Initialize epoch counter
        self.shuffle = shuffle

    def _load_images(self, video_id, frame_indices):
        """ Reads a list of frame indices from disk (Images). """
        batch_images = []
        video_dir = os.path.join(self.video_root, str(video_id))
        
        for idx in frame_indices:
            # Handle negative indices (Padding)
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
        """ 
        Extracts embeddings from the pre-loaded video tensor.
        Handles negative indices by returning zero vectors.
        """
        # Create output buffer [K, emb_dim]
        out = torch.zeros(len(indices), self.emb_dim)
        video_len = full_emb_tensor.shape[0]
        
        for i, frame_idx in enumerate(indices):
            if 0 <= frame_idx < video_len:
                out[i] = full_emb_tensor[frame_idx]
        return out

    def set_epoch(self, epoch):
        """Called from your training loop to update the shuffle seed"""
        self.epoch = epoch

    def __iter__(self):
        worker_info = get_worker_info()
        
        all_indices = list(range(len(self.df)))
        
        if self.shuffle:
            # Deterministic shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            
            # Shuffle the indices using PyTorch's random permutation
            indices_perm = torch.randperm(len(all_indices), generator=g).tolist()
        else:
            indices_perm = all_indices
        
        # --- 2. Shard the Work (Split among workers) ---
        if worker_info is None:
            # Single process loading
            my_indices = indices_perm
            worker_id = 0
        else:
            # Split the SHUFFLED indices among workers
            per_worker = int(math.ceil(len(all_indices) / float(worker_info.num_workers)))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(all_indices))
            
            # Slice the shuffled list
            my_indices = indices_perm[start:end]
            worker_id = worker_info.id

        # 2. Define Stream Starter
        def start_stream(row_idx):
            row = self.df.iloc[row_idx]
            vid_id = row['VideoID']
            
            # Load Embeddings
            video_emb = None
            total_frames = 50000 
            
            if self.use_emb:
                emb_path = os.path.join(self.video_root, f"{vid_id}.pt")
                try:
                    # video_emb shape: [Source_FPS_Duration, Emb_Dim]
                    video_emb = torch.load(emb_path, map_location='cpu')
                    total_frames = video_emb.shape[0]
                except FileNotFoundError:
                    video_emb = torch.zeros(1, self.emb_dim)
                    total_frames = 1
            else:
                total_frames = int(row.get('TotalFrames', 50000)) 
            
            # Pre-calculate Labels (Full Density / Source FPS)
            # We generate labels for every frame, then sub-sample later
            dense_labels = build_dense_label(row, total_frames, self.fps)
            
            return {
                'cursor': 0,
                'total': total_frames,
                'vid_id': vid_id,
                'labels': dense_labels,
                'row_idx': row_idx,
                'embeddings': video_emb
            }

        # 3. Initialize Buffers
        active_streams = [None] * self.batch_size 
        # active stream need to be a size of batch size because each dataset instance need to yield a batch
        # This is how dataloader (with batch=None) work : 
        # 1. launch multiple process to iterate through dataset 2. each process push its batch to the ipc queue 3. the main dataloader loop pop from this queue
        # Thus, we return a whole batch from Dataset instance and use batch=None when create dataloader
        idx_queue = my_indices.copy()
        
        # batch size is number of video to stream at once
        for i in range(self.batch_size):
            if idx_queue:
                # assign stream object to active stream
                active_streams[i] = start_stream(idx_queue.pop(0)) # stream metadata

        # 4. Stream Loop
        while any(s is not None for s in active_streams):
            batch_curr = []
            batch_ctx = []
            batch_lbl = []
            reset_mask = []
            batch_ctx_mask = []

            for i in range(self.batch_size):
                stream = active_streams[i]
                
                # --- A. Handle Empty Slots ---
                if stream is None:
                    # (Padding logic same as before, ensures output shape is chunk_size)
                    if self.use_emb:
                        batch_curr.append(torch.zeros(self.chunk_size, self.emb_dim))
                        if self.use_memory_bank:
                            batch_ctx.append(torch.zeros(self.context_num_samples, self.emb_dim))
                    else:
                        batch_curr.append(torch.zeros(self.chunk_size, 3, 224, 224))
                        if self.use_memory_bank:
                            batch_ctx.append(torch.zeros(self.context_num_samples, 3, 224, 224))
                    batch_lbl.append(torch.full((self.chunk_size,), -100, dtype=torch.long))
                    reset_mask.append(True)
                    continue

                # --- B. Calculate Indices (UPDATED FOR STRIDE) ---
                curr_start = stream['cursor']
                
                # We need `chunk_size` *items*, so we span `chunk_size * step` frames
                span_needed = self.chunk_size * self.step
                curr_end = min(curr_start + span_needed, stream['total'])
                
                # Generate indices with step
                # e.g., range(0, 2048, 2) -> 0, 2, 4... (Total 1024 items)
                curr_indices = list(range(curr_start, curr_end, self.step))
                
                # Context Indices (Remains relative to source time)
                # 1. Determine the "ideal" start point
                #    If current is 100, context 50s (250 frames), we want [-150 to 100]
                ctx_stop = curr_start
                ctx_start_ideal = ctx_stop - (self.context_num_samples * self.context_stride)
                
                # 2. Generate indices, but clamp to 0 (valid data only)
                #    We only want indices that actually exist in the video
                raw_ctx_indices = range(ctx_start_ideal, ctx_stop, self.context_stride)
                valid_ctx_indices = [idx for idx in raw_ctx_indices if idx >= 0]
                
                # 3. Calculate how much padding we need
                num_valid = len(valid_ctx_indices)
                num_pad = self.context_num_samples - num_valid
                
                # --- C. Load Data ---
                curr_tensor = None
                ctx_tensor = None
                ctx_mask = None 

                # Load Main Chunk (Same as before)
                if self.use_emb:
                    curr_tensor = self._get_embeddings(stream['embeddings'], curr_indices)
                else:
                    curr_tensor = self._load_images(stream['vid_id'], curr_indices)

                # --- LOAD CONTEXT (Right Padded) ---
                if self.use_memory_bank:
                    # A. Load only the VALID frames
                    if self.use_emb:
                        valid_ctx_tensor = self._get_embeddings(stream['embeddings'], valid_ctx_indices)
                    else:
                        valid_ctx_tensor = self._load_images(stream['vid_id'], valid_ctx_indices)
                    
                    # B. Create Right Padding (Zeros)
                    if num_pad > 0:
                        if self.use_emb:
                            pad_tensor = torch.zeros(num_pad, self.emb_dim)
                        else:
                            pad_tensor = torch.zeros(num_pad, 3, 224, 224)
                        
                        # C. Concatenate: [Valid_Data, Zeros] -> Right Padding
                        ctx_tensor = torch.cat([valid_ctx_tensor, pad_tensor], dim=0)
                        
                        # D. Create Mask: [1, 1, ... 0, 0]
                        # 1 for valid, 0 for pad
                        ctx_mask = torch.cat([
                            torch.ones(num_valid, dtype=torch.bool),
                            torch.zeros(num_pad, dtype=torch.bool)
                        ])
                    else:
                        # No padding needed
                        ctx_tensor = valid_ctx_tensor
                        ctx_mask = torch.ones(num_valid, dtype=torch.bool)

                # Load Labels (Slice using the same stride)
                # Note: stream['labels'] is dense (60fps). 
                # We extract specific indices manually to match the data skip.
                # Efficient way to index a tensor with a list
                lbl_tensor = stream['labels'][curr_indices] 

                # --- D. Right Padding (If near end of video) ---
                actual_sampled_len = len(curr_indices)
                
                if actual_sampled_len < self.chunk_size:
                    pad_len = self.chunk_size - actual_sampled_len
                    
                    if self.use_emb:
                        curr_pad = torch.zeros(pad_len, self.emb_dim)
                    else:
                        curr_pad = torch.zeros(pad_len, 3, 224, 224)
                    curr_tensor = torch.cat([curr_tensor, curr_pad], dim=0)
                    
                    lbl_pad = torch.full((pad_len,), -100, dtype=torch.long)
                    lbl_tensor = torch.cat([lbl_tensor, lbl_pad], dim=0)

                batch_curr.append(curr_tensor)
                if self.use_memory_bank:
                    batch_ctx.append(ctx_tensor)
                    batch_ctx_mask.append(ctx_mask)
                batch_lbl.append(lbl_tensor)
                
                # --- E. Advance Stream ---
                reset_mask.append(curr_start == 0)
                
                # Advance cursor by the RAW SPAN covered (not the sampled length)
                # e.g., we read 1024 frames but moved 2048 frames forward in time
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
            
            yield final_curr, final_ctx, final_lbl, final_mask, final_ctx_mask, worker_id

# MedicalStreamingDataset(
#     "/scratch/lt200353-pcllm/location/cas_colon/updated_Video_Label.csv", 
#     "/scratch/lt200353-pcllm/location/cas_colon/features_dinov3", 
#     2, 
#     chunk_size=1024, 
    
#     # FPS Configuration
#     fps=60,            # Source Video FPS
#     target_fps=30,     # Desired Training FPS (New Argument)
    
#     # Context / Memory Bank Config
#     use_memory_bank=True,
#     context_seconds=300, 
#     context_fps=1,
#     shuffle = True,

#     use_emb=True,
#     emb_dim=1024,
#     transform=None)



# for final_curr, final_ctx, final_lbl, final_mask, final_ctx_mask, worker_id in iter(myd2):
#      if cnt%10:
#              print(final_ctx.shape, final_ctx_mask.sum(), final_mask)
#      cnt+=1

# cnt=0
# for final_curr, final_ctx, final_lbl, final_mask, final_ctx_mask, worker_id in iter(myd2):
#      cnt+=1


# dl = DataLoader(myd2, batch_size=None, num_workers=2)
# for i in range(5):
#      myd2.set_epoch(i)
#      dl = DataLoader(myd2, batch_size=None, num_workers=2)
#      cnt = 0
#      for x in tqdm(dl):
#              cnt+=1
#      print(len(myd2), cnt)