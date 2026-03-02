import os
import torch
import pandas as pd
import numpy as np
import math
from torch.utils.data import IterableDataset, get_worker_info

# --- 1. Define your new Class Map ---
# ['outside', 'insertion', 'ceacum', 'ileum', 'ascending',
#        'transverse', 'descending', 'sigmoid', 'rectum']
CLASS_MAP = {
    'outside': 0,
    'inside': 1, 
    'ceacum': 2,
    'ileum': 3,
    'ascending': 4,
    'transverse': 5,
    'descending': 6,
    'sigmoid': 7, 'rectum': 8,
}
NUM_CLASSES = len([k for k in CLASS_MAP if CLASS_MAP[k] >= 0])

class MedicalStreamingDataset(IterableDataset):
    def __init__(self, 
                 label_files,       
                 video_root, 
                 metadata_csv,      
                 batch_size_per_worker, 
                 chunk_size=1024, 
                 target_fps=25,     # The universal FPS we want to sample our data at
                 
                 use_memory_bank=False,
                 context_seconds=100, 
                 context_fps=5,
                 shuffle=False,
                 num_future_seconds=3,

                 use_emb=True,      # Only support use_emb=True for now
                 emb_dim=1024
                 ):
        
        self.label_files = sorted(label_files)
        self.video_root = video_root
        self.batch_size = batch_size_per_worker
        self.chunk_size = chunk_size
        self.use_memory_bank = use_memory_bank

        self.use_emb = use_emb
        self.emb_dim = emb_dim
        
        # Extract video IDs from the CSV filenames
        self.video_ids = [os.path.basename(f).replace(".csv", "") for f in self.label_files]
        if len(self.video_ids) == 0:
            raise ValueError("The provided list of label files is empty.")

        # --- Load Metadata & Calculate Lengths ---
        meta_df = pd.read_csv(metadata_csv)
        self.fps_dict = meta_df.set_index('unique_video_name')['fps'].to_dict()

        # Count the number of label rows (frames) for each video
        self.video_lengths = []
        for lf in self.label_files:
            with open(lf, 'r', encoding='utf-8') as f:
                self.video_lengths.append(sum(1 for _ in f) - 1) # -1 to ignore the header

        self.target_fps = float(target_fps)
        self.context_fps = float(context_fps)
        self.context_seconds = context_seconds
        self.context_num_samples = int(context_seconds * context_fps)
        self.num_future = num_future_seconds
        self.epoch = 0 
        self.shuffle = shuffle
        
        # --- Calculate Total Target Frames ---
        # Because source videos have different FPS, we map their total source frames 
        # to how many frames they *would* have at our universal target_fps.
        self.target_lengths = []
        for i, vid in enumerate(self.video_ids):
            src_fps = float(self.fps_dict.get(vid, 25.0))
            src_total = self.video_lengths[i]
            
            ratio = src_fps / self.target_fps
            target_total = math.floor(src_total / ratio)
            self.target_lengths.append(target_total)

    def _get_embeddings(self, full_emb_tensor, indices):
        """Safely extracts embeddings given a list of indices, returning zeros for out-of-bounds."""
        out = torch.zeros(len(indices), self.emb_dim)
        video_len = full_emb_tensor.shape[0]
        
        for i, frame_idx in enumerate(indices):
            if 0 <= frame_idx < video_len:
                out[i] = full_emb_tensor[frame_idx]
        return out

    def set_epoch(self, epoch):
        # Crucial for distributed training to ensure different shuffles per epoch
        self.epoch = epoch

    def __len__(self):
        """
        Estimates the total number of batches in the dataset.
        Simulates the streaming process to count exactly how many chunks will be yielded.
        """
        span_needed = self.chunk_size
        all_indices = list(range(len(self.video_ids)))

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices_perm = torch.randperm(len(all_indices), generator=g).tolist()
        else:
            indices_perm = all_indices

        idx_queue = indices_perm.copy()
        active_chunks_left = [0] * self.batch_size

        def get_chunks(row_idx):
            frames = self.target_lengths[row_idx]
            if frames <= 0: return 0
            return math.ceil(frames / span_needed)

        # Initialize the first set of active streams
        for i in range(self.batch_size):
            if idx_queue:
                active_chunks_left[i] = get_chunks(idx_queue.pop(0))

        # Run the simulation
        total_batches = 0
        while any(chunks > 0 for chunks in active_chunks_left):
            total_batches += 1
            for i in range(self.batch_size):
                if active_chunks_left[i] > 0:
                    active_chunks_left[i] -= 1
                    # If a stream finishes, load the next video from the queue
                    if active_chunks_left[i] == 0 and idx_queue:
                        active_chunks_left[i] = get_chunks(idx_queue.pop(0))

        return total_batches

    def __iter__(self):
        worker_info = get_worker_info()
        all_indices = list(range(len(self.video_ids)))
        
        # 1. Handle Shuffling based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices_perm = torch.randperm(len(all_indices), generator=g).tolist()
        else:
            indices_perm = all_indices
        
        # 2. Handle Multi-Processing (DataLoader workers)
        # Split the video list evenly among active workers to avoid duplicate data
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
            """Helper function to load a new video into an active stream slot."""
            vid_id = self.video_ids[row_idx]
            label_path = self.label_files[row_idx]
            src_total = self.video_lengths[row_idx]
            target_total = self.target_lengths[row_idx]
            src_fps = float(self.fps_dict.get(vid_id, 25.0))
            
            # Load embeddings
            video_emb = None
            if self.use_emb:
                emb_path = os.path.join(self.video_root, f"{vid_id}.pt") 
                try:
                    video_emb = torch.load(emb_path, map_location='cpu')
                    # Guard against embedding files being shorter than the label file
                    src_total = min(src_total, video_emb.shape[0]) 
                except FileNotFoundError:
                    # Fallback if missing
                    video_emb = torch.zeros(src_total, self.emb_dim)

            # Load and map labels
            df = pd.read_csv(label_path)
            # Inside start_stream(row_idx)
            cleaned_labels = df['GT'].astype(str).str.strip().str.lower()
            dense_labels_np = cleaned_labels.map(CLASS_MAP).fillna(-100).values
            
            # Align label lengths with embedding lengths
            if len(dense_labels_np) > src_total:
                dense_labels_np = dense_labels_np[:src_total]
            elif len(dense_labels_np) < src_total:
                pad = np.full(src_total - len(dense_labels_np), -100)
                dense_labels_np = np.concatenate([dense_labels_np, pad])

            dense_labels = torch.from_numpy(dense_labels_np).long()            
            
            # Return a "stream state" dictionary
            return {
                'cursor': 0,                   # Tracks how far we are in this video (in target frames)
                'target_total': target_total,
                'src_total': src_total,
                'src_fps': src_fps,
                'vid_id': vid_id,
                'labels': dense_labels,
                'embeddings': video_emb
            }

        # 3. Initialize the stream buffers
        # We process 'batch_size' videos concurrently. 
        active_streams = [None] * self.batch_size 
        idx_queue = my_indices.copy()
        
        for i in range(self.batch_size):
            if idx_queue:
                active_streams[i] = start_stream(idx_queue.pop(0))

        # 4. Main Streaming Loop
        # Keeps yielding batches until all active streams are empty (None)
        while any(s is not None for s in active_streams):
            batch_curr, batch_ctx, batch_lbl, batch_future_lbl, reset_mask, batch_ctx_mask = [], [], [], [], [], []

            for i in range(self.batch_size):
                stream = active_streams[i]
                
                # If this specific stream slot is empty, yield blank padded tensors for this batch element
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

                curr_start = stream['cursor']
                src_fps = stream['src_fps']
                src_total = stream['src_total']
                
                # --- A. Time Generation & Alignment ---
                # Determine the target frame indices for this chunk
                curr_end = min(curr_start + self.chunk_size, stream['target_total'])
                target_indices = np.arange(curr_start, curr_end)
                
                # Convert target frames to seconds, then map to the source video's actual FPS
                curr_times_sec = target_indices / self.target_fps
                curr_indices_np = np.round(curr_times_sec * src_fps).astype(int)
                curr_indices = np.clip(curr_indices_np, 0, src_total - 1).tolist()
                
                # --- B. Context (Memory Bank) Processing ---
                curr_tensor, ctx_tensor, ctx_mask = None, None, None 
                
                if self.use_memory_bank:
                    # Look backward in time from the start of the current chunk
                    ctx_end_time = curr_times_sec[0] if len(curr_times_sec) > 0 else 0.0
                    ctx_times_sec = ctx_end_time - np.arange(self.context_num_samples, 0, -1) / self.context_fps
                    raw_ctx_indices = np.round(ctx_times_sec * src_fps).astype(int)
                    
                    # Filter out indices before the video started (negative time)
                    valid_mask = raw_ctx_indices >= 0
                    valid_ctx_indices = raw_ctx_indices[valid_mask].tolist()
                    
                    num_valid = len(valid_ctx_indices)
                    num_pad = self.context_num_samples - num_valid
                    
                    # Safely extract embeddings (or setup dummy images if use_emb=False)
                    if self.use_emb:
                        valid_ctx_tensor = self._get_embeddings(stream['embeddings'], valid_ctx_indices)
                    else:
                        valid_ctx_tensor = torch.zeros(len(valid_ctx_indices), 3, 224, 224)
                    
                    # If we don't have enough past context (e.g., at the start of a video), zero-pad it
                    if num_pad > 0:
                        pad_tensor = torch.zeros(num_pad, self.emb_dim) if self.use_emb else torch.zeros(num_pad, 3, 224, 224)
                        ctx_tensor = torch.cat([valid_ctx_tensor, pad_tensor], dim=0) # Right padding
                        ctx_mask = torch.cat([torch.ones(num_valid, dtype=torch.bool), torch.zeros(num_pad, dtype=torch.bool)])
                    else:
                        ctx_tensor = valid_ctx_tensor
                        ctx_mask = torch.ones(self.context_num_samples, dtype=torch.bool)

                # --- C. Load Main Data & Current Labels ---
                if self.use_emb:
                    curr_tensor = self._get_embeddings(stream['embeddings'], curr_indices)
                else:
                    curr_tensor = torch.zeros(len(curr_indices), 3, 224, 224)

                lbl_tensor = stream['labels'][curr_indices] 

                # --- D. Future Labels Extraction ---
                # Look forward by 'num_future' seconds for each frame in the current chunk
                future_offsets_sec = np.arange(1, self.num_future + 1)
                future_times_sec = curr_times_sec[:, None] + future_offsets_sec[None, :]
                future_indices_np = np.round(future_times_sec * src_fps).astype(int)
                
                # Check which future times actually exist within the video
                valid_future_mask = future_indices_np < src_total
                future_lbl_tensor = torch.full((len(curr_indices), self.num_future), -100, dtype=torch.long)
                
                # Assign actual future labels where valid (leaves out-of-bounds as -100)
                valid_f_idx = future_indices_np[valid_future_mask]
                future_lbl_tensor[torch.from_numpy(valid_future_mask)] = stream['labels'][valid_f_idx]

                # --- E. Right Chunk Padding (End of Video) ---
                # If we are at the end of a video and the remaining frames don't fill a whole chunk, pad the rest.
                actual_sampled_len = len(curr_indices)
                if actual_sampled_len < self.chunk_size:
                    pad_len = self.chunk_size - actual_sampled_len
                    curr_pad = torch.zeros(pad_len, self.emb_dim) if self.use_emb else torch.zeros(pad_len, 3, 224, 224)
                    curr_tensor = torch.cat([curr_tensor, curr_pad], dim=0)
                    
                    lbl_pad = torch.full((pad_len,), -100, dtype=torch.long)
                    lbl_tensor = torch.cat([lbl_tensor, lbl_pad], dim=0)
                    
                    lbl_pad_future = torch.full((pad_len, self.num_future), -100, dtype=torch.long)
                    future_lbl_tensor = torch.cat([future_lbl_tensor, lbl_pad_future], dim=0)

                # Append to the batch lists
                batch_curr.append(curr_tensor)
                if self.use_memory_bank:
                    batch_ctx.append(ctx_tensor)
                    batch_ctx_mask.append(ctx_mask)
                batch_lbl.append(lbl_tensor)
                batch_future_lbl.append(future_lbl_tensor)
                
                # reset_mask is True for the first chunk of a video (tells RNNs/Transformers to reset hidden states)
                reset_mask.append(curr_start == 0)
                
                # --- F. Advance Cursor & Stream Replacement ---
                # Move the cursor forward by the chunk size
                stream['cursor'] += self.chunk_size
                
                # If the video is finished, pop the next video from the queue into this slot
                if stream['cursor'] >= stream['target_total']:
                    if idx_queue:
                        active_streams[i] = start_stream(idx_queue.pop(0))
                    else:
                        active_streams[i] = None

            # 5. Stack the batch lists into PyTorch tensors and Yield
            final_curr = torch.stack(batch_curr)
            final_lbl = torch.stack(batch_lbl)
            final_mask = torch.tensor(reset_mask)
            final_ctx = torch.stack(batch_ctx) if self.use_memory_bank else None
            final_ctx_mask = torch.stack(batch_ctx_mask) if self.use_memory_bank else None
            final_future_lbl = torch.stack(batch_future_lbl)
            
            yield final_curr, final_ctx, final_lbl, final_future_lbl, final_mask, final_ctx_mask, worker_id