import time
import torch
import numpy as np
from mamba_ssm.utils.generation import InferenceParams
from model.ContextMamba import ContextMambav2_for_inference
from model.CMamba import MambaTemporalSegmentation
from dataset.cas_locationv3 import MedicalStreamingDataset
from train_context_mamba_joint_v2 import MambaTemporalConfig
from torch.utils.data import DataLoader
from extract_dinov3 import DINOv3FeatureExtractor, MODEL_ID
from tqdm import tqdm

def evaluate_streaming_fps(model, dataloader, device="cuda"):
    """
    Evaluates the real-time frame-by-frame processing speed of the ContextMamba model.
    Tracks feature extraction and sequence modeling latency separately.
    Assumes dataloader batch_size is 1.
    """
    model.eval()
    model.to(device)
    
    # Setup dummy feature extractor
    dummy_model = DINOv3FeatureExtractor(MODEL_ID).to(device)
    dummy_model.eval() 
    dummy_image = torch.rand(1, 3, 224, 224, device=device)
    
    # Assign layer indices for Mamba cache
    model.assign_layer_indices()
    
    # Initialize Mamba's KV Cache equivalent
    inference_params = InferenceParams(max_seqlen=100000, max_batch_size=1)
    
    total_frames = 0
    
    # NEW: Lists to store per-frame latencies in milliseconds
    latencies_feat_ms = []
    latencies_model_ms = []
    latencies_total_ms = []

    print("Starting streaming inference benchmark...")
    video_cnt = 0
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            final_curr, final_ctx, final_lbl, final_future_lbl, final_mask, final_ctx_mask, worker_id = batch
            
            final_curr = final_curr.to(device)
            if final_ctx is not None:
                final_ctx = final_ctx.to(device)
            
            B, Chunk_Len, D = final_curr.shape
            
            if B > 1:
                print("Warning: Batch size > 1 detected. Cache resets might bleed across streams.")

            precomputed_ctx = None
            chunk_compressor_ms = 0.0 # Track amortized compressor time for this chunk
            
            if final_ctx is not None and final_ctx_mask.sum() > 0:
                # prevent weird initialize
                precomputed_ctx = model.compressor(final_ctx)
                
                # time the second time
                torch.cuda.synchronize()
                start_compress = time.perf_counter()
                precomputed_ctx = model.compressor(final_ctx)
                torch.cuda.synchronize()
                end_compress = time.perf_counter()
                
                compressor_latency = end_compress - start_compress
                print(f"Compressor Latency for chunk: {compressor_latency:.4f}s")
                
                # Amortize compressor time across all frames in this chunk (in ms)
                chunk_compressor_ms = (compressor_latency / Chunk_Len) * 1000

            # Handle Video Boundaries
            if final_mask[0].item(): 
                inference_params.key_value_memory_dict.clear()
                inference_params.seqlen_offset = 0
                print("STARTING NEW VIDEO", flush=True)
                video_cnt += 1

            if video_cnt > 3:
                break
                
            # Simulate Frame-by-Frame Real-Time Streaming
            for t in tqdm(range(Chunk_Len)):
                frame_t = final_curr[:, t:t+1, :] 
                
                # --- 1. Measure Feature Extraction Latency ---
                torch.cuda.synchronize()
                start_feat = time.perf_counter()
                
                dummy = dummy_model(dummy_image)
                
                torch.cuda.synchronize()
                end_feat = time.perf_counter()
                
                # --- 2. Measure Mamba Sequence Model Latency ---
                torch.cuda.synchronize()
                start_model = time.perf_counter()
                
                logits_wo_future, future_logits, logits_w_future, next_states = model(
                    vision_embeddings=frame_t,
                    compressed_ctx=precomputed_ctx, 
                    use_temporal_scale=True,
                    inference_params=inference_params
                )
                
                torch.cuda.synchronize()
                end_model = time.perf_counter()
                
                # Tell Mamba we moved forward 1 step
                inference_params.seqlen_offset += 1
                
                # Accumulate per-frame time (skipping the first frame for GPU warmup)
                if total_frames > 0: 
                    feat_ms = (end_feat - start_feat) * 1000
                    model_ms = (end_model - start_model) * 1000
                    total_ms = feat_ms + model_ms + chunk_compressor_ms
                    
                    latencies_feat_ms.append(feat_ms)
                    latencies_model_ms.append(model_ms)
                    latencies_total_ms.append(total_ms)
                else:
                    print(f"Warmup Frame: Feat {end_feat - start_feat:.4f}s, Model {end_model - start_model:.4f}s")
                    print("IGNORE TIME OF FIRST FRAME", flush=True)
                    
                total_frames += 1

            # Optional: Print running average FPS periodically
            if batch_idx % 3 == 0 and len(latencies_total_ms) > 0:
                current_avg_total_ms = np.mean(latencies_total_ms)
                current_fps = 1000.0 / current_avg_total_ms if current_avg_total_ms > 0 else 0
                print(f"Processed {total_frames} frames... Running Total FPS: {current_fps:.2f}")

    # --- Final Statistics Calculation ---
    if len(latencies_total_ms) == 0:
        print("No valid frames processed to calculate statistics.")
        return 0, 0, 0

    # Calculate Means
    avg_feat = np.mean(latencies_feat_ms)
    avg_model = np.mean(latencies_model_ms)
    avg_total = np.mean(latencies_total_ms)
    
    # Calculate Standard Deviations
    std_feat = np.std(latencies_feat_ms)
    std_model = np.std(latencies_model_ms)
    std_total = np.std(latencies_total_ms)
    
    avg_total_fps = 1000.0 / avg_total if avg_total > 0 else 0.0
    
    print("\n" + "="*50)
    print("INFERENCE BENCHMARK COMPLETE")
    print("="*50)
    print(f"Total Valid Frames Processed: {len(latencies_total_ms)}")
    print("-" * 50)
    print("LATENCY BREAKDOWN (per frame, mean ± std):")
    print(f"  Feature Extraction (DINOv3): {avg_feat:.2f} ± {std_feat:.2f} ms")
    print(f"  Sequence Model (Mamba):      {avg_model:.2f} ± {std_model:.2f} ms")
    print(f"  Total Combined Latency:      {avg_total:.2f} ± {std_total:.2f} ms")
    print("-" * 50)
    print(f"END-TO-END FPS:                {avg_total_fps:.2f} fps")
    print("="*50)

    return avg_total_fps, avg_feat, avg_model

def main():
    val_dataset = MedicalStreamingDataset(
        "./cv_folds_generated/fold3_test.csv", 
        "/scratch/lt200353-pcllm/location/cas_colon/features_dinov3", 
        1, 
        chunk_size=1800, 
        fps=60,            
        target_fps=30,     
        use_memory_bank=True,
        context_seconds=600, 
        context_fps=4,
        shuffle = False,
        use_emb=True,
        emb_dim=1024,
        transform=None)
    
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=1)
    config = MambaTemporalConfig(d_model=1024, n_layer=8)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100) 
    
    model = MambaTemporalSegmentation(
        config=config, 
        vision_dim=1024, 
        num_classes=10, 
        device='cuda', 
        loss_fn=loss_fn
    )
    
    full_model = ContextMambav2_for_inference(base_model=model.backbone, d_model=1024, num_classes=10, num_future=3, use_multihead=False).to('cuda')
    evaluate_streaming_fps(full_model, val_loader)

if __name__ == "__main__":
    main()
