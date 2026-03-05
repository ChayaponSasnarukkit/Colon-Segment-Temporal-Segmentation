import time
import torch
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
    
    # Separate timing accumulators
    total_time_feat = 0.0
    total_time_model = 0.0
    total_time_compress = 0.0

    print("Starting streaming inference benchmark...")

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
            if final_ctx is not None and final_ctx_mask.sum()>0:
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
                total_time_compress += compressor_latency
                #total_time_model+=additional_latency_per_minutes

            # Handle Video Boundaries
            if final_mask[0].item(): 
                inference_params.key_value_memory_dict.clear()
                inference_params.seqlen_offset = 0
                print("STARTING NEW VIDEO", flush=True)
                
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
                
                # Accumulate time (skipping the first frame for GPU warmup)
                if total_frames > 0: 
                    total_time_feat += (end_feat - start_feat)
                    total_time_model += (end_model - start_model)
                else:
                    print((end_feat - start_feat), (end_model - start_model))
                    print("IGNORE TIME OF FIRST FRAME", flush=True)
                if total_frames == 20:
                    print((end_feat - start_feat), (end_model - start_model), flush=True)
                total_frames += 1

            if batch_idx % 3 == 0 and total_frames > 1:
                # Calculate running totals
                running_time_feat = max(total_time_feat, 1e-5)
                running_time_model = max(total_time_model, 1e-5)
                running_total_time = running_time_feat + running_time_model + total_time_compress
                
                current_fps = (total_frames - 1) / running_total_time
                print(f"Processed {total_frames} frames... Running Total FPS: {current_fps:.2f}")

    # --- Final Statistics Calculation ---
    # Ensure we don't divide by zero
    valid_frames = max(total_frames - 1, 1) 
    
    avg_latency_feat_ms = (total_time_feat / valid_frames) * 1000
    avg_latency_model_ms = (total_time_model / valid_frames) * 1000
    avg_latency_compress_ms = (total_time_compress / valid_frames) * 1000
    total_latency_ms = avg_latency_feat_ms + avg_latency_model_ms + avg_latency_compress_ms
    
    avg_total_fps = 1000.0 / total_latency_ms if total_latency_ms > 0 else 0.0
    
    print("\n" + "="*50)
    print("INFERENCE BENCHMARK COMPLETE")
    print("="*50)
    print(f"Total Frames Processed: {total_frames}")
    print("-" * 50)
    print("LATENCY BREAKDOWN (per frame):")
    print(f"  Feature Extraction (DINOv3): {avg_latency_feat_ms:.2f} ms")
    print(f"  Sequence Model (Mamba):      {avg_latency_model_ms:.2f} ms")
    print(f"  Total Combined Latency:      {total_latency_ms:.2f} ms")
    print("-" * 50)
    print(f"END-TO-END FPS:                {avg_total_fps:.2f} fps")
    print("="*50)

    return avg_total_fps, avg_latency_feat_ms, avg_latency_model_ms

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
