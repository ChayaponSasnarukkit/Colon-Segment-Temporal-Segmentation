import torch
import numpy as np
from mamba_ssm.utils.generation import InferenceParams
from model.ContextMamba import ContextMambav2_for_inference
from model.CMamba import MambaTemporalSegmentation
from dataset.cas_locationv3 import MedicalStreamingDataset
from train_cas_colon_mid3 import MambaTemporalConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

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

@torch.no_grad()
def evaluate_streaming_performance(model, dataloader, device="cuda"):
    """
    Evaluates the frame-by-frame streaming performance of the ContextMamba model.
    Proves that continuous cache passing matches chunk-based sequence modeling.
    """
    model.eval()
    model.to(device)
    
    # REQUIRED: Assign layer indices so Mamba layers don't overwrite each other's cache
    model.assign_layer_indices()
    
    # Initialize Mamba's global state cache
    inference_params = InferenceParams(max_seqlen=100000, max_batch_size=1)
    
    all_preds = []
    all_labels = []

    print("Starting exact streaming inference evaluation...")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Streaming Videos")):
        final_curr, final_ctx, final_lbl, final_future_lbl, final_mask, final_ctx_mask, worker_id = batch
        
        final_curr = final_curr.to(device)
        final_lbl = final_lbl.to(device)
        if final_ctx is not None:
            final_ctx = final_ctx.to(device)
        
        B, Chunk_Len, D = final_curr.shape
        
        if B > 1:
            raise ValueError("Streaming inference requires batch_size=1 to maintain continuous state.")

        # Handle Video Boundaries (Reset cache for new videos)
        if final_mask[0].item(): 
            inference_params.key_value_memory_dict.clear()
            inference_params.seqlen_offset = 0
            
        precomputed_ctx = None
        if final_ctx is not None and final_ctx_mask.sum() > 0:
            # Compress the historical context once per chunk (Amortization)
            precomputed_ctx = model.compressor(final_ctx)

        # ---------------------------------------------------------
        # THE STREAMING LOOP: Feed 1 frame at a time
        # ---------------------------------------------------------
        for t in range(Chunk_Len):
            frame_t = final_curr[:, t:t+1, :] 
            label_t = final_lbl[:, t].item()
            
            # Pass `t` as the chunk_step!
            _, _, logits_w_future, _ = model(
                vision_embeddings=frame_t,
                compressed_ctx=precomputed_ctx, 
                use_temporal_scale=True,
                inference_params=inference_params,
                chunk_step=t  # <--- NEW
            )
            
            # Tick the GLOBAL clock forward
            inference_params.seqlen_offset += 1
            
            # Extract prediction
            pred_t = torch.argmax(logits_w_future, dim=-1).item()
            
            all_preds.append(pred_t)
            all_labels.append(label_t)

    # --- Calculate Final Metrics ---
    if len(all_labels) == 0:
        print("No valid labels found for evaluation.")
        return
        
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print("\n" + "="*50)
    print("STREAMING INFERENCE RESULTS")
    print("="*50)
    print(f"Total Valid Frames Evaluated: {len(all_labels)}")
    print(f"Accuracy: {val_acc * 100:.2f}%")
    print(f"Macro F1: {val_f1_macro * 100:.2f}%")
    print("="*50)

def main():
    # --- 1. Parameters (Must match training exactly) ---
    WEIGHT_PATH = "/project/lt200353-pcllm/3d_report_gen/cas_colon/mid_a4tune3i411_3_full_shuffle/fold1/lr5e5_b4.pth" # Replace with your actual path
    
    # --- 2. Dataset Setup ---
    # Standard contiguous dataset (Not the test overlapping one) to prove exact match
    val_dataset = MedicalStreamingDataset(
        "./cv_folds_generated/fold1_test.csv", # Make sure this fold matches your weight path
        "/project/lt200353-pcllm/3d_report_gen/cas_colon/features_dinov3/", 
        1, 
        chunk_size=1800, 
        fps=60,            
        target_fps=30,     
        use_memory_bank=True,
        context_seconds=600, 
        context_fps=4,
        shuffle=False,
        use_emb=True,
        emb_dim=1024,
        transform=None
    )
    
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=1)
    
    # --- 3. Model Architecture Setup ---
    config = MambaTemporalConfig(d_model=1024, n_layer=8)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100) 
    
    base_model = MambaTemporalSegmentation(
        config=config, 
        vision_dim=1024, 
        num_classes=10, 
        device='cuda', 
        loss_fn=loss_fn
    )
    
    # Make sure these params strictly match your training ContextMambav2!
    full_model = ContextMambav2_for_inference(
        base_model=base_model.backbone, 
        d_model=1024, 
        num_classes=10, 
        num_future=3, 
        compression_ratio=240.0,
        target_fps=30.0,
        context_fps=4.0,
        query_fps=30.0,
        use_multihead=True
    ).to('cuda')

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # num_action_classes = len(CLASS_MAP)
    # config = MambaTemporalConfig(d_model=1024, n_layer=8)
    # loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100) 
    
    # model = MambaTemporalSegmentation(config=config, vision_dim=1024, num_classes=num_action_classes, device=device, loss_fn=loss_fn)
    # cfg_fps = hparams.get("fps", 60)
    # cfg_target_fps = hparams.get("target_fps", 30)
    # cfg_context_fps = hparams.get("context_fps", 4)
    # cfg_query_fps = hparams.get("query_fps", 30)
    # cfg_compression_ratio = hparams.get("compression_ratio", 240.0)
    # cfg_frames_per_query = hparams.get("frames_per_query", [24, 10])
    # cfg_vbatch = hparams.get("vbatch", 4)
    # # Initialize the Large RealColon Head but with CAS temporal parameters
    # full_model = ContextMambav2(
    #     base_model=model.backbone, d_model=1024, num_classes=num_action_classes, 
    #     num_future=3, use_multihead=True,
    #     target_fps=cfg_target_fps, context_fps=cfg_context_fps, query_fps=cfg_query_fps, 
    #     compression_ratio=cfg_compression_ratio, frames_per_query=cfg_frames_per_query
    # ).to(device)
    
    # --- 4. Load Weights ---
    print(f"Loading weights from: {WEIGHT_PATH}")
    state_dict = torch.load(WEIGHT_PATH, map_location='cuda')
    
    # Load state dict (strict=True ensures you didn't miss any shape mismatches like the compressor)
    full_model.load_state_dict(state_dict, strict=True) 
    print("Weights loaded successfully.")
    
    # --- 5. Evaluate ---
    evaluate_streaming_performance(full_model, val_loader)

if __name__ == "__main__":
    main()