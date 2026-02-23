from model.CMamba import MixerModel, MambaTemporalSegmentation
from model.Compressor import MultiLevelCompressor
from model.MambaFusion import QueryAwareMambaBlock, CausalQueryAwareMambaBlock
import torch
import torch.nn as nn

class ContextMamba(nn.Module):
    def __init__(
        self,
        base_model: MixerModel,
        d_model: int,
        num_classes: int,
        num_future: int,
        compression_ratio: float = 240.0,
        target_fps: float = 30.0,
        context_fps: float = 4.0,
        query_fps: float = 30.0,
        dropout: float = 0.1,
        **factory_kwargs
    ):
        super().__init__()
        # Store architecture variables
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_future = num_future
        self.compression_ratio = compression_ratio
        self.target_fps = target_fps
        self.context_fps = context_fps
        self.query_fps = query_fps

        # Sub-modules
        # self.base_model = MixerModel(
        #     d_model=config.d_model,
        #     n_layer=config.n_layer,
        #     d_intermediate=config.d_intermediate,
        #     vision_dim=vision_dim,
        #     ssm_cfg=config.ssm_cfg,
        #     rms_norm=config.rms_norm,
        #     fused_add_norm=config.fused_add_norm,
        #     **factory_kwargs
        # ) # return hidden, next
        self.base_model = base_model
        self.compressor = MultiLevelCompressor()
        self.fusion = QueryAwareMambaBlock(d_model=d_model)

        # Anticipation: predict the next num_future second ahead using context and current
        self.anticipation_head = CausalQueryAwareMambaBlock(
            d_model=d_model, 
            num_queries=num_future, 
            d_state=128, 
            d_conv=4, 
            expand=2
        )

        # Classifiers
        def build_mlp_head(in_dim, out_classes, hidden_expansion=2):
            hidden_dim = in_dim * hidden_expansion
            return nn.Sequential(
                nn.LayerNorm(in_dim),                    # 1. Normalization
                nn.Linear(in_dim, hidden_dim),           # 3. Enrichment MLP (Layer 1)
                nn.GELU(),                               # Activation
                nn.Dropout(dropout),                     # 2. Dropout
                nn.Linear(hidden_dim, out_classes)       # 3. Enrichment MLP (Head)
            )

        # Classifiers (Upgraded to MLP heads)
        self.classifier_wo_future = build_mlp_head(d_model, num_classes)
        self.future_classifier = build_mlp_head(d_model, num_classes)
        self.classifier_w_future = build_mlp_head(d_model, num_classes)
        
        # Projection layer for fusing current state and anticipated future
        # Added Norm, Activation, and Dropout to enrich this fusion step as well
        self.future_fusion_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model, **factory_kwargs),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def compute_normalized_dts(self, B, M, K, device, dtype):
        """
        Calculates the normalized time scales based on the maximum temporal span.
        M = query sequence length
        K = scene/context sequence length
        """
        max_equiv_frames = self.compression_ratio * (self.target_fps / self.context_fps) # 240*30/4=60*30=1800
        
        s_equiv_frames = self.compression_ratio * (self.target_fps / self.context_fps)
        dt_s_value = s_equiv_frames / max_equiv_frames
        
        q_equiv_frames = 1.0 * (self.target_fps / self.query_fps)
        dt_q_value = q_equiv_frames / max_equiv_frames
        
        dt_s = torch.full((B, K, 1), dt_s_value, device=device, dtype=dtype)
        dt_q = torch.full((B, M, 1), dt_q_value, device=device, dtype=dtype)
        
        return dt_s, dt_q

    def forward(self, vision_embeddings, contexts, pass_states=None, labels=None):
        device = vision_embeddings.device
        dtype = vision_embeddings.dtype

        # 1. Compress the historical context [B, T_long, D] -> [B, K, D]
        compressed_ctx = self.compressor(contexts)

        # 2. Extract baseline query features -> [B, M, D]
        x, next_states = self.base_model(
            vision_embeddings=vision_embeddings, 
            pass_states=pass_states, 
            #labels=labels
        )

        B, M, D = x.shape
        if compressed_ctx is None:
            # K=0, compress_context =[], dt_s=[]
            # calculate dt_q
            max_equiv_frames = self.compression_ratio * (self.target_fps / self.context_fps)
            
            q_equiv_frames = 1.0 * (self.target_fps / self.query_fps)
            dt_q_value = q_equiv_frames / max_equiv_frames
            dt_q = torch.full((B, M, 1), dt_q_value, device=device, dtype=dtype)
            
            enhanced_embeddings = x # no fusion needed since compress_context is empty
            full_history = enhanced_embeddings # [B, K+M, D] since K=0, full_history = enhanced_embeddings
            full_dt_s = dt_q # [B, K+M, 1] since K=0, full_dt_s = dt_q
            dt_q_val = self.target_fps / (self.compression_ratio * (self.target_fps / self.context_fps))
            full_dt_q = torch.full((B, self.num_future, 1), dt_q_val, device=device, dtype=dtype)
            future_token = self.anticipation_head(F_s=full_history, delta_t_s=full_dt_s, delta_t_q=full_dt_q) # no need to slice K since K=0
            future_token_q = future_token # [B, M, num_future, D]

        else:
            K = compressed_ctx.shape[1]

            dt_s, dt_q = self.compute_normalized_dts(B, M, K, device=device, dtype=dtype)
        
            # 3. Mamba Cross-Fusion

            enhanced_embeddings = self.fusion(
                F_s=compressed_ctx,
                F_q=x,
                delta_t_s=None,
                delta_t_q=None
            )
            # 4. Anticipation Head Setup
            # Fix: Concatenate along the sequence dimension (dim=1)
            full_history = torch.cat([compressed_ctx, enhanced_embeddings], dim=1) # [B, K+M, D]
            full_dt_s = torch.cat([dt_s, dt_q], dim=1)                             # [B, K+M, 1]
            
            # Fix: Create a properly shaped tensor for the future queries [B, num_future, 1]
            dt_q_val = self.target_fps / (self.compression_ratio * (self.target_fps / self.context_fps))
            full_dt_q = torch.full((B, self.num_future, 1), dt_q_val, device=device, dtype=dtype)
            
            # Output shape: [B, K+M, num_future, D]
            future_token = self.anticipation_head(F_s=full_history, delta_t_s=full_dt_s, delta_t_q=full_dt_q)
            # We only want the anticipation corresponding to the M current query steps.
            # We discard the predictions made over the K context prefix.
            future_token = future_token[:, K:, :, :] # [B, M, num_future, D]

            # 5. Future-Aware Fusion (TODO Completed)
            future_token_q = future_token # [B, M, num_future, D]

        # Baseline predictions
        logits_wo_future = self.classifier_wo_future(enhanced_embeddings)

        future_logits = self.future_classifier(future_token)
        
        # Pool the future predictions to get a single context vector per timestep
        summarized_future = future_token_q.mean(dim=2) # [B, M, D]
        
        # Concatenate and project back to original dimension
        fused_features = torch.cat([enhanced_embeddings, summarized_future], dim=-1) # [B, M, 2D]
        fusion_projection = self.future_fusion_proj(fused_features)            # [B, M, D]
        future_aware_embeddings = enhanced_embeddings + fusion_projection
        
        # Final prediction using the future-aware features
        logits_w_future = self.classifier_w_future(future_aware_embeddings)
        
        return logits_wo_future, future_logits, logits_w_future, next_states
