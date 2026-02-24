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
        self.future_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=8,          # 8 heads is standard for d_model=1024
            dropout=dropout, 
            batch_first=True
        )
        
        # 2. Projection and Gating
        self.future_fusion_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model, **factory_kwargs),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Gate to control information flow and prevent "phase bleeding"
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model, **factory_kwargs),
            nn.Sigmoid()
        )

        self.fusion_norm = nn.LayerNorm(d_model)
    
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
        # summarized_future = future_token_q.mean(dim=2) # [B, M, D]
        
        # Concatenate and project back to original dimension
        # fused_features = torch.cat([enhanced_embeddings, summarized_future], dim=-1) # [B, M, 2D]
        # fusion_projection = self.future_fusion_proj(fused_features)            # [B, M, D]
        # future_aware_embeddings = enhanced_embeddings + fusion_projection

        B, M, num_future, D = future_token_q.shape

        # 2. Reshape for MultiheadAttention
        # We collapse Batch and Sequence length so each step M acts as an independent query
        # Query (Current Frame): [B*M, 1, D]
        q = enhanced_embeddings.view(B * M, 1, D)
        
        # Keys/Values (Future Anticipation): [B*M, num_future, D]
        # NOTE: Using .detach() here stops the gradient tug-of-war. The anticipation head 
        # is forced to only learn from loss_future, not from loss_w.
        # kv = future_token_q.detach().view(B * M, num_future, D) 
        kv = future_token_q.view(B * M, num_future, D) 
        
        # 3. Apply Cross Attention
        # attn_output shape: [B*M, 1, D]
        attn_output, _ = self.future_cross_attn(query=q, key=kv, value=kv)
        
        # 4. Reshape back to sequence dimensions
        attended_future = attn_output.view(B, M, D) # [B, M, D]

        # --- Gated Residual Connection ---
        
        # Concatenate current and attended future
        fused_features = torch.cat([enhanced_embeddings, attended_future], dim=-1) # [B, M, 2D]
        
        # Project back to d_model
        fusion_projection = self.future_fusion_proj(fused_features)                # [B, M, D]
        
        # Calculate dynamic gate (0 to 1) based on the concatenated features
        gate = self.fusion_gate(fused_features)                                    # [B, M, D]
        
        # Apply gated residual connection and normalize
        future_aware_embeddings = self.fusion_norm(enhanced_embeddings + (gate * fusion_projection))
        
        # Final prediction using the safely fused features
        logits_w_future = self.classifier_w_future(future_aware_embeddings)
        
        return logits_wo_future, future_logits, logits_w_future, next_states
    

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Dependency Placeholders (Assuming these exist in your codebase) ---
# from .your_modules import MixerModel, MultiLevelCompressor, QueryAwareMambaBlock, CausalQueryAwareMambaBlock, TimeScaleAwareMamba2

class CMeRTNearFutureGenerator(nn.Module):
    """
    Implements the 'Near-Future Generator' from CMeRT (Sec 4.3).
    It uses a Transformer Decoder Unit (TDU) to generate future frames 
    solely from Long-Term memory.
    """
    def __init__(self, d_model, num_future_queries=3, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.num_future_queries = num_future_queries
        
        # Learnable Queries (Q_F in the paper) [num_future, d_model]
        self.learnable_queries = nn.Parameter(torch.randn(num_future_queries, d_model))
        
        # Transformer Decoder Layer (The "TDU")
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

    def forward(self, compressed_long_term):
        """
        Args:
            compressed_long_term: [B, K, D] (Keys/Values)
        Returns:
            generated_future: [B, num_future, D]
        """
        B = compressed_long_term.shape[0]
        # Expand queries to batch size: [B, num_future, D]
        queries = self.learnable_queries.unsqueeze(0).expand(B, -1, -1)
        
        # Cross-Attention: Queries attend to Long-Term Memory
        generated_future = self.transformer_decoder(
            tgt=queries, 
            memory=compressed_long_term
        )
        return generated_future


class ContextMambaCmeRT(nn.Module):
    def __init__(
        self,
        base_model,              # Your MixerModel instance
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
        
        # --- 1. Config & Attributes ---
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_future = num_future
        self.compression_ratio = compression_ratio
        self.target_fps = target_fps
        self.context_fps = context_fps
        self.query_fps = query_fps

        # --- 2. Core Modules ---
        self.base_model = base_model
        self.compressor = MultiLevelCompressor()
        self.fusion = QueryAwareMambaBlock(d_model=d_model)

        # --- 3. Anticipation Heads (The Comparison) ---
        
        # A) Your Method: Full Context Mamba
        self.anticipation_head = CausalQueryAwareMambaBlock(
            d_model=d_model, 
            num_queries=num_future, 
            d_state=128, 
            d_conv=4, 
            expand=2
        )

        # B) Ablation Method: CMeRT (Transformer, Long-Term Only)
        self.cmert_anticipation_head = CMeRTNearFutureGenerator(
            d_model=d_model,
            num_future_queries=num_future,
            nhead=8, # Adjust heads as needed
            dim_feedforward=d_model,
            dropout=dropout
        )

        # C) Edge Case Handling: Empty Long-Term Token
        # Shape: [1, 1, D]. Used when compressed_ctx is None in CMeRT mode.
        self.empty_long_term_token = nn.Parameter(torch.randn(1, 1, d_model))

        # --- 4. Classifiers & Projection Heads ---
        self.classifier_wo_future = self._build_mlp_head(d_model, num_classes, dropout)
        self.future_classifier = self._build_mlp_head(d_model, num_classes, dropout)
        self.classifier_w_future = self._build_mlp_head(d_model, num_classes, dropout)
        
        # --- 5. Fusion Modules (Future -> Current) ---
        self.future_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=8, 
            dropout=dropout, 
            batch_first=True
        )
        
        self.future_fusion_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model, **factory_kwargs),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model, **factory_kwargs),
            nn.Sigmoid()
        )

        self.fusion_norm = nn.LayerNorm(d_model)

    def _build_mlp_head(self, in_dim, out_classes, dropout, hidden_expansion=2):
        hidden_dim = in_dim * hidden_expansion
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_classes)
        )

    def compute_normalized_dts(self, B, M, K, device, dtype):
        """Calculates normalized time scales."""
        max_equiv_frames = self.compression_ratio * (self.target_fps / self.context_fps)
        
        s_equiv_frames = self.compression_ratio * (self.target_fps / self.context_fps)
        dt_s_value = s_equiv_frames / max_equiv_frames
        
        q_equiv_frames = 1.0 * (self.target_fps / self.query_fps)
        dt_q_value = q_equiv_frames / max_equiv_frames
        
        dt_s = torch.full((B, K, 1), dt_s_value, device=device, dtype=dtype)
        dt_q = torch.full((B, M, 1), dt_q_value, device=device, dtype=dtype)
        
        return dt_s, dt_q

    def forward(self, vision_embeddings, contexts, pass_states=None, labels=None):
        """
        Args:
            ablation_cmert_style (bool): If True, uses CMeRT-style anticipation 
                                         (Long-Term only, no leakage).
        """
        device = vision_embeddings.device
        dtype = vision_embeddings.dtype

        # 1. Compress Historical Context -> [B, K, D] or None
        compressed_ctx = self.compressor(contexts)

        # 2. Extract Current Features (Short-Term) -> [B, M, D]
        x, next_states = self.base_model(
            vision_embeddings=vision_embeddings, 
            pass_states=pass_states
        )
        B, M, D = x.shape

        # 3. Enhance Short-Term Features (Context Fusion)
        # This part is shared logic. If context exists, we enhance 'x'.
        if compressed_ctx is not None:
            K = compressed_ctx.shape[1]
            # Calculate DTs
            dt_s, dt_q = self.compute_normalized_dts(B, M, K, device=device, dtype=dtype)
            
            # Fuse Long-Term into Short-Term
            enhanced_embeddings = self.fusion(
                F_s=compressed_ctx,
                F_q=x,
                delta_t_s=None, # Assuming simple fusion doesn't need DT or it's handled internally
                delta_t_q=None
            )
        else:
            # Fallback: No context to fuse
            enhanced_embeddings = x

        # 4. Generate Anticipation (The Fork)
        # Goal: Produce future_token_q with shape [B, M, num_future, D]
        # === CMeRT PATH (Long-Term Only) ===
        
        # Handle Edge Case: No Long-Term Memory
        if compressed_ctx is None:
            # Use learnable "Empty" token so we don't peek at 'x'
            # Shape: [B, 1, D]
            long_term_memory = self.empty_long_term_token.expand(B, -1, -1)
        else:
            long_term_memory = compressed_ctx

        # Generate Future [B, num_future, D]
        # This uses Cross-Attention (Q=Learned, K/V=LongTerm)
        cmert_future = self.cmert_anticipation_head(long_term_memory)
        
        # Expand to M steps: [B, num_future, D] -> [B, M, num_future, D]
        # CMeRT predicts one "future" for the whole window, so we repeat it.
        future_token_q = cmert_future.unsqueeze(1).expand(-1, M, -1, -1)

        # 5. Fusion Logic (Future -> Current)
        # Input 'future_token_q' is guaranteed to be [B, M, num_future, D] here.
        
        # A) Baseline Logits (Without Future)
        logits_wo_future = self.classifier_wo_future(enhanced_embeddings)

        # B) Future Logits (Auxiliary Loss)
        # CMeRT predict future per chunk, so this is [B, num_future, num_classes] not [B, M, num_future, num_classes]
        # In training loop, we need to change the loss calculated on this
        future_logits = self.future_classifier(cmert_future)

        # C) Fuse Future into Current
        # Reshape for Attention: 
        # Query (Current): [B*M, 1, D]
        # Key/Val (Future): [B*M, num_future, D]
        q = enhanced_embeddings.view(B * M, 1, D)
        kv = future_token_q.view(B * M, self.num_future, D)

        # Cross Attention: Current Frame attends to its Anticipated Future
        attn_output, _ = self.future_cross_attn(query=q, key=kv, value=kv)
        attended_future = attn_output.view(B, M, D) # [B, M, D]

        # Gated Residual Fusion
        fused_features = torch.cat([enhanced_embeddings, attended_future], dim=-1) # [B, M, 2D]
        fusion_projection = self.future_fusion_proj(fused_features)                # [B, M, D]
        gate = self.fusion_gate(fused_features)                                    # [B, M, D]
        
        future_aware_embeddings = self.fusion_norm(
            enhanced_embeddings + (gate * fusion_projection)
        )

        # Final Logits
        logits_w_future = self.classifier_w_future(future_aware_embeddings)

        return logits_wo_future, future_logits, logits_w_future, next_states
