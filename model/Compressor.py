import torch
import torch.nn as nn
from functools import partial
import math

# Import both Mamba versions from the official library
try:
    from mamba_ssm import Mamba 
    from mamba_ssm.modules.mamba2 import Mamba2
except ImportError:
    print("Please install mamba-ssm: pip install mamba-ssm")

class MambaRMSNorm(nn.Module):
    """Pre-normalization layer."""
    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    
class MambaMLP(nn.Module):
    """Standard Feed-Forward Network with Dropout."""
    def __init__(self, hidden_dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        intermediate_dim = int(hidden_dim * mlp_ratio)
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.act = nn.SiLU() 
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)
        self.drop = nn.Dropout(dropout) # Add dropout layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.drop(x) # Apply dropout before returning

class GatedFusionMambaBlock(nn.Module):
    def __init__(
        self, 
        hidden_dim, 
        frames_per_query, 
        bidirectional=True, 
        backbone="Mamba2",
        state_size=16,
        conv_kernel=4,
        expand=2,
        head_dim=64,
        use_mlp=True,     
        mlp_ratio=4.0,     
        dropout=0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.frames_per_query = frames_per_query
        self.bidirectional = bidirectional
        self.use_mlp = use_mlp
        # 1. Pre-Norm
        self.norm1 = MambaRMSNorm(self.hidden_dim)
        
        # 2. GPA Parameters (Adaptive Gating)
        self.gate_drop = nn.Dropout(p=dropout)
        self.weight_drop = nn.Dropout(p=dropout)
        self.weight_fc = nn.Linear(self.hidden_dim, self.frames_per_query)
        self.gate_fc = nn.Linear(self.hidden_dim, 1)
        
        nn.init.zeros_(self.weight_fc.bias)
        nn.init.zeros_(self.gate_fc.bias)
        with torch.no_grad():
            self.weight_fc.weight.mul_(1e-3)
            self.gate_fc.weight.mul_(1e-3)

        # 3. Dynamic Backbone Selection
        if backbone == "Mamba1":
            mixer_cls = partial(
                Mamba, d_model=self.hidden_dim, d_state=state_size, 
                d_conv=conv_kernel, expand=expand
            )
        elif backbone == "Mamba2":
            mixer_cls = partial(
                Mamba2, d_model=self.hidden_dim, d_state=state_size, 
                d_conv=conv_kernel, expand=expand, headdim=head_dim
            )
        else:
            raise ValueError("Choose 'Mamba1' or 'Mamba2'.")

        self.mixer_fwd = mixer_cls()
        if self.bidirectional:
            self.mixer_bwd = mixer_cls()

        # 4. Residual Dropout
        self.resid_drop = nn.Dropout(dropout)

        # 5. Optional MLP Block
        if self.use_mlp:
            self.norm2 = MambaRMSNorm(self.hidden_dim)
            self.mlp = MambaMLP(self.hidden_dim, mlp_ratio, dropout=dropout)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        bsz, seq_len, hidden_dim = hidden_states.shape
        chunk_with_query = self.frames_per_query + 1
        n_chunk = seq_len // chunk_with_query

        # === Temporal Gated Aggregation (GPA) ===
        hidden_4d = hidden_states.reshape(bsz, n_chunk, chunk_with_query, hidden_dim)
        frames = hidden_4d[:, :, :self.frames_per_query, :]
        queries = hidden_4d[:, :, self.frames_per_query, :]

        w_in = self.weight_drop(queries)
        alpha = torch.softmax(self.weight_fc(w_in), dim=-1)
        aggregator = (frames * alpha.unsqueeze(-1)).sum(dim=2)

        gating_in = self.gate_drop(queries)
        gating = torch.sigmoid(self.gate_fc(gating_in))
        epsilon = 0.01
        gating = gating * (1 - 2 * epsilon) + epsilon
        gating_broad = gating.expand(-1, -1, hidden_dim)

        # f = (1-g)q + ga
        queries_new = queries * (1 - gating_broad) + aggregator * gating_broad

        # Re-concatenate the frames and the newly updated queries along the sequence dimension
        hidden_4d = torch.cat([frames, queries_new.unsqueeze(2)], dim=2)
        hidden_states = hidden_4d.reshape(bsz, seq_len, hidden_dim)

        # === Sequence Modeling (Global Scan) ===
        if self.bidirectional:
            out_fwd = self.mixer_fwd(hidden_states)
    
            # Apply contiguous() before passing to the kernel and after flipping back
            hidden_bwd = hidden_states.flip([1]).contiguous()
            out_bwd = self.mixer_bwd(hidden_bwd).flip([1]).contiguous()
            
            hidden_states = out_fwd + out_bwd
        else:
            hidden_states = self.mixer_fwd(hidden_states)

        hidden_states = self.resid_drop(hidden_states)
        hidden_states = hidden_states + residual

        # === 3. Optional Channel Mixing (MLP) ===
        if self.use_mlp:
            residual_mlp = hidden_states
            hidden_states = self.norm2(hidden_states)
            hidden_states = self.mlp(hidden_states)
            # (Dropout for the MLP is already handled inside the MambaMLP's forward pass)
            hidden_states = hidden_states + residual_mlp

        return hidden_states + residual


class TemporalCompressor(nn.Module):
    """
    Accepts raw vision embeddings, handles query insertion, processes them 
    through Mamba layers, and outputs the compressed sequence.
    """
    def __init__(
        self, 
        hidden_dim=1024, 
        frames_per_query=10, 
        num_layers=3, 
        bidirectional=True, 
        backbone="Mamba2",
        padding=False,
        use_mlp=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.frames_per_query = frames_per_query
        self.padding = padding
        self.use_mlp = use_mlp
        
        # The learnable query token that acts as our "blank canvas" for each chunk
        std = 1.0 / math.sqrt(hidden_dim)
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * std)
        
        # The stack of Mamba blocks
        self.blocks = nn.ModuleList([
            GatedFusionMambaBlock(
                hidden_dim=hidden_dim, 
                frames_per_query=frames_per_query, 
                bidirectional=bidirectional, 
                backbone=backbone,
                use_mlp=self.use_mlp
            ) 
            for _ in range(num_layers)
        ])

    def forward(self, vision_embs):
        """
        Args:
            vision_embs: (batch_size, num_frames, hidden_dim)
        Returns:
            compressed_embs: (batch_size, num_chunks, hidden_dim)
        """
        bsz, num_frames, hidden_dim = vision_embs.shape
        
        # 1. Padding (Ensure num_frames is perfectly divisible by frames_per_query)
        remainder = num_frames % self.frames_per_query
        if remainder != 0:
            if self.padding:
                pad_len = self.frames_per_query - remainder
                # Pad the sequence with zeros along the temporal dimension
                padding = torch.zeros(bsz, pad_len, hidden_dim, device=vision_embs.device, dtype=vision_embs.dtype)
                vision_embs = torch.cat([vision_embs, padding], dim=1)
                num_frames = vision_embs.shape[1]
            else:
                vision_embs = vision_embs[:, :-remainder, :]
                num_frames = vision_embs.shape[1]
                if num_frames == 0:
                    # Return an empty tensor of the correct shape to avoid crashing downstream
                    return None

        num_chunks = num_frames // self.frames_per_query

        if num_chunks == 0:
            return None

        # 2. Reshape into chunks to prepare for query insertion
        # Shape: (bsz, num_chunks, frames_per_query, hidden_dim)
        chunked_embs = vision_embs.reshape(bsz, num_chunks, self.frames_per_query, hidden_dim)

        # 3. Expand the learned query token to match the batch and chunk size
        # Shape: (bsz, num_chunks, 1, hidden_dim)
        queries = self.query_token.expand(bsz, num_chunks, 1, hidden_dim)

        # 4. Insert the query token at the end of every chunk
        # Shape: (bsz, num_chunks, frames_per_query + 1, hidden_dim)
        chunked_with_queries = torch.cat([chunked_embs, queries], dim=2)

        # 5. Flatten back to a 1D sequence for Mamba
        # The sequence is now slightly longer: num_frames + num_chunks
        hidden_states = chunked_with_queries.reshape(bsz, -1, hidden_dim)

        # 6. Pass through the stacked Mamba blocks
        for block in self.blocks:
            hidden_states = block(hidden_states)
            
        # 7. Extract the final compressed query tokens
        chunk_with_query_len = self.frames_per_query + 1
        final_4d = hidden_states.reshape(bsz, num_chunks, chunk_with_query_len, hidden_dim)
        
        # We only want the last token of every chunk (the fully updated query)
        compressed_embs = final_4d[:, :, self.frames_per_query, :]
        
        return compressed_embs

class TemporalMambaBlock(nn.Module):
    """Wraps the raw Mamba mixer with Pre-Norm, Residual, and an optional MLP."""
    def __init__(self, hidden_dim, mixer_cls, use_mlp=True, mlp_ratio=4.0):
        super().__init__()
        # --- Mixer Pathway ---
        self.norm1 = MambaRMSNorm(hidden_dim)
        self.mixer = mixer_cls()
        
        # --- Optional MLP Pathway ---
        self.use_mlp = use_mlp
        if self.use_mlp:
            self.norm2 = MambaRMSNorm(hidden_dim)
            self.mlp = MambaMLP(hidden_dim, mlp_ratio)
            
    def forward(self, hidden_states):
        # 1. Mamba Mixer Block (Sequence Modeling)
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mixer(hidden_states)
        hidden_states = hidden_states + residual
        
        # 2. Optional MLP Block (Channel Mixing)
        if self.use_mlp:
            residual = hidden_states
            hidden_states = self.norm2(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = hidden_states + residual
            
        return hidden_states

def apply_depth_scaled_init(model, total_layers):
    """
    Applies GPT-2 style depth-scaled initialization to the output projections
    of residual branches to prevent variance explosion in deep networks.
    """
    # If using MLP, there are 2 residual branches per layer (Mixer + MLP)
    n_residuals_per_layer = 2 
    
    for name, param in model.named_parameters():
        # Target the final projection of the MLP and the Mamba Mixer
        if name.endswith("fc2.weight") or name.endswith("out_proj.weight"):
            # Re-initialize with Kaiming Uniform
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            
            # Scale down based on depth
            with torch.no_grad():
                param /= math.sqrt(n_residuals_per_layer * total_layers)
                
        # Zero out linear biases (standard practice)
        elif name.endswith(".bias") and param.dim() == 1:
            nn.init.zeros_(param)

class MultiLevelCompressor(nn.Module):
    def __init__(
        self, 
        hidden_dim=1024, 
        num_stages=2,
        frames_per_query=[24, 10], 
        num_layers_per_stage=2, 
        bidirectional=True, 
        backbone="Mamba2"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.frames_per_query = frames_per_query
        self.num_stages = num_stages
        
        # The stack of Mamba blocks
        self.stages = nn.ModuleList([
            TemporalCompressor(
                hidden_dim=hidden_dim,
                frames_per_query=frames_per_query[i],
                num_layers=num_layers_per_stage,          # Stack of 3 layers
                bidirectional=bidirectional,    # Use forward and backward scanning
                backbone=backbone      # Highly optimized multi-head Mamba
            )
            for i in range(num_stages)
        ])

        if backbone == "Mamba1":
            mixer_cls = partial(
                Mamba, d_model=self.hidden_dim, d_state=16, 
                d_conv=4, expand=2
            )
        elif backbone == "Mamba2":
            mixer_cls = partial(
                Mamba2, d_model=self.hidden_dim, d_state=16, 
                d_conv=4, expand=2, headdim=64
            )
        else:
            raise ValueError("Choose 'Mamba1' or 'Mamba2'.")

        self.temporal_blocks = nn.ModuleList([
            TemporalMambaBlock(hidden_dim=self.hidden_dim, mixer_cls=mixer_cls)
            for _ in range(num_stages - 1)
        ])

        total_layers = (num_stages * num_layers_per_stage) + (num_stages - 1)
        apply_depth_scaled_init(self, total_layers)

    def forward(self, x):
        for i in range(self.num_stages-1):
            x = self.stages[i](x)
            if x is None:
                return None # Exit early if context is dropped
            x = self.temporal_blocks[i](x)
        x = self.stages[-1](x) # [B, T//compress_ratio, dim] where compress_ratio=self.frames_per_query[i] for i in range(self.num_stages)
        return x

def test_compressor_flow():
    print("Initializing MultiLevelCompressor...")
    
    # Configuration
    bsz = 2
    num_frames = 305
    hidden_dim = 1024
    frames_per_query = [30, 10]
    
    # 1. Instantiate the model
    # Note: Ensure you apply the depth-scaled init we discussed in the __init__!
    model = MultiLevelCompressor(
        hidden_dim=hidden_dim,
        num_stages=2,
        frames_per_query=frames_per_query,
        num_layers_per_stage=3,
        bidirectional=True,
        backbone="Mamba2" # Make sure mamba-ssm is installed for this to run
    )
    
    # Move to GPU if available for a realistic test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 2. Create dummy vision embeddings (e.g., from a frozen ViT or CLIP)
    # Shape: (batch_size, num_frames, hidden_dim)
    dummy_input = torch.randn(bsz, num_frames, hidden_dim, device=device)
    dummy_input.requires_grad_(True) # To check if gradients reach the input
    
    print(f"Feeding input of shape: {dummy_input.shape}")
    
    # 3. Forward Pass
    output = model(dummy_input)
    
    # 4. Assert Output Shape
    # Let's do the math:
    # Stage 1: 300 frames / 30 frames_per_query = 10 compressed tokens
    # Stage 2: 10 frames / 10 frames_per_query = 1 final compressed token
    expected_seq_len = num_frames // (frames_per_query[0] * frames_per_query[1])
    expected_shape = (bsz, expected_seq_len, hidden_dim)
    
    assert output.shape == expected_shape, f"❌ Shape mismatch! Expected {expected_shape}, but got {output.shape}"
    print(f"✅ Forward pass successful! Output shape matches: {output.shape}")
    
    # 5. Dummy Loss and Backward Pass
    # We use a simple mean to simulate a loss reduction
    loss = output.mean()
    loss.backward()
    
    # 6. Gradient Health Check
    has_nans = False
    dead_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                dead_params.append(name)
            elif torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                has_nans = True
                print(f"❌ NaN/Inf found in gradients of layer: {name}")
                
    if not has_nans and len(dead_params) == 0:
        print("✅ Backward pass successful! Gradients flowed smoothly without NaNs.")
    else:
        if len(dead_params) > 0:
            print(f"⚠️ Warning: {len(dead_params)} parameters received no gradients.")
            # Print first few to debug
            print(f"   Examples: {dead_params[:3]}")

if __name__ == "__main__":
    test_compressor_flow()
