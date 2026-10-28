"""'+M' ablation model: Base Mamba + long-term memory + historical fusion.

No future anticipation, no Time-Scale Aware Mamba, no TCN head (the TCN head
belongs to the +F stage onward). The compressor is selectable, so the Q3
comparison changes ONLY how memory tokens are formed:
    compressor="mamba"     -> MultiLevelCompressorv2 (ours)
    compressor="attention" -> AttentionMemoryCompressor (LSTR/CMeRT-style)

Base Mamba, QueryAwareMambaBlock fusion and the MLP classifier head are built
exactly as in ContextMambav2TASver.

forward() returns a dict:
    logits       [B, M, C]  frame-wise prediction (used for loss and evaluation)
    next_states  {'base_states': ..., 'tcn_caches': None}
"""

import torch.nn as nn

from model.Compressor import MultiLevelCompressorv2
from model.MambaFusion import QueryAwareMambaBlock


def build_mlp_head(in_dim, out_classes, hidden_expansion, dropout):
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, in_dim * hidden_expansion),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(in_dim * hidden_expansion, out_classes),
    )

"""Attention-based long-term memory compressor (rebuttal baseline, Reviewer Q3).

Follows the LSTR long-term memory encoder used by CMeRT:
  feature head (Linear + ReLU)  ->  sinusoidal PE + dropout
  ->  cascade of Transformer-decoder stages, each with its own learnable
      queries (nn.Embedding) cross-attending to the previous stage's output
  ->  post-norm layers with a final LayerNorm per stage (layer_norm(d, True)).

One deliberate difference from LSTR, so that ONLY the compressor changes:
  the output keeps the same token count as MultiLevelCompressorv2
  (one token per complete `frames_per_token` frames, None below that), using
  the first n queries of the last stage. LSTR instead always outputs a fixed
  number of tokens and masks padded memory. Keeping the count fixed to ours
  leaves fusion, Delta-t assignment and the anticipation sequence unchanged.

Drop-in replacement for MultiLevelCompressorv2:
    input : contexts [B, T, D]  (raw long-term buffer, oldest frame first)
    output: [B, N, d_model] with N = T // frames_per_token, or None if N == 0
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal PE + dropout, as in LSTR's tr.PositionalEncoding (batch-first)."""

    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[: x.shape[1]].unsqueeze(0).to(x.dtype)
        return self.dropout(x)


class AttentionMemoryCompressor(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        d_model=1024,
        frames_per_token=300,        # = model.compression_ratio (context frames per token)
        max_tokens=10,               # = long-term span / frames_per_token
        enc_module=((16, 1), (10, 2)),  # [(num_queries, num_layers), ...]; copy from your
                                        # CMeRT cfg ENC_MODULE, last stage's queries = max_tokens
        num_heads=16,                # cfg.MODEL.LSTR.NUM_HEADS
        dim_feedforward=1024,        # cfg.MODEL.LSTR.DIM_FEEDFORWARD
        dropout=0.2,                 # cfg.MODEL.LSTR.DROPOUT
        activation="relu",           # cfg.MODEL.LSTR.ACTIVATION
        use_feature_head=True,       # LSTR/CMeRT project features before the memory
    ):
        super().__init__()
        if enc_module[-1][0] != max_tokens:
            raise ValueError("Last stage must have max_tokens queries to match our token count.")
        if not use_feature_head and input_dim != d_model:
            raise ValueError("input_dim must equal d_model without a feature head.")

        self.frames_per_token = int(frames_per_token)
        self.max_tokens = int(max_tokens)

        self.feature_head = (
            nn.Sequential(nn.Linear(input_dim, d_model), nn.ReLU(inplace=True))
            if use_feature_head else nn.Identity()
        )
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        self.enc_queries = nn.ModuleList()
        self.enc_modules = nn.ModuleList()
        for num_queries, num_layers in enc_module:
            self.enc_queries.append(nn.Embedding(num_queries, d_model))
            layer = nn.TransformerDecoderLayer(
                d_model, num_heads, dim_feedforward, dropout, activation,
                batch_first=True, norm_first=False,   # post-norm, as in LSTR
            )
            self.enc_modules.append(
                nn.TransformerDecoder(layer, num_layers, norm=nn.LayerNorm(d_model))
            )

    def forward(self, x):
        if x is None:
            return None
        bsz, num_frames, _ = x.shape
        n_tokens = num_frames // self.frames_per_token
        if n_tokens == 0:
            return None  # same behavior as MultiLevelCompressorv2
        if n_tokens > self.max_tokens:
            raise ValueError(
                f"Buffer gives {n_tokens} tokens but max_tokens={self.max_tokens}; "
                "increase max_tokens to match the long-term span."
            )

        # Keep exactly the frames MultiLevelCompressorv2 keeps (drops trailing remainder).
        x = x[:, : n_tokens * self.frames_per_token]

        memory = self.pos_encoding(self.feature_head(x))
        last = len(self.enc_modules) - 1
        for i, (queries, module) in enumerate(zip(self.enc_queries, self.enc_modules)):
            q = queries.weight
            if i == last:
                q = q[:n_tokens]  # query i plays "token i", oldest first, like ours
            q = q.unsqueeze(0).expand(bsz, -1, -1)
            memory = module(tgt=q, memory=memory)
        return memory


def count_params(module):
    return sum(p.numel() for p in module.parameters())


if __name__ == "__main__":
    # Shape / edge-case check (pure PyTorch, no mamba_ssm needed).
    torch.manual_seed(0)
    comp = AttentionMemoryCompressor()
    comp.eval()
    print(f"Attention compressor params: {count_params(comp) / 1e6:.1f}M")
    for T in [0, 150, 299, 300, 450, 1205, 3000]:
        out = comp(torch.randn(1, T, 1024))
        shape = None if out is None else tuple(out.shape)
        expected = T // 300
        ok = (out is None and expected == 0) or (out is not None and out.shape[1] == expected)
        print(f"T={T:5d} -> {shape}  {'OK' if ok else 'MISMATCH'}")
        if out is not None:
            assert torch.isfinite(out).all()
    comp.train()
    out = comp(torch.randn(1, 3000, 1024))
    out.mean().backward()
    dead = [n for n, p in comp.named_parameters() if p.grad is None]
    print("No dead params" if not dead else f"Params without grad: {dead}")
    
class MambaMemory(nn.Module):
    def __init__(
        self,
        base_model,                    # MixerModel (base_container.backbone)
        d_model: int,
        num_classes: int,
        compressor: str = "mamba",
        compression_ratio: float = 300.0,   # context frames per memory token
        max_tokens: int = 10,               # long-term span / compression_ratio
        # --- ours (MultiLevelCompressorv2) ---
        frames_per_query=(30, 10),
        num_layers_per_stage: int = 4,
        # --- attention baseline ---
        attn_kwargs: dict = None,           # enc_module, num_heads, dim_feedforward, dropout, activation
        # --- head ---
        dropout: float = 0.1,
        hidden_expansion: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.base_model = base_model

        if compressor == "mamba":
            self.compressor = MultiLevelCompressorv2(
                hidden_dim=d_model,
                frames_per_query=list(frames_per_query),
                num_layers_per_stage=num_layers_per_stage,
            )
        elif compressor == "attention":
            self.compressor = AttentionMemoryCompressor(
                input_dim=d_model,
                d_model=d_model,
                frames_per_token=int(compression_ratio),
                max_tokens=max_tokens,
                **(attn_kwargs or {}),
            )
        else:
            raise ValueError(f"Unknown compressor: {compressor}")

        self.fusion = QueryAwareMambaBlock(d_model=d_model)
        self.classifier = build_mlp_head(d_model, num_classes, hidden_expansion, dropout)

    def forward(self, vision_embeddings, contexts, pass_states=None, labels=None):
        base_pass_states = pass_states.get("base_states") if isinstance(pass_states, dict) else None

        compressed_ctx = self.compressor(contexts)
        x, base_next_states = self.base_model(
            vision_embeddings=vision_embeddings, pass_states=base_pass_states,
        )

        if compressed_ctx is None:
            enhanced = x  # no complete memory token yet (same as TASver)
        else:
            enhanced = self.fusion(F_s=compressed_ctx, F_q=x, delta_t_s=None, delta_t_q=None)

        return {
            "logits": self.classifier(enhanced),
            "next_states": {"base_states": base_next_states, "tcn_caches": None},
        }
