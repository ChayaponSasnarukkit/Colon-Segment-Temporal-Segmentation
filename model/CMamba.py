import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mlp import GatedMLP

# from mamba_ssm.modules.block import Block

from mamba_ssm.utils.generation import GenerationMixin

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def selective_scan_with_init_hidden(
    u, delta, A, B, C, D=None, z=None, 
    delta_bias=None, delta_softplus=False, init_state=None
):
    """
    Tricks the CUDA selective_scan_fn into accepting an initial hidden state
    by algebraically modifying the B matrix at step t=0.
    """
    if init_state is None:
        return selective_scan_fn(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state=True)
        
    # Expand B and C to have the D dimension, as the state injection is D-dependent
    dim = A.shape[0]
    if B.dim() == 3: # [B, N, L] -> [B, D, N, L]
        B = B.unsqueeze(1).expand(-1, dim, -1, -1).clone()
    else:
        B = B.clone()
        
    if C.dim() == 3:
        C = C.unsqueeze(1).expand(-1, dim, -1, -1).clone()
    else:
        C = C.clone()

    delta2 = delta + (delta_bias[..., None].float() if delta_bias is not None else 0.0)
    
    # --- STABILITY FIX: Prevent Division by Zero ---
    u_zero = u[:, :, 0]
    u_safe = u_zero.clone()
    u_safe[u_safe == 0] = 1e-3  # Absolute zero fallback
    u_safe = torch.where(u_safe.abs() < 1e-3, u_safe.sign() * 1e-3, u_safe)
    
    v = F.softplus(delta2[:, :, 0]) * u_safe
    
    # Calculate the B modifier for t=0
    sig = (init_state * torch.exp(F.softplus(delta2[:, :, 0]).unsqueeze(-1) * A.unsqueeze(0))) / v.unsqueeze(-1)
    
    # Inject into the first timestep
    B[:, :, :, 0] = B[:, :, :, 0] + sig
    
    return selective_scan_fn(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state=True)


class MambaTBPTT(Mamba):
    """
    Subclasses the official Mamba block to intercept the forward pass,
    route it to our CUDA scan interface, and manage the convolutional state.
    """
    def forward(self, hidden_states, inference_params=None, pass_init_state=None, pass_conv_state=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out, None, None

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None and pass_init_state is None and pass_conv_state is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            # next_conv_state = x[..., -(self.d_conv - 1):].clone() if self.d_conv > 1 else None

            if pass_conv_state is not None:
                #if causal_conv1d_fn is not None:
                if False:
                    x_cl = x.permute(0, 2, 1).clone().permute(0, 2, 1)
                    state_cl = pass_conv_state.permute(0, 2, 1).clone().permute(0, 2, 1)

                    # 2. Extract next_conv_state and immediately force it into channel-last layout
                    if self.d_conv > 1:
                        sliced_state = x[..., -(self.d_conv - 1):]
                        next_conv_state = sliced_state.transpose(1, 2).contiguous().transpose(1, 2)
                    else:
                        next_conv_state = None
                    print(x_cl.shape, state_cl.shape)
                    x = causal_conv1d_fn(
                        x=x_cl,
                        weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                        bias=self.conv1d.bias,
                        initial_states=state_cl,
                        return_final_states=False,
                        activation=self.activation,
                    )
                    # x = x_cl.transpose(1, 2).contiguous()
                else:
                    # Native PyTorch fallback (Strictly causal via manual padding)
                    next_conv_state = x[..., -(self.d_conv - 1):].clone() if self.d_conv > 1 else None
                    
                    if pass_conv_state is not None:
                        x_padded = torch.cat([pass_conv_state, x], dim=-1)
                        x_conv = F.conv1d(
                            x_padded, 
                            weight=self.conv1d.weight, 
                            bias=self.conv1d.bias, 
                            padding=0, 
                            groups=self.conv1d.groups 
                        )
                    else:
                        x_conv = self.conv1d(x)[..., :seqlen]
                        
                    x = self.act(x_conv)
            else:
                next_conv_state = x[..., -(self.d_conv - 1):].clone() if self.d_conv > 1 else None
                if causal_conv1d_fn is None:
                    x = self.act(self.conv1d(x)[..., :seqlen])
                else:
                    assert self.activation in ["silu", "swish"]
                    x = causal_conv1d_fn(
                        x=x,
                        weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                        bias=self.conv1d.bias,
                        activation=self.activation,
                    )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y, next_ssm_state = selective_scan_with_init_hidden(
                x,
                dt,
                A,
                B,
                C,
                D=self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                init_state=pass_init_state, # Inject custom state here
            )
            if ssm_state is not None:
                ssm_state.copy_(next_ssm_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out, next_ssm_state, next_conv_state


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )
        
        # === FIX ===: Intercept the mixer output to unpack TBPTT states safely
        mixer_out = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)
        
        if isinstance(mixer_out, tuple):
            # If using MambaTBPTT, extract the states and keep the tensor for the MLP/Residual
            hidden_states, next_ssm_state, next_conv_state = mixer_out
        else:
            # Fallback for standard standard Mamba/Attention mixers
            hidden_states = mixer_out
            next_ssm_state, next_conv_state = None, None
        # === END FIX ===

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            hidden_states = self.mlp(hidden_states)

        # === FIX ===: Return the states alongside the standard outputs
        return hidden_states, residual, next_ssm_state, next_conv_state
        # === END FIX ===

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

import math
import copy
from functools import partial
import torch
import torch.nn as nn

# Assuming MambaTBPTT, Block (the updated one), RMSNorm, and layer_norm_fn are imported

def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        
        # === FIX ===: Replace the standard Mamba mixer with your MambaTBPTT
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else MambaTBPTT,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
        # === END FIX ===
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vision_dim: int, # === FIX ===: Swapped vocab_size for vision_dim
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # === FIX ===: Replaced Embedding with Linear projection for vision features
        self.vision_proj = nn.Linear(vision_dim, d_model, **factory_kwargs)
        # === END FIX ===

        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or RMSNorm is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    # === FIX ===: Accept vision_embeddings instead of input_ids
    def forward(self, vision_embeddings, pass_states=None, inference_params=None, **mixer_kwargs):
        # === FIX ===: Project pre-extracted vision embeddings
        hidden_states = self.vision_proj(vision_embeddings)
        # === END FIX ===
        
        residual = None
        new_states = []

        for i, layer in enumerate(self.layers):
            # Extract specific states for this layer
            layer_ssm_state = pass_states[i][0] if pass_states is not None else None
            layer_conv_state = pass_states[i][1] if pass_states is not None else None
            
            # Pass custom states into the custom Block
            hidden_states, residual, next_ssm_state, next_conv_state = layer(
                hidden_states, 
                residual, 
                inference_params=inference_params, 
                pass_init_state=layer_ssm_state,
                pass_conv_state=layer_conv_state,
                **mixer_kwargs
            )
            
            # Save the states for the next TBPTT chunk
            new_states.append((next_ssm_state, next_conv_state))

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
            
        return hidden_states, new_states
    
from collections import namedtuple

class MambaTemporalSegmentation(nn.Module):
    def __init__(self, config, vision_dim: int, num_classes: int, device=None, dtype=None, loss_fn=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Instantiate your modified official MixerModel
        self.backbone = MixerModel(
            d_model=config.d_model,
            n_layer=config.n_layer,
            d_intermediate=config.d_intermediate,
            vision_dim=vision_dim,
            ssm_cfg=config.ssm_cfg,
            rms_norm=config.rms_norm,
            fused_add_norm=config.fused_add_norm,
            **factory_kwargs
        )
        
        # Classification head (per-frame prediction)
        self.classifier = nn.Linear(config.d_model, num_classes, bias=False, **factory_kwargs)
        if loss_fn:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, vision_embeddings, pass_states=None, labels=None):
        # 1. Forward through backbone
        hidden_states, new_states = self.backbone(vision_embeddings, pass_states=pass_states)
        
        # 2. Project to classes
        logits = self.classifier(hidden_states) 

        # 3. Compute per-frame Loss
        loss = None
        if labels is not None:
            # Flatten to (B * L, num_classes) and (B * L)
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        Output = namedtuple("Output", ["loss", "logits", "next_states"])
        return Output(loss=loss, logits=logits, next_states=new_states)
    
def detach_states(states):
    """Safely detaches both SSM and Convolutional states from the computational graph."""
    if states is None:
        return None
        
    detached_states = []
    for ssm_state, conv_state in states:
        d_ssm = ssm_state.detach() if ssm_state is not None else None
        d_conv = conv_state.detach() if conv_state is not None else None
        detached_states.append((d_ssm, d_conv))
        
    return detached_states

def apply_reset_mask(states, reset_mask):
    """
    Selectively zeroes out the states for specific videos in the batch.
    mask shape: (Batch,) where True means "reset this video".
    """
    if states is None:
        return None
        
    masked_states = []
    for ssm_state, conv_state in states:
        new_ssm = ssm_state
        new_conv = conv_state
        
        if ssm_state is not None:
            # Reshape mask to broadcast across the state dimensions: (B, 1, 1...)
            broadcast_mask = reset_mask.view(-1, *[1]*(ssm_state.dim() - 1))
            new_ssm = ssm_state.masked_fill(broadcast_mask, 0.0)
            
        if conv_state is not None:
            broadcast_mask = reset_mask.view(-1, *[1]*(conv_state.dim() - 1))
            new_conv = conv_state.masked_fill(broadcast_mask, 0.0)
            
        masked_states.append((new_ssm, new_conv))
        
    return masked_states

def train_tbptt_fast(model, dataloader, optimizer, device, epochs=1):
    model.train()
    
    for epoch in range(epochs):
        # We hold the states across chunks in this list
        current_states = None 
        
        for step, (vision_embeddings, reset_mask, labels) in enumerate(dataloader):
            vision_embeddings = vision_embeddings.to(device)
            labels = labels.to(device)
            
            # If the dataset flags a new video, wipe the states
            if reset_mask.any() and current_states is not None:
                current_states = apply_reset_mask(current_states, reset_mask)

            optimizer.zero_grad()
            
            # 1. Forward Pass (Runs in parallel chunk mode via CUDA!)
            outputs = model(
                vision_embeddings=vision_embeddings, 
                pass_states=current_states,
                labels=labels
            )
            
            # 2. Backpropagate
            outputs.loss.backward()
            optimizer.step()
            
            # 3. Detach states for the next chunk's T-BPTT
            # This prevents PyTorch from backpropagating forever and causing OOM
            current_states = detach_states(outputs.next_states)
            
            if step % 10 == 0:
                print(f"Step {step} | Loss: {outputs.loss.item():.4f}")

# class MixerModelTBPTT(nn.Module):
#     def __init__(self, config: MambaConfig, vision_dim: int, device=None, dtype=None, residual_in_fp32=True):
#         super().__init__()
#         factory_kwargs = {"device": device, "dtype": dtype}
#         self.residual_in_fp32 = residual_in_fp32
        
#         self.vision_proj = nn.Linear(vision_dim, config.d_model, **factory_kwargs)
        
#         # Instantiate layers
#         self.layers = nn.ModuleList([
#             Block(
#                 config.d_model,
#                 mixer_cls=lambda d_model: MambaTBPTT(d_model=d_model, d_state=config.ssm_cfg.get("d_state", 16), d_conv=config.ssm_cfg.get("d_conv", 4), expand=config.ssm_cfg.get("expand", 2), **factory_kwargs),
#                 mlp_cls=nn.Identity,
#                 norm_cls=lambda d: nn.LayerNorm(d, eps=1e-5, **factory_kwargs),
#                 fused_add_norm=False,
#             )
#             for _ in range(config.n_layer)
#         ])
#         self.norm_f = nn.LayerNorm(config.d_model, eps=1e-5, **factory_kwargs)

#         # --- FIX: Apply official GPT-2/Mamba weight initialization ---
#         self.apply(
#             lambda module: _init_weights(
#                 module,
#                 n_layer=config.n_layer,
#                 n_residuals_per_layer=1 # 1 because we don't have interleaved MLPs
#             )
#         )

#     def forward(self, vision_embeddings, pass_states=None):
#         hidden_states = self.vision_proj(vision_embeddings)
        
#         # --- FIX: Optional FP32 cast for numerical stability in the residual stream ---
#         residual = hidden_states.to(torch.float32) if self.residual_in_fp32 else hidden_states 
#         new_states = []
        
#         for i, layer in enumerate(self.layers):
#             layer_ssm_state = pass_states[i][0] if pass_states is not None else None
#             layer_conv_state = pass_states[i][1] if pass_states is not None else None
            
#             # Layer norm expects the dtype of its weights (usually fp16/bf16), so cast back if needed
#             normed_hidden = layer.norm(residual.to(dtype=layer.norm.weight.dtype))
            
#             # Unpack the block to pass custom states
#             mixer_out, next_ssm_state, next_conv_state = layer.mixer(
#                 normed_hidden, 
#                 pass_init_state=layer_ssm_state,
#                 pass_conv_state=layer_conv_state
#             )
            
#             new_states.append((next_ssm_state, next_conv_state))
            
#             # Accumulate the residual
#             residual = residual + mixer_out.to(dtype=residual.dtype)

#         # Final norm
#         hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
#         return hidden_states, new_states


# class MambaActionRecognitionModel(nn.Module, GenerationMixin):
#     def __init__(self, config: MambaConfig, vision_dim: int, num_classes: int, device=None, dtype=None):
#         super().__init__()
#         self.config = config
#         factory_kwargs = {"device": device, "dtype": dtype}
        
#         self.backbone = MixerModelTBPTT(config, vision_dim, **factory_kwargs)
#         self.classifier = nn.Linear(config.d_model, num_classes, bias=False, **factory_kwargs)

#     def forward(self, vision_embeddings, pass_states=None, labels=None):
#         # pass_states is now expected to be a list of (ssm_state, conv_state) tuples
#         hidden_states, new_states = self.backbone(vision_embeddings, pass_states=pass_states)
        
#         # Note: If predicting one action per sequence rather than per frame, 
#         # you may want to pool `hidden_states` here before the classifier.
#         logits = self.classifier(hidden_states) 

#         loss = None
#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
#             loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

#         Output = namedtuple("Output", ["loss", "logits", "next_states"])
#         return Output(loss=loss, logits=logits, next_states=new_states)
    
# def train_tbptt_fast(model, dataloader, optimizer, device, epochs=1):
#     model.train()
    
#     for epoch in range(epochs):
#         # We hold the states across chunks in this list
#         current_states = None 
        
#         for step, (vision_embeddings, reset_mask, labels) in enumerate(dataloader):
#             vision_embeddings = vision_embeddings.to(device)
#             labels = labels.to(device)
            
#             # If the dataset flags a new video, wipe the states
#             if reset_mask.any() and current_states is not None:
#                 current_states = None 

#             optimizer.zero_grad()
            
#             # 1. Forward Pass (Runs in parallel chunk mode via CUDA!)
#             outputs = model(
#                 vision_embeddings=vision_embeddings, 
#                 pass_init_states=current_states,
#                 labels=labels
#             )
            
#             # 2. Backpropagate
#             outputs.loss.backward()
#             optimizer.step()
            
#             # 3. Detach states for the next chunk's T-BPTT
#             # This prevents PyTorch from backpropagating forever and causing OOM
#             current_states = [state.detach() for state in outputs.next_states]
            
#             if step % 10 == 0:
#                 print(f"Step {step} | Loss: {outputs.loss.item():.4f}")
