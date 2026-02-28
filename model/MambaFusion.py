# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

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

class TimeScaleAwareMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        # Add learnable gamma for explicit temporal scaling ---
        self.gamma = nn.Parameter(torch.rand(self.d_inner, **factory_kwargs))

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None, delta_t=None):
        """
        delta_t: (B, L, 1) time scale (e.g. norm_scale*1/(timestamp_of_current_token-timestamp_of_prev_token))
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
                return out

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
        use_fast_path = self.use_fast_path and (delta_t is None)
        if use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
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

            # Inject absolute time scale for the single step
            if delta_t is not None:
                # delta_t shape: (B, L, 1). Reshape to (B, 1, L) to match dt
                dt_span = rearrange(delta_t, "b l 1 -> b 1 l")
                # gamma is still a learnable parameter, but we force its output to be between -1 and 1
                # We multiply by a small scalar (e.g., 0.2) so the time shift doesn't overwhelm the base dt_bias
                bounded_gamma = 0.2 * torch.tanh(self.gamma)
                # self.gamma shape: (d_inner). View as (d_inner, 1) for broadcasting
                # dt = dt + (gamma * time_span)
                dt = dt + bounded_gamma.view(-1, 1) * dt_span
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state, delta_t=None):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        # Inject absolute time scale for the single step ---
        if delta_t is not None:
            # delta_t shape: (B, 1). self.gamma shape: (d_inner)
            # Broadast to (B, d_inner)
            # gamma is still a learnable parameter, but we force its output to be between -1 and 1
            # We multiply by a small scalar (e.g., 0.2) so the time shift doesn't overwhelm the base dt_bias
            bounded_gamma = 0.2 * torch.tanh(self.gamma)
            dt = dt + bounded_gamma.unsqueeze(0) * delta_t
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
    
# Copyright (c) 2024, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

from huggingface_hub import PyTorchModelHubMixin


class TimeScaleAwareMamba2(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # Add learnable gamma for explicit temporal scaling ---
        self.gamma = nn.Parameter(torch.rand(self.nheads, **factory_kwargs))

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None, delta_t=None):
        """
        delta_t: (B, L, 1) time scale (e.g. norm_scale*1/(timestamp_of_current_token-timestamp_of_prev_token))
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        if self.use_mem_eff_path and inference_params is None and delta_t is None:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
        else:
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )

            if delta_t is not None:
                # delta_t shape: (B, L, 1). dt shape: (B, L, nheads)
                # self.gamma shape: (nheads,). View as (1, 1, nheads) to broadcast
                dt = dt + self.gamma.view(1, 1, -1) * delta_t
            
            if conv_state is not None:
                if cu_seqlens is None:
                    # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
                else:
                    assert causal_conv1d_varlen_states is not None, "varlen inference requires causal_conv1d package"
                    assert batch == 1, "varlen inference only supports batch dimension 1"
                    conv_varlen_states = causal_conv1d_varlen_states(
                        xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
                    )
                    conv_state.copy_(conv_varlen_states)
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
                )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2)
            x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
                return_varlen_states=cu_seqlens is not None and inference_params is not None,
            )
            if ssm_state is not None:
                y, last_state, *rest = y
                if cu_seqlens is None:
                    ssm_state.copy_(last_state)
                else:
                    varlen_states = rest[0]
                    ssm_state.copy_(varlen_states)
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state, delta_t=None):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )
        
        if delta_t is not None:
            # delta_t shape: (B, 1). dt shape: (B, nheads). gamma shape: (nheads)
            dt = dt + self.gamma.unsqueeze(0) * delta_t
        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

import torch
import torch.nn as nn
from mamba_ssm import Mamba2  # Assuming the provided code is saved or installed

class QueryAwareMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=128, d_conv=4, expand=2, backbone="Mamba2"):
        """
        Args:
            d_model (int): Input dimension (C)
            d_state (int): SSM state dimension
            d_conv (int): Convolution kernel size
            expand (int): Expansion factor for internal dimension (C')
        """
        super().__init__()
        self.d_model = d_model
        
        # --- Normalization ---
        self.norm_s = nn.LayerNorm(d_model)
        self.norm_q = nn.LayerNorm(d_model)
        
        # --- Independent Sequence Processing ---
        if backbone=="Mamba1":
            self.mamba_s = TimeScaleAwareMamba(
                d_model=d_model, 
                d_state=d_state, 
                d_conv=d_conv, 
                expand=expand
            )
            
            self.mamba_q = TimeScaleAwareMamba(
                d_model=d_model, 
                d_state=d_state, 
                d_conv=d_conv, 
                expand=expand
            )
        else:
            self.mamba_s = TimeScaleAwareMamba2(
                d_model=d_model, 
                d_state=d_state, 
                d_conv=d_conv, 
                expand=expand
            )
            
            self.mamba_q = TimeScaleAwareMamba2(
                d_model=d_model, 
                d_state=d_state, 
                d_conv=d_conv, 
                expand=expand
            )
        
        # --- Value Projection ---
        # Projects the processed scene features for the cross-interaction update
        self.linear_F = nn.Linear(d_model, d_model)

    def forward(self, F_s, F_q, delta_t_s=None, delta_t_q=None):
        """
        Args:
            F_s: 3D scene features, shape (B, K, C)
            F_q: Box query sequence, shape (B, M, C)
            delta_t_s: Time scale for scene features, shape (B, K, 1). Optional.
            delta_t_q: Time scale for query sequence, shape (B, M, 1). Optional.
        Returns:
            Updated F_q, shape (B, M, C)
        """
        # 1. Normalization
        F_s_norm = self.norm_s(F_s)
        F_q_norm = self.norm_q(F_q)
        
        # 2. Independent Causal Scanning (Replaces Steps 3-9 in Algorithm 2)
        # Pass the newly added delta_t arguments to the time-scale-aware blocks
        # Output shapes: (B, K, C) and (B, M, C)
        y_s = self.mamba_s(F_s_norm, delta_t=delta_t_s) 
        y_q = self.mamba_q(F_q_norm, delta_t=delta_t_q)
        
        # 3. Cross-Interaction (Replaces Steps 11-12 in Algorithm 2)
        # Calculate attention/affinity matrix: (B, M, C) @ (B, C, K) -> (B, M, K)
        scale = self.d_model ** -0.5 
        attn_matrix = torch.matmul(y_q, y_s.transpose(-1, -2)) * scale 
        
        # Project processed scene features to Values: (B, K, C) -> (B, K, C)
        V = self.linear_F(y_s) 
        
        # Apply attention to Values: (B, M, K) @ (B, K, C) -> (B, M, C)
        F_q_update = torch.matmul(attn_matrix, V)
        
        # 4. Residual Connection (Step 13)
        out = F_q + F_q_update 
        
        return out
    
class CausalQueryAwareMambaBlock(nn.Module):
    def __init__(self, d_model, num_queries=3, d_state=128, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        
        # --- Learnable Queries ---
        # Initialize the learnable query of shape [M, D]
        self.learnable_query = nn.Parameter(torch.randn(num_queries, d_model))
        
        # --- Normalization ---
        self.norm_s = nn.LayerNorm(d_model)
        self.norm_q = nn.LayerNorm(d_model)
        
        # --- Independent Sequence Processing ---
        self.mamba_s = TimeScaleAwareMamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_q = TimeScaleAwareMamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        
        # --- Value Projection ---
        self.linear_F = nn.Linear(d_model, d_model)

    def forward(self, F_s, delta_t_s=None, delta_t_q=None):
        """
        Args:
            F_s: 3D scene features, shape (B, L, D)
            delta_t_s: Time scale for scene features. Optional.
            delta_t_q: Time scale for query sequence. Optional.
        Returns:
            out: Causally fused queries, shape (B, L, M, D)
        """
        B, L, D = F_s.shape
        M = self.num_queries
        
        # 0. Broadcast learnable query to match batch size: (B, M, D)
        F_q = self.learnable_query.unsqueeze(0).expand(B, -1, -1)
        
        # 1. Normalization
        F_s_norm = self.norm_s(F_s)
        F_q_norm = self.norm_q(F_q)
        
        # 2. Independent Causal Scanning
        # y_s: (B, L, D)
        # y_q: (B, M, D)
        y_s = self.mamba_s(F_s_norm, delta_t=delta_t_s) 
        y_q = self.mamba_q(F_q_norm, delta_t=delta_t_q)
        
        # 3. Value Projection for Scene Features
        V = self.linear_F(y_s) # (B, L, D)
        
        # --- 4. CAUSAL CROSS-INTERACTION ---
        # We want to use F.scaled_dot_product_attention with is_causal=True.
        # SDPA expects shape (Batch, NumHeads, SeqLen, Dim).
        # We will map our M queries to the "NumHeads" dimension, and L to "SeqLen".
        
        # Expand queries to every time step L: (B, M, L, D)
        Q = y_q.unsqueeze(2).expand(-1, -1, L, -1) 
        
        # Expand Keys and Values to match the "M" heads: (B, M, L, D)
        K = y_s.unsqueeze(1).expand(-1, M, -1, -1) 
        V_exp = V.unsqueeze(1).expand(-1, M, -1, -1)
        
        # Compute Causal Attention. 
        # For each time step j in L, it computes attention using only k <= j
        # Output shape: (B, M, L, D)
        F_q_update = F.scaled_dot_product_attention(Q, K, V_exp, is_causal=True)
        
        # Transpose to requested output shape: (B, L, M, D)
        F_q_update = F_q_update.transpose(1, 2)
        
        # 5. Residual Connection
        # Expand the original F_q (B, M, D) -> (B, 1, M, D) -> (B, L, M, D)
        out = F_q.unsqueeze(1) + F_q_update 
        
        return out
    

class CausalQueryAwareMambaBlockv2(nn.Module):
    def __init__(self, d_model, num_queries=3, d_state=128, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        
        # --- Learnable Queries ---
        self.learnable_query = nn.Parameter(torch.randn(num_queries, d_model))
        
        # --- Normalization ---
        self.norm_s = nn.LayerNorm(d_model)
        self.norm_q = nn.LayerNorm(d_model)
        
        # --- Independent Sequence Processing ---
        # (Assuming TimeScaleAwareMamba2 is defined elsewhere)
        self.mamba_s = TimeScaleAwareMamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_q = TimeScaleAwareMamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        
        # --- Key and Value Projections ---
        # Mapping y_s into distinct Key and Value spaces gives the network more capacity
        # to learn "what to match" (Key) separately from "what to extract" (Value).
        self.linear_K = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)
        self.linear_Q = nn.Linear(d_model, d_model)

    def forward(self, F_s, delta_t_s=None, delta_t_q=None):
        B, L, D = F_s.shape
        M = self.num_queries
        
        # 0. Broadcast learnable query to match batch size: (B, M, D)
        F_q = self.learnable_query.unsqueeze(0).expand(B, -1, -1)
        
        # 1. Normalization
        F_s_norm = self.norm_s(F_s)
        F_q_norm = self.norm_q(F_q)
        
        # 2. Independent Causal Scanning
        # y_s: (B, L, D)
        # y_q: (B, M, D)
        y_s = self.mamba_s(F_s_norm, delta_t=delta_t_s) 
        y_q = self.mamba_q(F_q_norm, delta_t=delta_t_q)
        
        # 3. Key and Value Projection for Scene Features
        Q = self.linear_Q(y_q)
        K = self.linear_K(y_s) # (B, L, D)
        V = self.linear_V(y_s) # (B, L, D)
        
        # --- 4. O(L) PARALLEL CROSS-INTERACTION ---
        # WHY THIS IS CAUSAL WITHOUT A MASK:
        # 1. Inherited Causality: Because of Mamba's state-space scan, the vector y_s[:, l, :] 
        #    (and consequently K[:, l, :] and V[:, l, :]) is already a compressed summary of 
        #    the causal history from time step 0 up to `l`. It contains absolutely no information 
        #    about future time steps (l+1, l+2, etc.).
        # 2. Localized Interaction: The einsum below computes the dot product between the queries 
        #    and the Keys strictly at the same time index `l`. It does not aggregate across the 
        #    sequence length `L`. 
        # Therefore, the update for step `l` only relies on K and V at step `l`, which themselves 
        # only know about the past. Causality is strictly maintained.
        
        # Compute dot-product similarity between queries and causal Keys.
        # Output shape: (B, L, M)
        scores = torch.einsum('bmd,bld->blm', Q, K) / math.sqrt(self.d_model)
        
        # Apply Sigmoid to act as an independent information gate for each query
        gates = torch.sigmoid(scores) # (B, L, M)
        
        # Apply the gate to the projected Values
        # V is (B, L, D) -> unsqueeze to (B, L, 1, D) for broadcasting
        # gates is (B, L, M) -> unsqueeze to (B, L, M, 1) for broadcasting
        # Output shape: (B, L, M, D)
        F_q_update = gates.unsqueeze(-1) * V.unsqueeze(2) 
        
        # 5. Residual Connection
        # Expand the original F_q (B, M, D) -> (B, 1, M, D) -> (B, L, M, D)
        out = F_q.unsqueeze(1) + F_q_update 
        
        return out
    
class MultiheadCausalQueryAwareMambaBlockv2(nn.Module):
    def __init__(self, d_model, num_queries=3, num_heads=8, d_state=128, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_heads = num_heads
        
        # Ensure d_model can be cleanly divided into heads
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.head_dim = d_model // num_heads
        
        # --- Learnable Queries ---
        self.learnable_query = nn.Parameter(torch.randn(num_queries, d_model))
        
        # --- Normalization ---
        self.norm_s = nn.LayerNorm(d_model)
        self.norm_q = nn.LayerNorm(d_model)
        
        # --- Independent Sequence Processing ---
        # (Assuming TimeScaleAwareMamba2 is defined elsewhere in your codebase)
        self.mamba_s = TimeScaleAwareMamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_q = TimeScaleAwareMamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        
        # --- Multi-Head Projections ---
        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_K = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)
        
        # We add an output projection to mix the information across the different 
        # heads after they have been independently gated.
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, F_s, delta_t_s=None, delta_t_q=None):
        B, L, D = F_s.shape
        M = self.num_queries
        H = self.num_heads
        Hd = self.head_dim
        
        # 0. Broadcast learnable query to match batch size: (B, M, D)
        F_q = self.learnable_query.unsqueeze(0).expand(B, -1, -1)
        
        # 1. Normalization
        F_s_norm = self.norm_s(F_s)
        F_q_norm = self.norm_q(F_q)
        
        # 2. Independent Causal Scanning
        y_s = self.mamba_s(F_s_norm, delta_t=delta_t_s) # (B, L, D)
        y_q = self.mamba_q(F_q_norm, delta_t=delta_t_q) # (B, M, D)
        
        # 3. Key and Value Projection
        Q = self.linear_Q(y_q) # (B, M, D)
        K = self.linear_K(y_s) # (B, L, D)
        V = self.linear_V(y_s) # (B, L, D)
        
        # --- 4. MULTI-HEAD O(L) CAUSAL CROSS-INTERACTION ---
        
        # Reshape to isolate the heads: (Batch, Seq/Query, Heads, Head_Dim)
        Q_multi = Q.view(B, M, H, Hd) 
        K_multi = K.view(B, L, H, Hd)     
        V_multi = V.view(B, L, H, Hd)     
        
        # Point-wise dot product interaction per head.
        # This computes the similarity between Queries and Keys for each head independently.
        # b=batch, m=queries, l=seq_len, h=heads, d=head_dim
        # Output shape: (B, L, M, H) - Notice 'd' is summed out by the dot product
        scores = torch.einsum('bmhd,blhd->blmh', Q_multi, K_multi) / math.sqrt(Hd)
        
        # Independent sigmoid gating per query, per head, per time step
        gates = torch.sigmoid(scores) # (B, L, M, H)
        
        # Apply gates to Values using broadcasting
        # V_multi needs an 'M' dimension: (B, L, 1, H, Hd)
        # gates needs a 'Hd' dimension:   (B, L, M, H, 1)
        V_expanded = V_multi.unsqueeze(2)
        gates_expanded = gates.unsqueeze(-1)
        
        # Broadcast multiply: (B, L, M, H, Hd)
        F_q_update_multi = gates_expanded * V_expanded
        
        # Flatten the heads back into the original d_model dimension: (B, L, M, D)
        F_q_update = F_q_update_multi.reshape(B, L, M, D)
        
        # Mix the concatenated heads together
        F_q_update = self.out_proj(F_q_update)
        
        # --- 5. Residual Connection ---
        # Expand the original F_q (B, M, D) -> (B, 1, M, D) so it broadcasts across L
        out = F_q.unsqueeze(1) + F_q_update 
        
        return out