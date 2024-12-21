import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

from einops import rearrange, repeat, reduce

from src.models.utils.embedding import timestep_embedding, TimestepEmbedding, PositionEmbedding

from mamba_ssm import Mamba


class LightT2M(nn.Module):
    def __init__(
            self,
            motion_dim=263,
            max_motion_len=196,
            text_dim=512,
            pos_emb="cos",
            dropout=-1,
            # stage
            stage_dim="256",
            num_groups=16,
            patch_size=8,
            # ssm config
            ssm_cfg=None,
            rms_norm=False,
            fused_add_norm=True,
    ):
        super().__init__()
        if "*" in stage_dim:
            base_dim = int(stage_dim.split("*")[0])
            stage_dim = [base_dim for _ in range(int(stage_dim.split("*")[1]))]
        else:
            stage_dim = [int(x) for x in stage_dim.split("-")]
            base_dim = stage_dim[0]

        if pos_emb == "cos":
            self.pos_emb = PositionEmbedding(max_motion_len, base_dim, dropout=0.1)
        elif pos_emb == "learn":
            self.pos_emb = PositionEmbedding(max_motion_len, base_dim, dropout=0.1, grad=True)
        else:
            raise ValueError(f"{pos_emb} not supported!")

        self.m_input_proj = nn.Linear(motion_dim, base_dim)
        self.t_input_proj = nn.Linear(text_dim, base_dim)
        self.time_emb = nn.Linear(base_dim, base_dim)

        create_mamba_block_fn = partial(create_mamba_block, ssm_cfg=ssm_cfg, norm_epsilon=1e-5, rms_norm=rms_norm,
                            residual_in_fp32=False, fused_add_norm=fused_add_norm, pre_norm=False)

        modules = []
        for i, cur_dim in enumerate(stage_dim):
            modules.append(StageBlock(base_dim, cur_dim, create_mamba_block_fn,
                                      mask_padding=True, num_groups=num_groups, patch_size=patch_size))
            base_dim = cur_dim

        self.layers = nn.ModuleList(modules)
        if dropout > 0:
            self.m_output_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(base_dim, motion_dim))
        else:
            self.m_output_proj = nn.Linear(base_dim, motion_dim)

    def forward(self, motion, motion_mask, timestep, text):
        motion = self.m_input_proj(motion)
        time_emb = self.time_emb(timestep_embedding(timestep, motion.shape[-1])).unsqueeze(dim=1)
        time_mask = torch.ones([time_emb.shape[0], 1], dtype=torch.bool, device=time_emb.device)

        text_feat = self.t_input_proj(text["text_emb"]).unsqueeze(dim=1)
        text_mask = torch.ones([text_feat.shape[0], 1], dtype=torch.bool, device=text_feat.device)

        x = torch.cat([time_emb, motion], dim=1)
        x_mask = torch.cat([time_mask, motion_mask], dim=1)
        x = self.pos_emb(x)

        for layer in self.layers:
            x, text_feat = layer(x, x_mask, text_feat, text_mask)

        out = x[:, 1:]

        out = self.m_output_proj(out)
        return out

class LocalModule(nn.Module):
    def __init__(self, model_dim, num_groups=16,  mask_padding=True):
        super().__init__()
        self.mask_padding = mask_padding

        self.conv = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, 1, 1, 0),
            nn.Conv1d(model_dim, model_dim, 3, 1, 1, groups=model_dim),
            nn.GroupNorm(num_groups=num_groups, num_channels=model_dim),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(model_dim)


    def forward(self, x, x_mask, y, y_mask, z=None):
        if self.mask_padding:
            x[~x_mask] = x[~x_mask] * torch.zeros_like(x[~x_mask], device=x.device)
            y[~y_mask] = y[~y_mask] * torch.zeros_like(y[~y_mask], device=x.device)

        x = self.norm(x + self.conv(x.permute([0, 2, 1])).permute([0, 2, 1]))

        return x, y

class MixedModule(nn.Module):
    def __init__(self, model_dim, build_mamba_block_fn, patch_size=8, mask_padding=True):
        super().__init__()
        self.patch_size = patch_size
        self.mask_padding = mask_padding

        self.local_conv = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, 1, 1, 0),
            nn.ReLU(),
            nn.Conv1d(model_dim, model_dim, patch_size, patch_size, 0, groups=model_dim)
        )
        self.global_mamba = build_mamba_block_fn(model_dim)
        self.final_fc = nn.Linear(model_dim * 2, model_dim)
        self.norm = nn.LayerNorm(model_dim)

        self.f_func = nn.Linear(model_dim * 2, model_dim)
        self.fuse_fn = nn.Linear(model_dim*2, model_dim)

    def inject_text(self, x, y):
        # x: [B, L ,D] y: [B, D]
        y_repeat = y.repeat([1, x.shape[1], 1])
        y_hat = self.f_func(torch.cat([x, y_repeat], dim=-1))
        _y_hat = y_repeat * torch.sigmoid_(y_hat)
        x_hat = self.fuse_fn(torch.cat([x, _y_hat], dim=-1))
        return x_hat


    def forward(self, x, x_mask, y, y_mask):
        if self.mask_padding:
            x[~x_mask] = x[~x_mask] * torch.zeros_like(x[~x_mask], device=x.device)
            y[~y_mask] = y[~y_mask] * torch.zeros_like(y[~y_mask], device=x.device)
        B, L, D = x.shape
        x1 = x[:, 1:]
        padding_size = x1.shape[1] % self.patch_size
        if padding_size != 0:
            x1 = torch.cat([x1, torch.zeros([B, self.patch_size - padding_size, D], device=x1.device)], dim=1)

        nx1 = self.local_conv(x1.permute([0, 2, 1])).permute([0, 2, 1])

        x2 = self.inject_text(nx1, y)

        nx2, _ = self.global_mamba(torch.cat([x2.flip([1]), x2], dim=1))
        x2 = nx2[:, x2.shape[1]:]

        x2 = repeat(x2, "B L D -> B (L S) D", S=self.patch_size)

        nx = self.final_fc(torch.cat([x1, x2], dim=-1))

        if padding_size != 0:
            nx = nx[:, :-(self.patch_size - padding_size)]
        out = torch.cat([x[:, :1], nx], dim=1)

        out = self.norm(out)
        return out, y

class StageBlock(nn.Module):
    def __init__(self, in_dim, dim, build_mamba_block_fn, mask_padding,
                 num_groups=16, patch_size=8):
        super().__init__()
        self.local_module1 = LocalModule(model_dim=dim, num_groups=num_groups, mask_padding=mask_padding)
        self.mixed_module = MixedModule(model_dim=dim, build_mamba_block_fn=build_mamba_block_fn,
                       patch_size=patch_size, mask_padding=mask_padding)
        self.local_module2 = LocalModule(model_dim=dim, num_groups=num_groups, mask_padding=mask_padding)

        self.input_proj = nn.Linear(in_dim, dim) if in_dim != dim else nn.Identity()

        if in_dim != dim:
            self.y_proj = nn.Linear(in_dim, dim)
        else:
            self.y_proj = nn.Identity()

    def forward(self, x, x_mask, y, y_mask):
        x = self.input_proj(x)
        y_ = self.y_proj(y)
        x, _ = self.local_module1(x, x_mask, y_, y_mask)
        x, _ = self.mixed_module(x, x_mask, y_, y_mask)
        x, _ = self.local_module2(x, x_mask, y_, y_mask)
        return x, y


try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from functools import partial
from torch import Tensor
from typing import Optional


class BaseMambaBlock(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,
        pre_norm=True,
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
        self.pre_norm = pre_norm
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, text_len=None, **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.pre_norm:
            return self.post_norm_forward(hidden_states, residual, inference_params, **mixer_kwargs)
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
                is_rms_norm=isinstance(self.norm, RMSNorm),
                text_len=text_len
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, text_len=text_len, **mixer_kwargs)

        return hidden_states, residual

    def post_norm_forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None,
                          inference_params=None,  **mixer_kwargs):
        new_hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        if not self.fused_add_norm:
            new_hidden_states = self.norm(hidden_states + new_hidden_states)
        else:
            new_hidden_states = layer_norm_fn(
                new_hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=hidden_states,
                prenorm=False,
                residual_in_fp32=False,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm),
            )

        return new_hidden_states, None

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_mamba_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    pre_norm=True,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx,  **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = BaseMambaBlock(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        pre_norm=pre_norm
    )
    block.layer_idx = layer_idx
    return block