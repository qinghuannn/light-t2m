# -*- coding: utf-8 -*-
import math
import torch
from torch import nn


# from flame
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# from MLD
class TimestepEmbedding(nn.Module):
    def __init__(self, channel: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()

        self.linear_1 = nn.Linear(channel, time_embed_dim)
        self.act = None
        if act_fn == "silu":
            self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample):
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)
        return sample


def PE1d_sincos(seq_length, dim):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if dim % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(dim))
    pe = torch.zeros(seq_length, dim)
    position = torch.arange(0, seq_length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                          -(math.log(10000.0) / dim)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe.unsqueeze(1)


class PositionEmbedding(nn.Module):
    """
    Absolute pos embedding (standard), learned.
    """

    def __init__(self, seq_length, dim, dropout, grad=False, randn_norm=False):
        super().__init__()
        if randn_norm:
            self.embed = nn.Parameter(torch.randn(seq_length, 1, dim), requires_grad=randn_norm)
        else:
            self.embed = nn.Parameter(data=PE1d_sincos(seq_length, dim), requires_grad=grad)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, shift=0):
        # x.shape: bs, seq_len, feat_dim
        l = x.shape[1]
        x = x.permute(1, 0, 2) + self.embed[shift:shift+l].expand(x.permute(1, 0, 2).shape)
        x = self.dropout(x.permute(1, 0, 2))
        return x


class PartialPositionEmbedding(nn.Module):
    """
    Absolute pos embedding (standard), learned.
    """

    def __init__(self, seq_length, dim, dropout=0, grad=False, randn_norm=False):
        super().__init__()
        if randn_norm:
            self.embed = nn.Parameter(torch.randn(seq_length, 1, dim), requires_grad=randn_norm)
        else:
            self.embed = nn.Parameter(data=PE1d_sincos(seq_length, dim), requires_grad=grad)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, pos_idx):
        # x: [B, L , D], pos_idx: [B, L]
        partial_embed = torch.index_select(self.embed, dim=0, index=pos_idx.view(-1))
        partial_embed = partial_embed.view(pos_idx.shape[0], pos_idx.shape[1], -1)
        x = x + partial_embed
        x = self.dropout(x)
        return x