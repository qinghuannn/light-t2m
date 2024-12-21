# -*- coding: utf-8 -*-
from typing import List

import numpy as np
import torch
import os
from torch import Tensor


# from flame

def remove_special_characters(text: str, special_characters=[",", ".", "/"]):
    for special_character in special_characters:
        text = text.replace(special_character, "")
        text = text.replace(" ", "_")

    return text


def replace_annotation_with_null(annotation: List[str], replace_prob: float):
    """
    Replace annotations with empty string.
    """

    replaced_annotation = list(annotation)
    replace_prob = (
        np.random.uniform(low=0.0, high=1.0, size=len(annotation)) < replace_prob
    )
    for replace_ind, to_be_replaced in enumerate(replace_prob):
        if to_be_replaced:
            replaced_annotation[replace_ind] = ""

    return replaced_annotation


def lengths_to_mask(lengths: List[int], device: torch.device) -> Tensor:
    """
    Generate mask array.
    """
    lengths = lengths.clone().detach()
    max_len = max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)
    return mask


def mask_before_summery(func, x, mask):
    if len(x.shape) == len(mask.shape) + 1:
        return func(x[mask].sum() / (mask.sum() * x.shape[-1]))
    return func(x[mask].sum() / mask.sum())


def normal_kl(mean1, logvar1, mean2, logvar2, timestemps=None):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    kl_div = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                    + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))
    if timestemps is None:
        return kl_div
    kl_div[timestemps==0] = 0
    return kl_div


def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used


def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    del x


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, warmup=0, eta_min=1e-5):
        self.warmup = warmup
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer)

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            if self.last_epoch >= self.warmup:
                lrs.append(self.eta_min
                           + (base_lr - self.eta_min) *
                           (1 + np.cos(np.pi * (self.last_epoch - self.warmup) / (self.T_max - self.warmup))) / 2)
            else:
                lrs.append(base_lr * (self.last_epoch * 1.0 / self.warmup))
        return lrs


class SimpleDict(dict):
    def __init__(self, d=None):
        if d is not None:
            for k,v in d.items():
                self[k] = v
        return super().__init__()

    def __key(self, key):
        return "" if key is None else key.lower()

    def __str__(self):
        import json
        return json.dumps(self)

    def __setattr__(self, key, value):
        self[self.__key(key)] = value

    def __getattr__(self, key):
        return self.get(self.__key(key))

    def __getitem__(self, key):
        return super().get(self.__key(key))

    def __setitem__(self, key, value):
        return super().__setitem__(self.__key(key), value)


def select_partial_motion(motion, prediction_map, mask_rest=True):
    device = motion.device
    B, L, _ = motion.shape
    # pmap = prediction_map.int()
    value, index = torch.sort((torch.arange(L, device=device)[None, :] - L - 1) * prediction_map, dim=-1)
    max_len = prediction_map.sum(dim=-1).max()
    partial_pos = index[:, :max_len]
    partial_mask = (value == 0)[:, :max_len]
    partial_motion = motion[torch.arange(B, device=device)[:, None], partial_pos]
    partial_motion[partial_mask] = 0

    return partial_motion, partial_pos, partial_mask

def calculate_mamba_flops_for_bimamba(m, hidden_states, inference_params=None, text_len=None, **kwargs):
    # print(hidden_states)
    # print(hidden_states[0].shape, inference_params, text_len)
    # https://github.com/state-spaces/mamba/issues/303
    B, L, D = hidden_states[0].shape
    flops = 0
    if m.bimamba_type == "v1":
        pass
    elif m.bimamba_type == "v2":
        flops += B * D * L # with D
        flops += B * D * L # with Z
        flops += 9 * B * L * D * m.d_inner
        flops *= 2
    else:
        flops += B * D * L  # with D
        flops += B * D * L # with Z
        flops += 9 * B * L * D * m.d_inner
    # add flops of in_proj
    flops += B * L * D * m.d_inner * 2
    # add flops of out_proj
    flops += B * L * D * m.d_inner
    # add flops of conv1d, kernel_size=4, groups=d_inner
    flops += B * L * D * m.d_inner * 4
    # divided by 2 because we count MAC (multiply-add counted as one flop)
    m.total_ops += flops / 2

# for mamba
def calculate_mamba_flops_for_mamba(m, hidden_states, inference_params=None, text_len=None, **kwargs):
    # print(hidden_states)
    # print(hidden_states[0].shape, inference_params, text_len)
    # https://github.com/state-spaces/mamba/issues/303
    B, L, D = hidden_states[0].shape
    flops = 0
    # selective scan
    flops += B * D * L  # with D
    flops += B * D * L # with Z
    flops += 9 * B * L * D * m.d_inner
    # add flops of in_proj
    flops += B * L * D * m.d_inner * 2
    # add flops of out_proj
    flops += B * L * D * m.d_inner
    # add flops of conv1d, kernel_size=4, groups=d_inner
    flops += B * L * D * m.d_inner * 4
    # divided by 2 because we count MAC (multiply-add counted as one flop)
    m.total_ops += flops / 2



if __name__ == '__main__':
    x = torch.arange(3*3*4).view([3, 4, 3])
    pmap = torch.tensor([[1, 1, 1, 0], [1, 0, 1, 0], [0, 0, 0, 0]])
    gmotion, gpos, gmask = select_partial_motion(x, pmap==1)
    print(gmotion)
    print(gpos)
    print(gmask)
    print("="*50)
    print(gmotion[~gmask])
    print(x[pmap==1])


