from typing import Any, Dict, List, Tuple

import hydra
import rootutils
import numpy as np
import torch

import tqdm
# from thop import clever_format, profile

import lightning.pytorch as L
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

import json
import os
import codecs as cs

import os.path as osp
from functools import partial


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def prepare_data(cfg, device):
    data_file = np.load(os.path.join(cfg.data_dir, "data_test.npy"), allow_pickle=True).item()
    mean = np.load(osp.join(cfg.data_dir, "Mean.npy"))
    std = np.load(osp.join(cfg.data_dir, "Std.npy"))
    motion_info = []
    if cfg.test_mode == "random_motion":
        randomly_selected_data = np.load(os.path.join(cfg.random_selected_file))
    else:
        randomly_selected_data = [cfg.long_motion_idx, ]
    for name in randomly_selected_data:
        info = {"texts": []}
        motion = data_file[name]["motion"]
        motion = (motion - mean) / std
        if len(motion) % 4 != 0:
            motion = motion[:4 * (len(motion) // 4)]
        info["name"] = name
        info["texts"] = [data_file[name]["text"][0]["caption"]]
        info["motion"] = torch.from_numpy(motion).to(device).unsqueeze(dim=0)
        info["len"] = torch.tensor([len(motion), ], dtype=torch.long, device=device)
        if cfg.tgt_length != "none":
            info["motion"] = torch.randn([1, cfg.tgt_length, motion.shape[-1]], device=device)
            info["len"] = torch.tensor([cfg.tgt_length, ], dtype=torch.long, device=device)
        motion_info.append(info)

    return motion_info, mean, std

def run_net(model, motion_info, run_id):

    for i in range(len(motion_info)):
        motion = motion_info[i]["motion"]
        text = motion_info[i]["texts"]
        length = motion_info[i]["len"]
        model.sample_motion(motion, length, text)


@torch.no_grad()
def calculate_run_time(_run, repetitions):
    torch.backends.cudnn.benchmark = True

    # warm up, GPU may be in sleep mode for energy saving, so need to warm up
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(10):
            _run(run_id=0)
            # _ = model(model_input[0],model_input[1])

    # wait for GPU to finish the warm up
    torch.cuda.synchronize()

    # Set up cuda Event for measuring time, this is the official PyTorch recommended interface, theoretically the most reliable 
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # Initialize a timing container
    timings = np.zeros((repetitions, 1))

    print('testing ...\n')
    with torch.no_grad():
        for rep in tqdm.tqdm(range(repetitions)):
            starter.record()
            # _ = model(model_input[0],model_input[1])
            _run(run_id=rep)
            ender.record()
            torch.cuda.synchronize()  # Waiting for GPU tasks to complete
            curr_time = starter.elapsed_time(ender)  # Time elapsed between starter and ender, in milliseconds
            timings[rep] = curr_time

    avg = timings.sum() / repetitions
    print('\navg={}ms\n'.format(avg))



def calculate_flops(model, motion_info):
    from thop import profile, clever_format
    import torch
    from src.models.utils.utils import calculate_mamba_flops_for_bimamba, lengths_to_mask, calculate_mamba_flops_for_mamba
    from mamba_ssm import Mamba, BiMamba

    motion = motion_info[3]["motion"]
    text = motion_info[3]["texts"]
    length = motion_info[3]["len"]
    # length = torch.tensor([196,], dtype=torch.long, device=model.device)
    # motion = torch.randn([1, length[0], motion.shape[-1]], device=model.device)
    # print("motion length: ", length)

    with torch.no_grad():
        text_embed = model.text_encoder(text, model.device)

    timestep = torch.randint(0, model.noise_scheduler.config.num_train_timesteps,
                             (motion.size(0),), device=motion.device).long()
    padding_mask = lengths_to_mask(length, model.device)
    noise = torch.randn_like(motion, device=motion.device)
    x_t = model.noise_scheduler.add_noise(motion, noise, timestep)
    # output = model.denoiser(x_t, padding_mask, timestep, text_embed)

    macs, params = profile(model.denoiser, inputs=(x_t, padding_mask, timestep, text_embed),
                           custom_ops={BiMamba: calculate_mamba_flops_for_bimamba,
                                       Mamba: calculate_mamba_flops_for_mamba})
    macs, params = clever_format([macs * 2, params], "%.3f")
    print("FLOPs: %s, Params: %s" % (macs, params))
    import time
    time.sleep(10)


def calculate_gpu_memory(cfg, model):
    motion_info, _, _ = prepare_data(cfg, model.device)
    cur_data = None
    for i in range(len(motion_info)):
        # print(len(motion_info[i]["motion"]))
        if motion_info[i]["motion"].shape[1]  == 196:
            cur_data = motion_info[i]
            break
    print("test sample name: ", cur_data["name"])
    bz = 512
    cur_data["motion"] = cur_data["motion"].repeat([bz, 1, 1])
    # print(cur_data["texts"])
    cur_data["texts"] = cur_data["texts"] * bz
    cur_data["len"] = cur_data["len"].repeat([bz,])
    # cur_data["motion_mask"] = cur_data["motion_mask"].repeat(bz, 1)
    run_net(model, [cur_data], run_id=0)

    # Get maximum memory reserved and cached
    max_reserved = torch.cuda.max_memory_reserved(model.device) / (1024 ** 2)  # MB
    max_cached = torch.cuda.max_memory_cached(model.device) / (1024 ** 2)      # MB
    print(f"max_memory_reserved: {max_reserved:.2f} MB")
    print(f"max_memory_cached: {max_cached:.2f} MB")


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    torch.set_float32_matmul_precision('high')

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    # datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    # trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)


    # if logger:
    #     log.info("Logging hyperparameters!")
    #     log_hyperparameters(object_dict)

    # state_dict = torch.load(cfg.ckpt_path, map_location="cpu")["state_dict"]
    # keys_list = list(state_dict.keys())
    # for key in keys_list:
    #     if 'orig_mod.' in key:
    #         deal_key = key.replace('_orig_mod.', '')
    #         state_dict[deal_key] = state_dict[key]
    #         del state_dict[key]
    # model.load_state_dict(state_dict, strict=False)
    model.metrics = None

    if cfg.device != "cpu":
        model.to(f"cuda:{cfg.device}")


    log.info("Starting testing!")

    # calculate_gpu_memory(cfg, model)
    # exit()

    motion_info, _, _ = prepare_data(cfg, model.device)

    calculate_flops(model, motion_info)
    run_func = partial(run_net, model=model, motion_info=motion_info)

    num_parameters = sum([x.numel() for x in model.denoiser.parameters() if x.requires_grad])
    log.info("Total trainable parameters: %.3fM" % (num_parameters / 1000_000))

    # calculate_model_size(model, motion_info)
    calculate_run_time(run_func, cfg.repeats)

    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="test_speed.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
