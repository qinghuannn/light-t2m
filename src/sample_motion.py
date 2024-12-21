from typing import Any, Dict, List, Tuple

import hydra
import rootutils
import numpy as np
import torch
import lightning.pytorch as L
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

import json
import os

import os.path as osp
from os.path import join as pjoin

import codecs as cs

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

from tqdm import tqdm
from src.data.humanml.scripts.motion_process import recover_from_ric


@torch.no_grad()
def generation(model, cfg):
    mean = np.load(osp.join(cfg.data_dir, "Mean.npy"))
    std = np.load(osp.join(cfg.data_dir, "Std.npy"))
    
    if not os.path.exists(cfg.save_path):
        os.mkdir(cfg.save_path)
    if cfg.device != "cpu":
        model.to(f"cuda:{cfg.device}")
    model.eval()

    mean = torch.from_numpy(mean).to(model.device)
    std = torch.from_numpy(std).to(model.device)

    save_path = pjoin(cfg.save_path, "gen_joints")
    os.makedirs(save_path, exist_ok=True)
    
    texts = [cfg.text] * cfg.repeats
    motion_length = cfg.length
    
    motion = torch.zeros([1, motion_length, 263], device=model.device)
    motion = motion.repeat([len(texts), 1, 1]).to(model.device)
    lens = torch.tensor(motion.shape[1], dtype=torch.long, device=model.device).repeat([len(texts)])
    # print(lens.shape, motion.shape)
    gen_motions = model.sample_motion(motion, lens, texts)
    gen_motions = gen_motions * std + mean
    gen_joints = recover_from_ric(gen_motions, 22).cpu().numpy()
    
    name = cfg.sample_name
    for i in range(len(texts)):
        np.save(pjoin(save_path, name + f"_{i}.npy"), gen_joints[i])    



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

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)


    if cfg.ckpt_path is None or cfg.ckpt_path == "none":
        print("No ckpt!")
        exit()
    else:
        # print("loading ckpt from ", cfg.ckpt_path)
        state_dict = torch.load(cfg.ckpt_path, map_location="cpu")["state_dict"]
        keys_list = list(state_dict.keys())
        # print(keys_list)
        for key in keys_list:
            if 'orig_mod.' in key:
                deal_key = key.replace('_orig_mod.', '')
                state_dict[deal_key] = state_dict[key]
                del state_dict[key]
        # print("cur", list(model.state_dict().keys()))
        model.load_state_dict(state_dict, strict=False)


    num_parameters = sum([x.numel() for x in model.denoiser.parameters() if x.requires_grad])
    log.info("Total parameters: %.3fM" % (num_parameters / 1000_000))

    log.info("Starting generation!")

    generation(model, cfg)

    log.info("Done!")
    return {}, None



@hydra.main(version_base="1.3", config_path="../configs", config_name="sample_motion.yaml")
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
