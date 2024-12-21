import os.path as osp
import copy


import numpy as np
import torch
from lightning.pytorch import LightningDataModule
# from lightning_utilities.core.rank_zero import rank_zero_only
from torch.utils.data import DataLoader, Dataset

from .utils import gcn_collate
from .humanml.dataset import T2MDataset, T2MDataset2


class HumanML3DDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        val_batch_size: int = -1,
        test_batch_size: int = 1,
        num_workers: int = 16,
        pin_memory: bool = False,
        njoints: int = 22,
        motion_dim: int = 263,
        augmentation: bool = True,
        w_vectorizer_path: str = '',
        dataset_name: str = 'hml3d',
        repeat_dataset: int = 1,
        old_dataset: bool=False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        if dataset_name == "hml3d":
            self.data_dir = osp.join(data_dir, "HumanML3D")
        else:
            # self.data_dir = osp.join(data_dir, "KIT")
            self.data_dir = osp.join(data_dir, "KIT-ML")
        self.njoints = njoints
        self.dataloader_options = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": False,
            "collate_fn": gcn_collate
        }

        self.name = dataset_name
        self.dataset = T2MDataset if not old_dataset else T2MDataset2
        self.w_vectorizer_path = w_vectorizer_path
        self.augmentation = augmentation

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: None):
        self.dataset_kwargs = {
            "njoints": self.njoints,
            "augmentation": self.augmentation,
            "w_vectorizer_path": self.w_vectorizer_path,
        }
        self.dataset_kwargs.update(self.get_mean_std())

    def train_dataloader(self):
        if self.train_dataset is None:
            if self.hparams.old_dataset:
                self.train_dataset = self.dataset(split_file=osp.join(self.data_dir, "train.txt"),
                                                  data_dir=self.data_dir, dataset_name=self.name,
                                                  repeat_dataset=self.hparams.repeat_dataset, **self.dataset_kwargs)
            else:
                self.train_dataset = self.dataset(data_file=osp.join(self.data_dir, "data_train.npy"),
                                                repeat_dataset=self.hparams.repeat_dataset, **self.dataset_kwargs)
        options = self.dataloader_options.copy()
        options["batch_size"] = self.hparams.batch_size
        return DataLoader(dataset=self.train_dataset, shuffle=True, drop_last=True, **options)

    def val_dataloader(self):
        if self.val_dataset is None:
            if self.hparams.old_dataset:
                self.val_dataset = self.dataset(split_file=osp.join(self.data_dir, "val.txt"),
                                                  data_dir=self.data_dir, dataset_name=self.name,
                                                   **self.dataset_kwargs)
            else:
                self.val_dataset = self.dataset(data_file=osp.join(self.data_dir, "data_val.npy"),
                                                    **self.dataset_kwargs)
        options = self.dataloader_options.copy()
        options["batch_size"] = self.hparams.val_batch_size
        if options["batch_size"] == -1:
            options["batch_size"] = self.hparams.batch_size
        return DataLoader(dataset=self.val_dataset, shuffle=False, drop_last=False, **options)

    def test_dataloader(self):
        if self.test_dataset is None:
            if self.hparams.old_dataset:
                self.test_dataset = self.dataset(split_file=osp.join(self.data_dir, "test.txt"),
                                                  data_dir=self.data_dir, dataset_name=self.name,
                                                   **self.dataset_kwargs)
            else:
                self.test_dataset = self.dataset(data_file=osp.join(self.data_dir, "data_test.npy"),
                                                         **self.dataset_kwargs)
            self.test_dataset.is_mm = False

        options = self.dataloader_options.copy()
        options["batch_size"] = self.hparams.test_batch_size

        return DataLoader(dataset=self.test_dataset, shuffle=True, drop_last=False, **options)
        # return DataLoader(dataset=self.test_dataset, shuffle=False, drop_last=True, **options)

    def mm_mode(self, mm_on=True, mm_num_samples=100):
        # random select samples for mm
        if mm_on:
            self.idx2name = copy.deepcopy(self.test_dataset.idx2name)
            idx_key_lists = np.random.choice(list(self.test_dataset.idx2name.keys()), mm_num_samples, replace=False)
            self.mm_list = {idx: self.test_dataset.idx2name[key] for idx, key in enumerate(idx_key_lists)}
            self.test_dataset.idx2name = self.mm_list
            self.test_dataset.is_mm = True
        else:
            self.test_dataset.is_mm = False
            self.test_dataset.idx2name = self.idx2name

    def get_mean_std(self):
        data_mean = np.load(osp.join(self.data_dir, "Mean.npy"))
        data_std = np.load(osp.join(self.data_dir, "Std.npy"))
        return {"mean": data_mean, "std": data_std}



if __name__ == '__main__':
    datamodule = HumanML3DDataModule("./data")
    datamodule.setup(None)
    dataloader = datamodule.test_dataloader()
    for i, data in enumerate(dataloader):
        # motion, text, length, word_embs, pos_ohot, text_len, tokens = data
        print(data["motion"].shape, data["text"], data["length"], data["text_len"])
        break