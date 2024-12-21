import codecs as cs
import os.path
import random
import json
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils import data

from .scripts.motion_process import process_file, recover_from_ric
from .scripts.word_vectorizer import WordVectorizer

class T2MDataset(data.Dataset):
    def __init__(
        self,
        data_file,
        njoints,
        w_vectorizer_path,
        repeat_dataset=1,
        augmentation=True,
        uint_length=4,
        max_text_len=20,
        **kwargs
    ):
        self.njoints = njoints
        self.augmentation = augmentation
        self.is_train = "train" in os.path.basename(data_file)

        self.data_dict = np.load(data_file, allow_pickle=True).item()
        self.idx2name = {idx: key for idx, key in enumerate(self.data_dict)}

        self.mean, self.std = kwargs["mean"], kwargs["std"]

        self.repeat_dataset = repeat_dataset

        self.uint_length = uint_length
        self.w_vectorizer = WordVectorizer(w_vectorizer_path, "our_vab")
        self.max_text_len = max_text_len # for hml3d and kit

        self.is_mm = False

        self.prepare()
        print("num:", self.__len__())

    def prepare(self):
        for key in self.data_dict:
            # "Z Normalization"
            self.data_dict[key]["motion"] = (self.data_dict[key]["motion"] - self.mean) / self.std

    def __len__(self):
        return len(self.idx2name) * self.repeat_dataset

    def __getitem__(self, idx):
        data = self.data_dict[self.idx2name[idx % len(self.idx2name)]]
        text_list = data["text"]
        motion = data["motion"].copy()
        m_length = motion.shape[0]

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        # only used during val and test
        word_embeddings = np.array([0])
        pos_one_hots = np.array([0])
        sent_len = 0

        if not self.is_train:
            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
                sent_len = len(tokens)
                tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
                sent_len = len(tokens)

            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.augmentation:
            if np.random.randint(0, 3) == 0:
                m_length = (m_length // self.uint_length - 1) * self.uint_length
            else:
                m_length = (m_length // self.uint_length) * self.uint_length
            idx = np.random.randint(0, len(motion) - m_length + 1)
            motion = motion[idx: idx + m_length]

        return motion,  len(motion), caption, word_embeddings, pos_one_hots, sent_len

    def inv_transform(self, data):
        mean = torch.tensor(self.mean).to(data)
        std = torch.tensor(self.std).to(data)
        return data * std + mean

    # update
    def feats2joints(self, features):
        mean = torch.tensor(self.mean).to(features)
        std = torch.tensor(self.std).to(features)

        features = features * std + mean
        return recover_from_ric(features, self.njoints)


class T2MDataset2(data.Dataset):

    def __init__(
        self,
        dataset_name,
        data_dir,
        mean,
        std,
        split_file,
        njoints,
        w_vectorizer_path,
        max_motion_length=196,
        max_text_len=20,
        unit_length=4,
        repeat_dataset=1,
        **kwargs
    ):
        self.w_vectorizer = WordVectorizer(w_vectorizer_path, "our_vab")
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = 196
        # min_motion_len = 40 if dataset_name =='t2m' else 24
        self.min_motion_length = 40 if dataset_name =='hml3d' else 24
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        motion_dir = os.path.join(data_dir, "new_joint_vecs")
        self.motion_dir = motion_dir
        text_dir = os.path.join(data_dir, "texts")

        data_dict = {}
        with cs.open(split_file, "r") as f:
            self.id_list = [line.strip() for line in f.readlines()]

        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerate(self.id_list):
            try:
                motion = np.load(pjoin(motion_dir, name + ".npy"))
                if (len(motion)) < self.min_motion_length or (len(motion) >= 200):
                    bad_count += 1
                    continue
                text_data = []
                flag = False
                with (cs.open(pjoin(text_dir, name + ".txt")) as f):
                    for j, line in enumerate(f.readlines()):
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        # update
                        # if caption[-1] != ".":
                        #     caption += "."
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag * 20)]
                                if (len(n_motion)) < self.min_motion_length or ((len(n_motion) >= 200)):
                                    continue
                                new_name = (random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                # None
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    # print(count)
                    count += 1
                    # print(name)
            except:
                pass
        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        self.mean, self.std = mean, std
        self.nfeats = motion.shape[1]
        self.njoints = njoints
        # update
        # self.is_mm = False

        self.is_train = os.path.basename(split_file) == "train.txt"
        self.repeat_dataset = repeat_dataset
        print("num", self.__len__())

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        # print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def __len__(self):
        return (len(self.name_list) - self.pointer) * self.repeat_dataset

    def __getitem__(self, item):
        data_idx = self.pointer + item % (len(self.name_list) - self.pointer)
        data = self.data_dict[self.name_list[data_idx]]
        motion, m_length, text_list = data["motion"].copy(), data["length"], data["text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        "Z Normalization"
        motion = (motion - self.mean) / self.std
        return motion,  len(motion), caption, word_embeddings, pos_one_hots, sent_len

    def inv_transform(self, data):
        mean = torch.tensor(self.mean).to(data)
        std = torch.tensor(self.std).to(data)
        return data * std + mean

    # update
    def feats2joints(self, features):
        mean = torch.tensor(self.mean).to(features)
        std = torch.tensor(self.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def joints2feats(self, features):
        features = process_file(features, self.njoints)[0]
        mean = torch.tensor(self.mean).to(features)
        std = torch.tensor(self.std).to(features)
        features = (features - mean) / std
        return features
