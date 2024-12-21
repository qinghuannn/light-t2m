import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn

from .t2m_motionenc import MotionEncoderBiGRUCo, MovementConvEncoder
from .t2m_textenc import TextEncoderBiGRUCo


class T2MEvaluator(object):
    def __init__(self, dataset='hml3d', deps_dir="../deps/t2m_guo"):
        # hml3d == t2m
        self.dataset = 't2m' if dataset == 'hml3d' else dataset
        self.unit_length = 4
        ckpt_dir = os.path.join(deps_dir, self.dataset)
        motion_dim = 263 if self.dataset == "t2m" else 251
        self.t2m_textenc = TextEncoderBiGRUCo(word_size=300, pos_size=15, hidden_size=512, output_size=512)
        self.t2m_moveenc = MovementConvEncoder(input_size=motion_dim - 4, hidden_size=512, output_size=512)
        self.t2m_motionenc = MotionEncoderBiGRUCo(input_size=512, hidden_size=1024, output_size=512)
        t2m_checkpoint = torch.load(osp.join(ckpt_dir, "text_mot_match/model/finest.tar"), map_location='cpu')
        self.t2m_textenc.load_state_dict(t2m_checkpoint["text_encoder"])
        self.t2m_moveenc.load_state_dict(t2m_checkpoint["movement_encoder"])
        self.t2m_motionenc.load_state_dict(t2m_checkpoint["motion_encoder"])
        # for t2m, mean.npy and std.npy are copied from t2m/Comp_v6_KLD01/meat/
        # for kit, mean.npy and std.npy are copied from kit/Comp_v6_KLD005/meat/
        self.mean = torch.tensor(np.load(osp.join(ckpt_dir, 'text_mot_match/mean.npy')))
        self.std = torch.tensor(np.load(osp.join(ckpt_dir, 'text_mot_match/std.npy')))

    # @torch.autocast(device_type='cuda', dtype=torch.float32)
    @torch.no_grad()
    def extract_embedding(self, motion, motion_len, word_embs, pos_ohot, text_lengths):
        text_embed = self.extract_text_embedding(word_embs=word_embs, pos_ohot=pos_ohot, text_lengths=text_lengths)
        motion_embed = self.extract_motion_embedding(motion, motion_len)
        return motion_embed, text_embed

    @torch.autocast(device_type='cuda', dtype=torch.float32)
    @torch.no_grad()
    def extract_text_embedding(self, word_embs, pos_ohot, text_lengths, device):
        self.t2m_textenc.to(word_embs.device)
        self.t2m_textenc.eval()
        align_idx = np.argsort(text_lengths.data.tolist())[::-1].copy()
        inv_align_idx = np.argsort(align_idx)
        text_embed = self.t2m_textenc(word_embs[align_idx], pos_ohot[align_idx], text_lengths[align_idx])[inv_align_idx]
        return text_embed

    @torch.autocast(device_type='cuda', dtype=torch.float32)
    @torch.no_grad()
    def extract_motion_embedding(self, motion, motion_len):
        self.t2m_moveenc.to(motion.device)
        self.t2m_motionenc.to(motion.device)
        self.t2m_moveenc.eval()
        self.t2m_motionenc.eval()
        device = motion.device
        motion = (motion - self.mean.to(device)) / self.std.to(device)
        motion = motion.float()
        align_idx = np.argsort(motion_len.data.tolist())[::-1].copy()
        inv_align_idx = np.argsort(align_idx)
        cur_motion = motion[align_idx]
        length = torch.div(motion_len[align_idx], self.unit_length, rounding_mode="floor")
        motion_embed = self.t2m_motionenc(self.t2m_moveenc(cur_motion[..., :-4]), length)[inv_align_idx]
        # length = torch.div(motion_len, self.unit_length, rounding_mode="floor")
        # motion_embed = self.t2m_motionenc(self.t2m_moveenc(motion[..., :-4]), length)
        return motion_embed