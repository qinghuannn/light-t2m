import torch
import torch as th
import torch.nn as nn
from typing import List, Dict

import clip
from transformers import RobertaTokenizer, RobertaModel, logging
logging.set_verbosity_error()

from ..utils.utils import lengths_to_mask


class Roberta(torch.nn.Module):
    def __init__(self, freeze_lm=True, max_text_len=32):
        super(Roberta, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.lm = RobertaModel.from_pretrained("roberta-base")
        self.max_text_len = max_text_len
        if freeze_lm:
            for param in self.lm.parameters():
                param.requires_grad = False

    def forward(self, text, device, **kwargs):
        # set max_text_len to 2 to prevent nan output when only one token
        text_len = max(min(min([len(x) for x in text]), self.max_text_len), 2) + 1
        encoded_input = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=text_len,
        ).to(device)
        out = self.lm(**encoded_input)
        return {"text_emb": out["pooler_output"], "hidden": out["last_hidden_state"],
                "mask": encoded_input["attention_mask"].bool()}


class CLIP(torch.nn.Module):
    def __init__(self, freeze_lm=True):
        super(CLIP, self).__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device=torch.device('cpu'), jit=False)
        self.clip_model.visual = None
        # clip.model.convert_weights(self.clip_model)
        if freeze_lm:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    @property
    def dtype(self):
        return self.clip_model.text_projection.dtype

    def forward(self, text, device, **kwargs):
        text = clip.tokenize(text, truncate=True).to(device)
        x = self.clip_model.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.yaml.yaml.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        text_embed = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection
        mask = lengths_to_mask(text.argmax(dim=-1), device)
        if mask.shape[1] < x.shape[1]:
            x = x[:, :mask.size(1)]
        return {"text_emb": text_embed, "hidden": x, "mask": mask}
