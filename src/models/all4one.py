import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLModel,
    Qwen2_5_VLConfig,
    Qwen2_5_VLProcessor,
)
import numpy as np
from einops import rearrange
from models.ts2vec import TS2Vec
from math import sqrt
from models.TimeLLM import ReprogrammingLayer
from layers.StandardNorm import Normalize


class FlattenHead(nn.Module):
    def __init__(self, nf, seq_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, seq_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x.unsqueeze(-1)


class ALL4ONE(nn.Module):
    def __init__(self, config, device):
        super(ALL4ONE, self).__init__()
        self.seq_len = config.seq_len
        self.patch_size = config.patch_size
        self.stride = config.stride
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.d_llm = config.llm_dim

        # model
        # self.llm_model = Qwen2_5_VLModel.from_pretrained(
        #     "Qwen/Qwen2.5-VL-7B-Instruct",
        #     output_attentions=True,
        #     output_hidden_states=True,
        # )
        # for param in self.llm_model.parameters():
        #     param.requires_grad = False

        self.ts2vec = TS2Vec(config=config, input_dims=1, output_dims=1, device=device)

        # outprojection
        if config.task == "forecast":
            self.output_projection = FlattenHead(
                nf=config.seq_len, seq_window=config.pred_len
            )

        self.normalize_layers = Normalize(config.enc_in, affine=False)

    def get_ts2vec_loss(self, x_enc):
        return self.ts2vec.fit(x_enc)

    def forward(self, x_enc, x_mask, y, y_mask):
        B, T, N = x_enc.shape
        x_enc = self.normalize_layers(x_enc, "norm")

        # time series embedding
        x_enc = self.ts2vec.encode(
            x_enc,
            # encoding_window="full_series",
            # causal=True,
            # sliding_length=1,
            # sliding_padding=200
        )
        x_enc = self.output_projection(x_enc)
        x_enc = self.normalize_layers(x_enc, "denorm")

        return x_enc
