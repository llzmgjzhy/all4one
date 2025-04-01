import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Qwen2_5_VLModel,
    Qwen2_5_VLConfig,
    Qwen2_5_VLProcessor,
    Qwen2_5_VisionTransformerPretrainedModel,
)
import numpy as np
from einops import rearrange
from models.ts2vec import TS2Vec
from math import sqrt


class All4One(nn.Module):
    def __init__(self, config, data):
        super(All4One, self).__init__()
        self.seq_len = data.max_seq_len
        self.max_len = data.max_seq_len
        self.patch_size = config["patch_size"]
        self.stride = config["stride"]
        self.gpt_layers = 6
        self.d_piece = data.d_piece
        self.num_classes = len(data.class_names)
        self.d_model = config["d_model"]

        # model
        self.llm_model = Qwen2_5_VLModel.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            output_attentions=True,
            output_hidden_states=True,
        )
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(
            Qwen2_5_VLConfig
        )

        self.ts2vec = TS2Vec(input_dims=self.d_piece, device=config["device"], **config)

        # word embedding
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(
            config.d_model, config.n_heads, self.d_ff, self.d_llm
        )

    def forward(self, x_enc):
        B, T, N = x_enc.shape

        # time series embedding
        x_enc = self.ts2vec.encode(
            x_enc,
            encoding_window="full_series",
        )

        # align time series and text
        source_embeddings = self.mapping_layer(
            self.word_embeddings.permute(1, 0)
        ).permute(1, 0)

        enc_out = self.reprogramming_layer(x_enc, source_embeddings, source_embeddings)

        # image create and embedding
        image_embeds = self.visual(x_enc, return_dict=True, output_hidden_states=True)

        llm_enc_out = torch.cat([image_embeds, enc_out], dim=1)
        dec_out = self.llm_model(
            inputs_embeds=llm_enc_out,
        )

        # downstream task
        out = out[:, 0, :]
        out = self.ln_proj(out)
        out = self.dropout(out)
        out = self.out_layer(out)

        return out


class ReprogrammingLayer(nn.Module):
    def __init__(
        self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1
    ):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1.0 / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
