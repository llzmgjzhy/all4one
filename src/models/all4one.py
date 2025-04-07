import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Qwen2VLImageProcessorFast,
    Qwen2_5_VLForConditionalGeneration,
)
from einops import rearrange
from models.ts2vec import TS2Vec
from math import sqrt
from models.TimeLLM import ReprogrammingLayer
from layers.StandardNorm import Normalize
from models.util import tensor_line_plots


class FlattenHead(nn.Module):
    def __init__(self, seq_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(in_features=82 * 3584, out_features=seq_window)
        # self.linear = nn.Linear(in_features=1 * 3584, out_features=seq_window)
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
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        for param in self.model.parameters():
            param.requires_grad = False
        self.llm_model = self.model.model

        self.ts2vec = TS2Vec(
            config=config, input_dims=1, output_dims=1, device=device
        ).bfloat16()

        # visual module
        self.visual = self.model.visual
        self.image_processor = Qwen2VLImageProcessorFast()

        # ts2vec embedding align llm input dims
        self.ts2vec_embedding = nn.Linear(config.seq_len, config.llm_dim).to(
            dtype=torch.bfloat16, device=device
        )

        # outprojection
        if config.task == "forecast":
            self.output_projection = FlattenHead(
                seq_window=config.pred_len,
                head_dropout=config.dropout,
            ).to(dtype=torch.bfloat16, device=device)

        self.normalize_layers = Normalize(config.enc_in, affine=False)

    def get_ts2vec_loss(self, x_enc):
        return self.ts2vec.fit(x_enc)

    def forward(self, x_enc, x_mask, y, y_mask):
        B, T, N = x_enc.shape

        # create image tensor
        x_image = tensor_line_plots(x_enc)  # [B, 3, H, W]
        image_inputs = self.image_processor(
            images=x_image, videos=None, do_rescale=False
        ).to(device=x_enc.device, dtype=x_enc.dtype)
        images_embeds = self.visual(
            image_inputs["pixel_values"], grid_thw=image_inputs["image_grid_thw"]
        )
        images_embeds = images_embeds.reshape(B, -1, images_embeds.shape[-1])

        x_enc = self.normalize_layers(x_enc, "norm")
        # time series embedding
        x_enc = self.ts2vec.encode(
            x_enc,
            # encoding_window="full_series",
            # causal=True,
            # sliding_length=1,
            # sliding_padding=200
        )
        x_enc = self.ts2vec_embedding(x_enc.permute(0, 2, 1))

        llm_enc_out = torch.cat(
            [images_embeds, x_enc], dim=1
        )  # [B, token_num , llm_dim]
        dec_out = self.llm_model(inputs_embeds=llm_enc_out).last_hidden_state

        # output
        dec_out = self.output_projection(dec_out)  # [B, pred_len, 1]
        dec_out = self.normalize_layers(dec_out, "denorm")

        return dec_out
