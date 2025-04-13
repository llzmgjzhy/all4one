import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Qwen2VLImageProcessorFast,
    Qwen2_5_VLProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from einops import rearrange
from models.ts2vec import TS2Vec
from math import sqrt
from models.TimeLLM import ReprogrammingLayer
from models.embed import TokenEmbedding, PatchEmbedding
from layers.StandardNorm import Normalize
from models.util import tensor_line_plots
import matplotlib.pyplot as plt
import os


class FlattenHead(nn.Module):
    def __init__(self, nf, seq_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(in_features=nf, out_features=seq_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class ALL4ONE(nn.Module):
    def __init__(self, config, device):
        super(ALL4ONE, self).__init__()
        self.seq_len = config.seq_len
        self.patch_size = config.patch_size
        self.stride = config.stride
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.d_llm = config.llm_dim
        self.output_dim = config.output_dim

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
            config=config, input_dims=1, output_dims=config.output_dim, device=device
        )
        self.ts2vec.to(dtype=torch.bfloat16, device=device)

        # visual module
        self.visual = self.model.visual
        self.image_processor = Qwen2VLImageProcessorFast()

        # ts2vec embedding align llm input dims
        # self.ts2vec_embedding = nn.Linear(config.seq_len, config.llm_dim).to(
        #     dtype=torch.bfloat16, device=device
        # )
        # self.ts2vec_embedding = TokenEmbedding(
        #     c_in=config.seq_len,
        #     d_model=config.llm_dim,
        # ).to(dtype=torch.bfloat16, device=device)
        self.ts2vec_embedding = PatchEmbedding(
            config.llm_dim, config.patch_size, config.stride, config.dropout
        ).to(dtype=torch.bfloat16, device=device)
        self.patch_nums = int((config.seq_len - self.patch_size) / self.stride + 2)

        # llm_output projection
        self.llm_output_projection = nn.Linear(self.patch_nums, config.output_dim).to(
            dtype=torch.bfloat16, device=device
        )
        # self.llm_output_projection = nn.Sequential(
        #     nn.Conv1d(
        #         in_channels=self.output_dim,
        #         out_channels=self.output_dim,
        #         kernel_size=3,
        #         padding=1,
        #     ),
        #     nn.GELU(),
        # ).to(dtype=torch.bfloat16, device=device)
        # self.llm_output_projection = FlattenHead(
        #     nf=self.patch_nums * self.d_ff,
        #     seq_window=config.seq_len,
        #     head_dropout=config.dropout,
        # ).to(dtype=torch.bfloat16, device=device)

        # outprojection
        if config.task == "forecast":
            self.output_projection = FlattenHead(
                nf=self.seq_len * self.output_dim,
                seq_window=config.pred_len,
                head_dropout=config.dropout,
            ).to(dtype=torch.bfloat16, device=device)

        self.normalize_layers = Normalize(config.enc_in, affine=False)

    def get_ts2vec_loss(self, x_enc):
        return self.ts2vec.fit(x_enc)

    def forward(self, x_enc, x_mask, y, y_mask, stage="train", batch_idx=0):
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
        x_enc_embed, n_vars = self.ts2vec_embedding(
            x_enc.permute(0, 2, 1)
        )  # [B, patch_nums, llm_dim]

        llm_enc_out = torch.cat(
            [images_embeds, x_enc_embed], dim=1
        )  # [B, token_num , llm_dim]
        dec_out = self.llm_model(inputs_embeds=llm_enc_out).last_hidden_state
        dec_out = dec_out[:, -self.patch_nums :, : self.seq_len]

        # simulate residual connections to optimize on ts2vec and enable LLM to learn increments
        dec_out = self.llm_output_projection(
            dec_out.permute(0, 2, 1)
        )  # [B, output_dim, seq_len]
        dec_out = dec_out + x_enc  # [B, seq_len, output_dim]

        # output
        dec_out = self.output_projection(dec_out).unsqueeze(-1)  # [B, pred_len, 1]
        dec_out = self.normalize_layers(dec_out, "denorm")

        return dec_out


class ALL4ONEFAST(nn.Module):
    def __init__(self, config, device):
        super(ALL4ONEFAST, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.patch_size = config.patch_size
        self.stride = config.stride
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.d_llm = config.llm_dim
        self.output_dim = config.output_dim

        # dataset
        self.description = config.content

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
            config=config, input_dims=1, output_dims=config.output_dim, device=device
        ).bfloat16()

        # visual module
        self.visual = self.model.visual
        self.image_processor = Qwen2VLImageProcessorFast()

        # processor
        self.tokernizer = Qwen2_5_VLProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        ).tokenizer

        # ts2vec embedding align llm input dims
        self.ts2vec_embedding = PatchEmbedding(
            config.llm_dim, config.patch_size, config.stride, config.dropout
        ).to(dtype=torch.bfloat16, device=device)
        self.patch_nums = int((config.seq_len - self.patch_size) / self.stride + 2)

        # outprojection
        if config.task == "forecast":
            self.output_projection = FlattenHead(
                nf=self.patch_nums * self.seq_len,
                seq_window=self.pred_len,
                head_dropout=config.dropout,
            ).to(dtype=torch.bfloat16, device=device)

        self.normalize_layers = Normalize(config.enc_in, affine=False)

        # cache setting
        self.cache_dir = f"./src/models/cache/{config.data}/"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    def get_ts2vec_loss(self, x_enc):
        return self.ts2vec.fit(x_enc)

    def forward(self, x_enc, x_mask, y, y_mask, stage="train", batch_idx=0):
        B, T, N = x_enc.shape
        x_enc = self.normalize_layers(x_enc, "norm")

        # statistics prompt
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            prompt_ = (
                f"<|im_start|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information and corresponding image; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                "<|vision_start|><|image_pad|><|vision_end|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = (
            self.tokernizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            .to(device=x_enc.device)
            .input_ids
        )
        prompt_embeddings = self.llm_model.get_input_embeddings()(
            prompt.to(x_enc.device)
        )  # (batch, prompt_token, dim)

        # check cache
        cache_file = os.path.join(self.cache_dir, f"{stage}_{batch_idx}.pt")

        if os.path.exists(cache_file):
            images_embeds = torch.load(
                cache_file, map_location=x_enc.device, weights_only=False
            )["data"]
        else:
            # create image tensor
            x_image = tensor_line_plots(x_enc)  # [B, 3, H, W]
            image_inputs = self.image_processor(
                images=x_image, videos=None, do_rescale=False
            ).to(device=x_enc.device, dtype=x_enc.dtype)
            images_embeds = self.visual(
                image_inputs["pixel_values"], grid_thw=image_inputs["image_grid_thw"]
            )
            images_embeds = images_embeds.reshape(B, -1, images_embeds.shape[-1])

            # save cache
            torch.save(
                {
                    "data": images_embeds.cpu(),
                    "dtype": "bfloat16",
                },
                cache_file,
                _use_new_zipfile_serialization=True,
            )

        # x_enc embedding
        x_enc_embed, n_vars = self.ts2vec_embedding(
            x_enc.permute(0, 2, 1)
        )  # [B, patch_nums, llm_dim]

        # prompt add image
        mask_image = prompt == self.model.config.image_token_id
        mask_image_unsqueezed = mask_image.unsqueeze(-1)
        mask_image_expanded = mask_image_unsqueezed.expand_as(prompt_embeddings)
        image_mask = mask_image_expanded.to(x_enc.device)

        prompt_embeddings = prompt_embeddings.masked_scatter(image_mask, images_embeds)

        # add <im_end> embed
        im_end_token_id = self.model.config.eos_token_id
        im_end_token_id_tensor = torch.tensor([im_end_token_id], device=x_enc.device)
        im_end_embed = self.llm_model.get_input_embeddings()(im_end_token_id_tensor)
        im_end_embed = im_end_embed.unsqueeze(0).expand(B, 1, -1).contiguous()

        prompt_embeddings = torch.cat(
            [prompt_embeddings, x_enc_embed, im_end_embed], dim=1
        )

        llm_enc_out = prompt_embeddings  # [B, token_num , llm_dim]
        dec_out = self.llm_model(inputs_embeds=llm_enc_out).last_hidden_state
        dec_out = dec_out[:, -self.patch_nums :, : self.seq_len]

        # output projection
        dec_out = self.output_projection(dec_out).unsqueeze(
            -1
        )  # [B, pred_len, output_dim]
        dec_out = self.normalize_layers(dec_out, "denorm")

        return dec_out


class ALL4ONEonlyTS2VEC(nn.Module):
    def __init__(self, config, device):
        super(ALL4ONEonlyTS2VEC, self).__init__()
        self.seq_len = config.seq_len
        self.output_dim = config.output_dim

        self.ts2vec = TS2Vec(
            config=config, input_dims=1, output_dims=config.output_dim, device=device
        ).bfloat16()

        # outprojection
        if config.task == "forecast":
            self.output_projection = FlattenHead(
                nf=self.seq_len * self.output_dim,
                seq_window=config.pred_len,
                head_dropout=config.dropout,
            ).to(dtype=torch.bfloat16, device=device)

        self.normalize_layers = Normalize(config.enc_in, affine=False)

    def get_ts2vec_loss(self, x_enc):
        return self.ts2vec.fit(x_enc)

    def forward(self, x_enc, x_mask, y, y_mask, stage="train", batch_idx=0):
        B, T, N = x_enc.shape

        x_enc = self.normalize_layers(x_enc, "norm")
        # time series embedding
        # x_enc = self.ts2vec.encode(
        #     x_enc,
        #     # encoding_window="full_series",
        #     # causal=True,
        #     # sliding_length=1,
        #     # sliding_padding=200
        # )
        # x_enc = self.ts2vec_embedding(x_enc.permute(0, 2, 1))

        dec_out = x_enc
        # output
        dec_out = self.output_projection(dec_out).unsqueeze(-1)  # [B, pred_len, 1]
        dec_out = self.normalize_layers(dec_out, "denorm")

        return dec_out
