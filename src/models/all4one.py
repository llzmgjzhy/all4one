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


class SwiGLUMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
    ):
        super().__init__()
        self.hidden_size = in_features
        self.intermediate_size = hidden_features
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AdaptiveFeatureAggregation(nn.Module):
    def __init__(self, llm_dim, num_heads, pred_len, output_dim, dropout=0.0):
        super().__init__()
        self.llm_dim = llm_dim
        self.num_heads = num_heads
        self.output_dim = output_dim

        self.attention = nn.MultiheadAttention(
            embed_dim=self.llm_dim, num_heads=self.num_heads, batch_first=True
        )

        self.query = nn.Parameter(torch.randn(1, pred_len, self.llm_dim))

        self.linear = nn.Linear(self.llm_dim, self.output_dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, N = x.shape
        query = self.query.expand(B, -1, -1)
        attn_output, _ = self.attention(query, x, x)
        output = self.linear(attn_output)
        return self.output_dropout(output)


class CrossAttention(nn.Module):
    def __init__(self, llm_dim, num_heads, dropout=0.0):
        super(CrossAttention, self).__init__()
        self.llm_dim = llm_dim
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(
            embed_dim=self.llm_dim, num_heads=self.num_heads
        )

    def forward(self, x, content):
        B, T, N = x.shape
        attn_output, _ = self.attention(query=x, key=content, value=content)
        return attn_output


class TemporalAwareAggregation(nn.Module):
    def __init__(self, llm_dim, num_heads, output_dim, pred_len, dropout=0.0):
        super().__init__()
        self.temporal_pe = nn.Parameter(
            torch.randn(1, pred_len, llm_dim)
        )  # temporal position embedding
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=llm_dim, num_heads=num_heads, batch_first=True
        )
        self.gated_fusion = nn.Sequential(nn.Linear(2 * llm_dim, llm_dim), nn.Sigmoid())
        self.output_proj = nn.Linear(llm_dim, output_dim)

        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape

        # temporal aware embedding
        queries = self.temporal_pe.expand(B, -1, -1)  # [B, seq_len, D]

        # cross attention aggregation
        attn_out, _ = self.cross_attn(query=queries, key=x, value=x)

        # gated residual connection
        fused = (
            self.gated_fusion(torch.cat([attn_out, queries], dim=-1)) * attn_out
            + queries
        )

        return self.output_proj(self.output_dropout(fused))


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
        # values determined through experiments
        self.image_token_nums = 81

        # dataset
        self.description = config.content

        # model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        ).to(device=device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.llm_model = self.model.model

        # visual module
        self.visual = self.model.visual
        self.image_processor = Qwen2VLImageProcessorFast()
        self.merge_length = self.image_processor.merge_size**2

        # processor
        self.tokenizer = Qwen2_5_VLProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            use_fast=True,
        ).tokenizer

        self.image_token = (
            "<|image_pad|>"
            if not hasattr(self.tokenizer, "image_token")
            else self.tokenizer.image_token
        )

        # ts2vec embedding align llm input dims
        self.ts2vec_embedding = PatchEmbedding(
            config.llm_dim, config.patch_size, config.stride, config.dropout
        ).to(dtype=torch.bfloat16, device=device)
        self.ts_reprogramming = CrossAttention(
            llm_dim=config.llm_dim,
            num_heads=config.n_heads,
            dropout=config.dropout,
        ).to(dtype=torch.bfloat16, device=device)
        self.patch_nums = int((config.seq_len - self.patch_size) / self.stride + 2)

        # residual embedding, token embed on x_enc, and llm output will be added, speed up training
        self.x_enc_residual_embed = FlattenHead(
            nf=self.seq_len * self.output_dim,
            seq_window=config.pred_len,
            head_dropout=config.dropout,
        ).to(dtype=torch.bfloat16, device=device)
        x_enc_residual_embed_params = {
            k.replace("output_projection.linear", "linear"): v
            for k, v in torch.load(config.residual_path, map_location=device)[
                "state_dict"
            ].items()
        }
        self.x_enc_residual_embed.load_state_dict(x_enc_residual_embed_params)
        for param in self.x_enc_residual_embed.parameters():
            param.requires_grad = False

        # outprojection
        if config.task == "forecast":
            self.output_projection = AdaptiveFeatureAggregation(
                llm_dim=config.llm_dim,
                num_heads=config.n_heads,
                pred_len=config.pred_len,
                output_dim=config.output_dim,
                dropout=config.dropout,
            ).to(dtype=torch.bfloat16, device=device)
            # self.output_projection = TemporalAwareAggregation(
            #     llm_dim=config.llm_dim,
            #     num_heads=config.n_heads,
            #     output_dim=config.output_dim,
            #     pred_len=self.pred_len,
            #     dropout=config.dropout,
            # ).to(dtype=torch.bfloat16, device=device)

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

        x_enc_residual = self.x_enc_residual_embed(x_enc).unsqueeze(
            -1
        )  # [B, pred_len, N]

        prompt_template = (
            "<|im_start|>system\n"
            "You are a multimodal time-series analysis expert. Your task is to integrate temporal patterns, and visual features to make precise predictions.\n<|im_end|>\n"
            f"<|im_start|>Dataset description: {self.description}"
            f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information and corresponding image; "
            "Input image:"
            "<|vision_start|><|image_pad|><|vision_end|>"
            "Input time series:"
        )
        prompt = [prompt_template for _ in range(x_enc.shape[0])]

        # check cache
        cache_file = os.path.join(self.cache_dir, f"{stage}_{batch_idx}.pt")

        if os.path.exists(cache_file):
            images_embeds = torch.load(
                cache_file, map_location=x_enc.device, weights_only=False
            )["data"]
            image_grid_thw = torch.load(
                cache_file, map_location=x_enc.device, weights_only=False
            )["image_grid_thw"]
            image_token_nums = torch.load(
                cache_file, map_location=x_enc.device, weights_only=False
            )["image_token_nums"]
        else:
            # create image tensor
            x_image = tensor_line_plots(x_enc, flip=True)  # [B, 3, H, W]
            image_inputs = self.image_processor(
                images=x_image, videos=None, do_rescale=False
            ).to(device=x_enc.device, dtype=x_enc.dtype)
            images_embeds = self.visual(
                image_inputs["pixel_values"], grid_thw=image_inputs["image_grid_thw"]
            )
            image_grid_thw = image_inputs["image_grid_thw"]
            # recovery batch dim depend on image_grid_thw
            image_token_nums = [
                thw.prod().item() // self.merge_length for thw in image_grid_thw
            ]
            split_embeds = torch.split(images_embeds, image_token_nums, dim=0)
            images_embeds = torch.nn.utils.rnn.pad_sequence(
                split_embeds, batch_first=True
            )

            # images_embeds = images_embeds.reshape(B, -1, images_embeds.shape[-1])

            # save cache
            torch.save(
                {
                    "data": images_embeds.cpu(),
                    "image_grid_thw": image_inputs["image_grid_thw"].cpu(),
                    "image_token_nums": image_token_nums,
                    "dtype": "bfloat16",
                },
                cache_file,
                _use_new_zipfile_serialization=True,
            )

        # expand image token depend on the actual image_grid_thw
        if image_grid_thw is not None:
            for i in range(len(prompt)):
                while self.image_token in prompt[i]:
                    prompt[i] = prompt[i].replace(
                        self.image_token,
                        "<|placeholder|>" * (image_token_nums[i]),
                        1,
                    )
                prompt[i] = prompt[i].replace("<|placeholder|>", self.image_token)

        prompt = (
            self.tokenizer(
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

        # x_enc embedding
        x_enc_embed, n_vars = self.ts2vec_embedding(
            x_enc.permute(0, 2, 1)
        )  # [B, patch_nums, llm_dim]
        x_enc_embed = self.ts_reprogramming(
            x_enc_embed.transpose(0, 1), images_embeds.transpose(0, 1)
        ).transpose(0, 1)

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

        llm_enc_out = torch.cat(
            [prompt_embeddings, x_enc_embed, im_end_embed], dim=1
        )  # [B, token_num , llm_dim]
        dec_out = self.llm_model(inputs_embeds=llm_enc_out).last_hidden_state

        dec_out = self.output_projection(dec_out)  # [B,  pred_len, output_dim]
        # residual connection
        dec_out = dec_out + x_enc_residual
        dec_out = self.normalize_layers(dec_out, "denorm")

        return dec_out


class ALL4ONEonlyTS2VEC(nn.Module):
    def __init__(self, config, device):
        super(ALL4ONEonlyTS2VEC, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.output_dim = config.output_dim

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

        dec_out = x_enc
        # output
        dec_out = self.output_projection(dec_out).unsqueeze(-1)  # [B, pred_len, 1]
        # dec_out = self.mlp_projection(dec_out.permute(0, 2, 1)).transpose(
        #     1, 2
        # )  # [B, pred_len, 1]

        dec_out = self.normalize_layers(dec_out, "denorm")

        return dec_out
