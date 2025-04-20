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


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


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
    def __init__(self, llm_dim, num_heads, d_ff, output_dim, dropout=0.0):
        super().__init__()
        self.llm_dim = llm_dim
        self.num_heads = num_heads
        self.output_dim = output_dim

        self.base_proj = nn.Sequential(
            nn.Linear(output_dim, d_ff),
            nn.GELU(),
            Qwen2RMSNorm(d_ff),
            nn.Linear(d_ff, llm_dim),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=self.llm_dim,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=dropout,
        )

        self.mlp = SwiGLUMLP(
            in_features=self.llm_dim,
            hidden_features=self.llm_dim // 4,
        )
        self.input_norm = Qwen2RMSNorm(self.llm_dim, eps=1e-06)
        self.output_norm = Qwen2RMSNorm(self.llm_dim, eps=1e-06)

        self.output_mlp = Mlp(
            in_features=self.llm_dim,
            hidden_features=self.llm_dim // 4,
            out_features=self.output_dim,
            act_layer=nn.GELU,
            drop=dropout,
        )

    def forward(self, x, y_base):
        B, T, N = x.shape
        query = self.base_proj(y_base)  # [B, pred_len, llm_dim]
        query_norm = self.input_norm(query)
        attn_output, _ = self.attention(query_norm, x, x)
        x = attn_output

        x_delta = x + self.mlp(self.output_norm(x))
        # x_delta = self.output_norm(x_delta)
        return self.output_mlp(x_delta)  # [B, pred_len, output_dim]


class FusionReprogrammingLayer(nn.Module):
    def __init__(
        self,
        llm_dim,
        num_heads,
        d_model,
        d_ff,
        output_dim,
        hidden_dim=None,
        dropout=0.0,
    ):
        super().__init__()
        self.llm_dim = llm_dim
        self.num_heads = num_heads
        self.output_dim = output_dim

        self.attention = ReprogrammingLayer(
            d_model=d_model,
            n_heads=num_heads,
            d_keys=d_ff,
            d_llm=llm_dim,
            output_dim=output_dim,
            attention_dropout=dropout,
        )

        self.fc = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, y_base, base):
        B, T, N = x.shape

        increment = self.attention(y_base, x, x)  # [B, pred_len, output_dim]

        concat_fea = torch.cat(
            [increment, base], dim=-1
        )  # [B, pred_len, output_dim * 2]
        gate = self.fc(concat_fea)  # [B, pred_len, output_dim]
        fused = increment * gate + (1 - gate) * base  # [B, pred_len, output_dim]
        fused = self.norm(fused)
        return fused  # [B, pred_len, output_dim]


class CrossAttention(nn.Module):
    def __init__(self, llm_dim, num_heads, dropout=0.0):
        super(CrossAttention, self).__init__()
        self.llm_dim = llm_dim
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(
            embed_dim=self.llm_dim, num_heads=self.num_heads, dropout=dropout
        )
        self.input_norm = Qwen2RMSNorm(self.llm_dim, eps=1e-06)
        self.output_norm = Qwen2RMSNorm(self.llm_dim, eps=1e-06)
        self.mlp = SwiGLUMLP(
            in_features=self.llm_dim,
            hidden_features=self.llm_dim // 4,
        )

    def forward(self, x, content):
        B, T, N = x.shape
        x = self.input_norm(x)
        attn_output, _ = self.attention(query=x, key=content, value=content)

        x = attn_output + self.mlp(self.output_norm(attn_output))
        return x


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
        self.top_k = 5

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
            config.d_model, config.patch_size, config.stride, config.dropout
        ).to(dtype=torch.bfloat16, device=device)
        self.ts_reprogramming = ReprogrammingLayer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_keys=config.d_ff,
            d_llm=self.d_llm,
            attention_dropout=config.dropout,
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
            # self.output_projection = AdaptiveFeatureAggregation(
            #     llm_dim=config.llm_dim,
            #     num_heads=config.n_heads,
            #     d_ff=config.d_ff,
            #     output_dim=config.output_dim,
            #     dropout=config.dropout,
            # ).to(dtype=torch.bfloat16, device=device)
            self.y_base_embed = TokenEmbedding(
                c_in=config.output_dim,
                d_model=config.d_model,
            ).to(dtype=torch.bfloat16, device=device)
            self.output_projection = FusionReprogrammingLayer(
                llm_dim=config.llm_dim,
                num_heads=config.n_heads,
                d_model=config.d_model,
                d_ff=config.d_ff,
                output_dim=config.output_dim,
                hidden_dim=config.output_dim * 4,
                dropout=config.dropout,
            ).to(dtype=torch.bfloat16, device=device)

        self.normalize_layers = Normalize(config.enc_in, affine=False)

        # cache setting
        self.cache_dir = f"./src/models/cache/{config.data}/"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    def forward(self, x_enc, x_mask, y, y_mask, stage="train", batch_idx=0):
        B, T, N = x_enc.shape
        x_enc = self.normalize_layers(x_enc, "norm")

        x_enc_residual = self.x_enc_residual_embed(x_enc).unsqueeze(
            -1
        )  # [B, pred_len, N]

        # x_enc statistic
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc.float())
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                "<|im_start|>system\n"
                "You are a multimodal time-series analysis expert. Your task is to integrate temporal patterns, and visual features to make precise predictions.\n<|im_end|>\n"
                f"<|im_start|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information and corresponding image; "
                "Statistics input: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}\n"
                "Image input:"
                "<|vision_start|><|image_pad|><|vision_end|>"
                "Time series input:"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

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
            x_enc.permute(0, 2, 1).contiguous()
        )  # [B, patch_nums, llm_dim]
        x_enc_embed = self.ts_reprogramming(x_enc_embed, images_embeds, images_embeds)

        # prompt add image
        mask_image = prompt == self.model.config.image_token_id
        mask_image_unsqueezed = mask_image.unsqueeze(-1)
        mask_image_expanded = mask_image_unsqueezed.expand_as(prompt_embeddings)
        image_mask = mask_image_expanded.to(x_enc.device)

        prompt_embeddings = prompt_embeddings.masked_scatter(image_mask, images_embeds)

        # add <im_end> and assitant embed
        im_end_prompt = "<|im_end|>\n<|im_start|>assistant\n"
        im_end_prompt = (
            self.tokenizer(
                im_end_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            .to(device=x_enc.device)
            .input_ids
        )
        im_end_embed = self.llm_model.get_input_embeddings()(im_end_prompt)
        im_end_embed = im_end_embed.expand(B, -1, -1).contiguous()

        llm_enc_out = torch.cat(
            [prompt_embeddings, x_enc_embed, im_end_embed], dim=1
        )  # [B, token_num , llm_dim]
        dec_out = self.llm_model(inputs_embeds=llm_enc_out).last_hidden_state

        y_base = self.y_base_embed(x_enc_residual)
        dec_out = self.output_projection(
            dec_out, y_base, x_enc_residual
        )  # [B,  pred_len, output_dim]
        # dec_out = self.output_projection(
        #     dec_out, x_enc_residual
        # )  # [B,  pred_len, output_dim]
        # residual connection
        # dec_out = dec_out + x_enc_residual
        dec_out = self.normalize_layers(dec_out, "denorm")

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


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


class ReprogrammingLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_keys=None,
        d_llm=None,
        output_dim=None,
        attention_dropout=0.1,
    ):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        output_dim = output_dim if output_dim else d_llm
        self.out_projection = nn.Linear(d_keys * n_heads, output_dim)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        _, S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(B, S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(B, S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,bshe->blhe", A, value_embedding)

        return reprogramming_embedding
