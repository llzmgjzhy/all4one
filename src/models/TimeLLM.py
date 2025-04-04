import torch
import torch.nn as nn
from transformers import (
    LlamaConfig,
    LlamaModel,
    LlamaTokenizer,
    GPT2Config,
    GPT2Model,
    GPT2Tokenizer,
    BertConfig,
    BertModel,
    BertTokenizer,
)
from models.embed import PatchEmbedding
from math import sqrt
import torch.nn.functional as F


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, num_classes, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-3)
        self.linear = nn.Linear(nf, num_classes)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        # x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, config, data, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.patch_len = config["patch_size"]
        self.stride = config["stride"]
        self.seq_len = data.d_fea
        self.max_len = data.max_seq_len
        self.d_ff = config["d_ff"]  # dimension of fcn
        self.top_k = 5
        self.d_llm = config["llm_dim"]  # LLM model dimension
        self.num_classes = len(data.class_names)
        self.n_vars = data.n_phase

        if config["text_model"] == "LLAMA":
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained("huggyllama/llama-7b")
            self.llama_config.num_hidden_layers = config["llm_layers"]
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    "huggyllama/llama-7b",
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    "huggyllama/llama-7b",
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    "huggyllama/llama-7b",
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except (
                EnvironmentError
            ):  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    "huggyllama/llama-7b",
                    trust_remote_code=True,
                    local_files_only=False,
                )
        elif config["text_model"] == "GPT2":
            self.gpt2_config = GPT2Config.from_pretrained("openai-community/gpt2")

            self.gpt2_config.num_hidden_layers = config["llm_layers"]
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except (
                EnvironmentError
            ):  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    local_files_only=False,
                )
        elif config["text_model"] == "BERT":
            self.bert_config = BertConfig.from_pretrained(
                "google-bert/bert-base-uncased"
            )

            self.bert_config.num_hidden_layers = config["llm_layers"]
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    "google-bert/bert-base-uncased",
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    "google-bert/bert-base-uncased",
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    "google-bert/bert-base-uncased",
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except (
                EnvironmentError
            ):  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    "google-bert/bert-base-uncased",
                    trust_remote_code=True,
                    local_files_only=False,
                )
        else:
            raise Exception("LLM model is not defined")

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = "[PAD]"
            self.tokenizer.add_special_tokens({"pad_token": pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.use_prompt = True if config["prompt_domain"] else False
        if config["prompt_domain"]:
            self.description = "Medium voltage overhead power lines run for hundreds of miles to supply power to cities. These great distances make it expensive to manually inspect the lines for damage that doesn't immediately lead to a power outage, such as a tree branch hitting the line or a flaw in the insulator. These modes of damage lead to a phenomenon known as partial discharge — an electrical discharge which does not bridge the electrodes between an insulation system completely. Partial discharges slowly damage the power line, so left unrepaired they will eventually lead to a power outage or start a fire."

        self.dropout = nn.Dropout(config["dropout"])

        self.patch_embedding = PatchEmbedding(
            config["d_model"],
            self.patch_len,
            self.stride,
            config["dropout"],
            self.n_vars,
        )

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(
            config["d_model"], config["n_heads"], self.d_ff, self.d_llm
        )

        self.patch_nums = int((self.seq_len - self.patch_len) / self.stride + 1)
        self.head_nf = self.d_ff * self.patch_nums

        self.output_projection = FlattenHead(
            config["enc_in"],
            self.head_nf * self.n_vars,
            self.num_classes,
            head_dropout=config["dropout"],
        )

        # classification layers
        self.act = F.gelu
        self.ln_proj = nn.LayerNorm(self.head_nf * self.n_vars)

        self.out_layer = nn.Linear(self.head_nf * self.n_vars, self.num_classes)

    def forward(self, x_enc, x_mark_enc):

        B, N, T = x_enc.shape  # [b, n_vars*patch_nums, d_piece]
        x_enc = x_enc.contiguous().reshape(B * self.n_vars, -1, T)

        # prefix prompt setting
        if self.use_prompt:
            # min_values = torch.min(x_enc, dim=2)[0]
            # max_values = torch.max(x_enc, dim=2)[0]
            # medians = torch.median(x_enc, dim=2).values
            # lags = self.calcute_lags(x_enc)
            # trends = x_enc.diff(dim=2).sum(dim=2)

            prompt = []
            for b in range(x_enc.shape[0]):
                # min_values_str = str(min_values[b].tolist()[0])
                # max_values_str = str(max_values[b].tolist()[0])
                # median_values_str = str(medians[b].tolist()[0])
                # lags_values_str = str(lags[b].tolist())
                prompt_ = (
                    f"<|start_prompt|>Dataset description: {self.description}"
                    f"Task description: classification the input data given the {str(self.seq_len)} steps information; "
                    # "Input statistics: "
                    # f"min value {min_values_str}, "
                    # f"max value {max_values_str}, "
                    # f"median value {median_values_str}, "
                    # f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                    # f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
                )

                prompt.append(prompt_)

            prompt = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).input_ids
            prompt_embeddings = self.llm_model.get_input_embeddings()(
                prompt.to(x_enc.device)
            )  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(
            self.word_embeddings.permute(1, 0)
        ).permute(1, 0)

        x_enc = x_enc.reshape(B, N, T).contiguous()
        enc_out, n_vars = self.patch_embedding(
            x_enc
        )  # return (b, 480, 5000) -> (b * n_vars, patch_num, d_model)
        enc_out = self.reprogramming_layer(
            enc_out, source_embeddings, source_embeddings
        )  # return (b * 480, patch_num, d_ff)
        # enc_out = enc_out.reshape(
        #     B, -1, enc_out.shape[-1]
        # )  # (b, n_vars*patch_nums, d_ff)
        llama_enc_out = (
            torch.cat([prompt_embeddings, enc_out], dim=1)
            if self.use_prompt
            else enc_out
        )
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, : self.d_ff]

        # dec_out = torch.reshape(dec_out, (B, -1, dec_out.shape[-2], dec_out.shape[-1]))

        # TODO: check if the permute is necessary
        # dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        # dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums :])
        dec_out = self.act(dec_out).reshape(B, -1)
        dec_out = self.ln_proj(dec_out)
        dec_out = self.out_layer(dec_out)

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


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
