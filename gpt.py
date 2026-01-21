import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # 50257
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 12 // 4
    n_emb: int = 768
    logit_softcap: float = 15.0

    use_rope: bool = True
    use_rmsnorm: bool = True
    use_qk_norm: bool = True
    use_gqa: bool = True
    use_kv_cache: bool = True


@dataclass
class GPTConfigD20:
    block_size: int = 2048
    vocab_size: int = 50304
    n_layer: int = 20
    n_head: int = 10
    n_kv_head: int = 10
    n_emb: int = 1280
    logit_softcap: float = 15.0

    use_rope: bool = True
    use_rmsnorm: bool = True
    use_qk_norm: bool = True
    use_gqa: bool = True
    use_kv_cache: bool = True


def rope_cache(seq_len, head_dim, base: int = 10000):
    assert head_dim % 2 == 0

    # number of 2d dim pairs
    half_dim = head_dim // 2

    freq_seq = torch.arange(half_dim, dtype=torch.float32)
    freq_seq = base ** (-freq_seq / half_dim)

    t = torch.arange(seq_len, dtype=torch.float32)

    # outer product to get the angles
    freqs = torch.outer(t, freq_seq)  # (seq_len, half_dim)

    # compute the cos and sin matrices
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, half_dim)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, half_dim)

    return cos, sin


def rope_apply(x, cos, sin, offset=0):
    # x: (B, nH, T, Hs)
    B, H, T, Hs = x.shape
    half = Hs // 2

    x1 = x[..., :half]
    x2 = x[..., half:]

    # cast to float 32
    x1 = x1.float()
    x2 = x2.float()

    # cos, sin are (1, 1, seq_len, half_dim), slice to match T
    cos = cos[:, :, offset : offset + T, :]  # (1, 1, T, half_dim)
    sin = sin[:, :, offset : offset + T, :]  # (1, 1, T, half_dim)

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    out = torch.cat([rotated_x1, rotated_x2], dim=-1)
    return out.type_as(x)


def norm(x, use_rms=True):
    # Purely functional rmsnorm with no learnable params
    if use_rms:
        return F.rms_norm(x, (x.size(-1),))
    else:
        return F.layer_norm(x, (x.size(-1),))


class KVCache:
    def __init__(self, n_layers):
        self.n_layers = n_layers
        self.k = [None] * n_layers
        self.v = [None] * n_layers

    def seq_len(self):
        for kk in self.k:
            if kk is not None:
                return kk.size(-2)
        return 0

    def get(self, layer_idx):
        return self.k[layer_idx], self.v[layer_idx]

    def append(self, layer_idx, k, v):
        if self.k[layer_idx] is None:
            self.k[layer_idx] = k
            self.v[layer_idx] = v
        else:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], k], dim=-2)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], v], dim=-2)


class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.c_fc = nn.Linear(config.n_emb, 4 * config.n_emb)
        self.c_proj = nn.Linear(4 * config.n_emb, config.n_emb)
        self.c_proj.residual_proj = True

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_emb = config.n_emb
        self.n_kv_head = config.n_kv_head

        self.head_dim = config.n_emb // config.n_head

        if not self.config.use_gqa:
            self.c_attn = nn.Linear(config.n_emb, 3 * config.n_emb)
        else:
            self.c_q = nn.Linear(
                config.n_emb, config.n_head * self.head_dim, bias=False
            )
            self.c_k = nn.Linear(
                config.n_emb, config.n_kv_head * self.head_dim, bias=False
            )
            self.c_v = nn.Linear(
                config.n_emb, config.n_kv_head * self.head_dim, bias=False
            )

        self.c_proj = nn.Linear(config.n_emb, config.n_emb)
        self.c_proj.residual_proj = True

    def forward(self, x, cos_sin, kv_cache: KVCache | None = None, layer_idx=0):
        B, T, C = x.shape
        past_len = kv_cache.seq_len() if kv_cache is not None else 0

        # project Q, K, V
        if not self.config.use_gqa:
            qkv = self.c_attn(x)  # B, T, 3*C
            q, k, v = qkv.split(self.n_emb, dim=2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # B, nH, T, Hs
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # B, nH, T, Hs
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # B, nH, T, Hs
        else:
            q = (
                self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            )  # B, nH, T, Hs
            k = (
                self.c_k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
            )  # B, n_kvH, T, Hs
            v = (
                self.c_v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
            )  # B, n_kvH, T, Hs

        # qk norm
        if self.config.use_qk_norm:
            q = norm(q, use_rms=self.config.use_rmsnorm)
            k = norm(k, use_rms=self.config.use_rmsnorm)

        # rope
        if self.config.use_rope:
            cos, sin = cos_sin
            q = rope_apply(q, cos, sin, offset=past_len)
            k = rope_apply(k, cos, sin, offset=past_len)

        k_full, v_full = k, v
        if kv_cache is not None:
            k_past, v_past = kv_cache.get(layer_idx)
            kv_cache.append(layer_idx, k, v)

            if k_past is not None:
                k_full = torch.cat([k_past, k], dim=-2)
                v_full = torch.cat([v_past, v], dim=-2)

        y = F.scaled_dot_product_attention(
            q, k_full, v_full, is_causal=(T > 1), enable_gqa=self.config.use_gqa
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache=None, layer_idx=0):
        x = x + self.attn(
            norm(x, use_rms=self.config.use_rmsnorm),
            cos_sin,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
        )
        x = x + self.mlp(norm(x, use_rms=self.config.use_rmsnorm))
        return x


class GPTModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # precompute rope cache
        if config.use_rope:
            cos, sin = rope_cache(config.block_size, config.n_emb // config.n_head)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        else:
            self.cos = self.sin = None

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_emb),
                wpe=(
                    nn.Embedding(config.block_size, config.n_emb)
                    if not config.use_rope
                    else None
                ),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            )
        )

        self.lm_head = nn.Linear(config.n_emb, config.vocab_size, bias=False)

        # using 0.9 to not drift to extreme values. other option is clamping
        self.resid_lambdas = nn.Parameter(torch.full((config.n_layer,), 0.9))

        # not tying weights to use different optimizers
        # self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02

            # smaller init for lm_head
            if module is self.lm_head:
                std = 0.001

            # normal init for GPT2
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # GPT-2 has special residual projections scaled by 1 / sqrt(2*n_layer)
        if (
            isinstance(module, nn.Linear)
            and hasattr(module, "residual_proj")
            and module is not self.lm_head
        ):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
            )

    def forward(self, idx, targets=None, kv_cache=None):
        B, T = idx.shape
        cos_sin = self.cos, self.sin

        x = self.transformer.wte(idx)  # B, T, C (n_emb)
        if self.config.use_rope is False:
            positions = torch.arange(0, T, dtype=torch.long, device=idx.device)
            x = x + self.transformer.wpe(positions)  # B, T, C

        x = norm(x, use_rms=self.config.use_rmsnorm)

        for layer_idx, block in enumerate(self.transformer.h):  # B, T, C
            x = (
                x * self.resid_lambdas[layer_idx]
            )  # using resid_lambda to scale residuals (doing it here to apply to entire layer including attention)
            x = block(x, cos_sin, kv_cache=kv_cache, layer_idx=layer_idx)

        x = norm(x, use_rms=self.config.use_rmsnorm)
        logits = self.lm_head(x)  # B, T, C (vocab_size)
        # apply logit softcap, this is helpful with untied weights and split optims
        logits = self.config.logit_softcap * torch.tanh(
            logits / self.config.logit_softcap
        )

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        idx,
        max_tokens,
        temperature=1.0,
        top_k=50,
        print_as_you_go=False,
        use_kv_cache=True,
    ):
        B, T = idx.shape

        # enforce block_size limit for ROPE
        if T > self.config.block_size:
            idx = idx[:, -self.config.block_size :]
            T = idx.size(1)

        kv_cache = KVCache(self.config.n_layer) if use_kv_cache else None

        # prefill cache by running tokens through once
        logits, _ = self(idx, kv_cache=kv_cache)  # B, T, C (vocab_size)
        next_logits = logits[:, -1, :]  # B, C (vocab_size)

        for _ in range(max_tokens):
            probs = F.softmax(next_logits / temperature, dim=-1)

            if top_k is not None:
                topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
                ix = torch.multinomial(topk_probs, 1)
                xcol = torch.gather(topk_indices, -1, ix)
            else:
                xcol = torch.multinomial(probs, 1)

            # append new token to sequence
            idx = torch.cat((idx, xcol), dim=1)

            if print_as_you_go:
                token_id = xcol.item()
                token_str = self.tokenizer.decode([token_id])
                print(token_str, end="", flush=True)

            if use_kv_cache:
                # ROPE block size safety (we have pre-computed fixed angles)
                if kv_cache.seq_len() >= self.config.block_size:
                    raise RuntimeError("Exceeded RoPE window during generation")

                # feed only the new token
                logits, _ = self(xcol, kv_cache=kv_cache)
                next_logits = logits[:, -1, :]

            else:
                if idx.size(1) > self.config.block_size:
                    idx = idx[:, -self.config.block_size :]

                logits, _ = self(idx)
                next_logits = logits[:, -1, :]

        return idx


def get_param_groups(model):
    embed_params = []
    lm_head_params = []
    matrix_params = []
    scalar_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if name == "resid_lambdas":
            scalar_params.append(p)
        elif "transformer.wte" in name:
            embed_params.append(p)
        elif "lm_head" in name:
            lm_head_params.append(p)
        elif p.dim() >= 2:
            matrix_params.append(p)
        else:
            scalar_params.append(p)

    return embed_params, lm_head_params, matrix_params, scalar_params


def configure_optimizer(model):
    embed, lm_head, matrix, scalar = get_param_groups(model)

    optim_groups = [
        # embeddings: highest LR, no decay
        {"params": embed, "lr": 2.0e-3, "weight_decay": 0.0, "name": "embed"},
        # lm head: slightly lower
        {"params": lm_head, "lr": 8.0e-4, "weight_decay": 0.0, "name": "lm_head"},
        # transformer matrices: main bulk
        {"params": matrix, "lr": 1.5e-3, "weight_decay": 0.05, "name": "matrix"},
        # norms / scalars
        {"params": scalar, "lr": 2.0e-4, "weight_decay": 0.0, "name": "scalar"},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=torch.cuda.is_available(),
    )

    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    return optimizer


# commenting in favor of split optimizers
# def configure_optimizer(model, lr):
#   decay, no_decay = [], []
#   for name, p in model.named_parameters():
#     if not p.requires_grad:
#       continue
#     if p.dim() >= 2:
#       decay.append(p)
#     else:
#       no_decay.append(p)

#   optim_params = [
#       { "params": decay, "weight_decay": 0.1 },
#       { "params": no_decay, "weight_decay": 0.0 }
#   ]

#   try:
#     optimizer = torch.optim.AdamW(optim_params, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
#   except TypeError:
#     optimizer = torch.optim.AdamW(optim_params, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=False)

#   return optimizer
