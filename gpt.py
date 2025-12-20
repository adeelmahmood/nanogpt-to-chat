import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

from dataclasses import dataclass


@dataclass
class GPTConfig:
  block_size: int = 1024
  vocab_size: int = 50257 # 50304
  n_layer: int = 12
  n_head: int = 12
  n_emb: int = 768


class MLP(nn.Module):
  def __init__(self, config) -> None:
    super().__init__()
    self.config = config

    self.c_fc = nn.Linear(config.n_emb, 4 * config.n_emb)
    self.gelu = nn.GELU(approximate='tanh')
    self.c_proj = nn.Linear(4 * config.n_emb, config.n_emb)
    self.c_proj.residual_proj = True

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return x


class CausalSelfAttention(nn.Module):
  def __init__(self, config) -> None:
    super().__init__()
    self.config = config
    self.n_head = config.n_head
    self.n_emb = config.n_emb

    self.c_attn = nn.Linear(config.n_emb, 3 * config.n_emb)
    self.c_proj = nn.Linear(config.n_emb, config.n_emb)
    self.c_proj.residual_proj = True
    # self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

  def forward(self, x):
    B, T, C = x.shape

    q, k, v = self.c_attn(x).split(self.n_emb, dim=2)
    q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # B, nH, T, Hs
    k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # B, nH, T, Hs
    v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # B, nH, T, Hs

    # scores = (q @ k.transpose(-2, -1)) / k.size(-1)**0.5 # B, nH, T, T
    # scores = scores.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
    # weights = F.softmax(scores, dim=-1)
    # context = weights @ v # B, nH, T, Hs
    context = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    context = context.transpose(1, 2).contiguous().view(B, T, C)

    context = self.c_proj(context)
    return context


class Block(nn.Module):
  def __init__(self, config) -> None:
    super().__init__()
    self.config = config

    self.ln_1 = nn.LayerNorm(config.n_emb)
    self.attn = CausalSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_emb)
    self.mlp = MLP(config)

  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x


class GPTModel(nn.Module):
  def __init__(self, config) -> None:
    super().__init__()
    self.config = config
    self.tokenizer = tiktoken.get_encoding("gpt2")

    self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size, config.n_emb),
        wpe = nn.Embedding(config.block_size, config.n_emb),
        h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
        ln_f = nn.LayerNorm(config.n_emb),
    ))

    self.lm_head = nn.Linear(config.n_emb, config.vocab_size, bias=False)

    self.lm_head.weight = self.transformer.wte.weight

    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        # normal init for GPT2
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # GPT-2 has special residual projections scaled by 1 / sqrt(2*n_layer)
    if isinstance(module, nn.Linear) and hasattr(module, "residual_proj"):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

  def forward(self, idx, targets=None):
    B, T = idx.shape

    tok_emb = self.transformer.wte(idx) # B, T, C (n_emb)
    pos_emb = self.transformer.wpe(torch.arange(0, T, dtype=torch.long, device=idx.device)) # T, C
    x = tok_emb + pos_emb # B, T, C
    for block in self.transformer.h: # B, T, C
      x = block(x)
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x) # B, T, C (vocab_size)

    loss = None
    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    return logits, loss

  @torch.no_grad()
  def generate(self, idx, max_tokens, temperature=1.0, top_k=50, print_as_you_go=False):
    for _ in range(max_tokens):
      idx_trim = idx if idx.size(-1) < self.config.block_size else idx[:, -self.config.block_size:]
      logits,_ = self(idx_trim)
      logits = logits[:, -1, :]
      probs = F.softmax(logits / temperature, dim=-1)
      topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
      ix = torch.multinomial(topk_probs, 1)
      xcol = torch.gather(topk_indices, -1, ix)
      idx = torch.cat((idx, xcol), dim=1)

      if print_as_you_go:
        token_id = xcol.item()
        token_str = self.tokenizer.decode([token_id])
        print(token_str, end="", flush=True)
    return idx


def configure_optimizer(model, lr):
  decay, no_decay = [], []
  for name, p in model.named_parameters():
    if not p.requires_grad:
      continue
    if p.dim() >= 2:
      decay.append(p)
    else:
      no_decay.append(p)

  optim_params = [
      { "params": decay, "weight_decay": 0.1 },
      { "params": no_decay, "weight_decay": 0.0 }
  ]

  try:
    optimizer = torch.optim.AdamW(optim_params, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
  except TypeError:
    optimizer = torch.optim.AdamW(optim_params, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=False)

  return optimizer