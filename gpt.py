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
  n_kv_head: int = 12 // 4
  n_emb: int = 768

  # ablation toggles
  use_rope: bool = True
  use_rmsnorm: bool = True
  use_qk_norm: bool = False
  use_gqa: bool = False


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


def rope_apply(x, cos, sin):
  # x: (B, nH, T, Hs)
  B, H, T, Hs = x.shape
  half = Hs // 2

  x1 = x[..., :half]
  x2 = x[..., half:]

  # cos, sin are (1, 1, seq_len, half_dim), slice to match T
  cos = cos[:, :, :T, :]  # (1, 1, T, half_dim)
  sin = sin[:, :, :T, :]  # (1, 1, T, half_dim)

  rotated_x1 = x1 * cos - x2 * sin
  rotated_x2 = x1 * sin + x2 * cos

  return torch.cat([rotated_x1, rotated_x2], dim=-1)
  

def norm(x, use_rms=True):
    # Purely functional rmsnorm with no learnable params
    if use_rms:
      return F.rms_norm(x, (x.size(-1),))
    else:
      return F.layer_norm(x, (x.size(-1),))


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
      self.c_q = nn.Linear(config.n_emb, config.n_head * self.head_dim, bias=False)
      self.c_k = nn.Linear(config.n_emb, config.n_kv_head * self.head_dim, bias=False)
      self.c_v = nn.Linear(config.n_emb, config.n_kv_head * self.head_dim, bias=False)
    
    self.c_proj = nn.Linear(config.n_emb, config.n_emb)
    self.c_proj.residual_proj = True
    
  def forward(self, x, cos_sin):
    B, T, C = x.shape

    # project Q, K, V
    if not self.config.use_gqa:
      qkv = self.c_attn(x) # B, T, 3*C
      q, k, v = qkv.split(self.n_emb, dim=2)
      q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # B, nH, T, Hs
      k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
      v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
    else:
      q = self.c_q(x).view(B, T, self.n_head, self.head_dim) # B, nH, Hs, T
      k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
      v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

    # qk norm
    if self.config.use_qk_norm:
      q = norm(q, use_rms=self.config.use_rmsnorm)
      k = norm(k, use_rms=self.config.use_rmsnorm)

    q = q.transpose(1, 2) # B, nH, T, Hs
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # rope 
    if self.config.use_rope:
      cos, sin = cos_sin
      q = rope_apply(q, cos, sin)
      k = rope_apply(k, cos, sin)

    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.config.use_gqa)
    y = y.transpose(1, 2).contiguous().view(B, T, C)

    y = self.c_proj(y)
    return y


class Block(nn.Module):
  def __init__(self, config) -> None:
    super().__init__()
    self.config = config

    self.attn = CausalSelfAttention(config)
    self.mlp = MLP(config)

  def forward(self, x, cos_sin):
    x = x + self.attn(norm(x, use_rms=self.config.use_rmsnorm), cos_sin)
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
      self.register_buffer('cos', cos, persistent=False)
      self.register_buffer('sin', sin, persistent=False)
    else:
      self.cos = self.sin = None

    self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size, config.n_emb),
        h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
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
    cos_sin = self.cos, self.sin

    x = self.transformer.wte(idx) # B, T, C (n_emb)
    x = norm(x, use_rms=self.config.use_rmsnorm)
    for block in self.transformer.h: # B, T, C
      x = block(x, cos_sin)
    x = norm(x, use_rms=self.config.use_rmsnorm)
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