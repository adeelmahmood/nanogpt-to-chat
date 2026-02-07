Mixture of Experts (MoE) - Theory & Implementation

🧠 Theory Overview

Core Concept

MoE replaces dense FFN layers with multiple "expert" networks, where only a subset of experts are activated per token. This allows scaling model capacity without proportionally
increasing compute.

Key Benefits:

- Sparse activation: Only 1-2 out of N experts active per token
- Increased capacity: More parameters without linear compute growth
- Specialization: Different experts learn different patterns/domains
- Efficiency: Sub-linear scaling of inference cost

Mathematical Foundation

For input x, instead of one FFN:
y = FFN(x) # Dense: all parameters used

MoE uses:
y = Σ(i=1 to N) G(x)\_i \* Expert_i(x) # Sparse: only top-k experts used

Where:

- G(x) = Gating function (router) - learns which experts to use
- Expert_i(x) = Individual expert networks (typically FFNs)
- Only top-k experts (k=1 or 2) are actually computed

Architecture Components

1. Router/Gating Network: Decides which experts to activate
2. Expert Networks: Individual FFN specialists
3. Load Balancing: Ensures experts are used equally
4. Expert Parallelism: Distributes experts across GPUs/devices

---

🔧 Implementation in Your Codebase

1. MoE MLP Layer

Here's how to replace your current MLP with MoE:

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class MoEMLP(nn.Module):
def **init**(self, config, num_experts: int = 8, top_k: int = 2):
super().**init**()
self.config = config
self.num_experts = num_experts
self.top_k = top_k
self.hidden_size = config.n_emb
self.intermediate_size = 4 \* config.n_emb

          # Router/Gating network
          self.gate = nn.Linear(self.hidden_size, num_experts, bias=False)

          # Expert networks (same structure as your current MLP)
          self.experts = nn.ModuleList([
              nn.ModuleDict({
                  'c_fc': nn.Linear(self.hidden_size, self.intermediate_size),
                  'c_proj': nn.Linear(self.intermediate_size, self.hidden_size)
              }) for _ in range(num_experts)
          ])

          # Mark output projection as residual
          for expert in self.experts:
              expert.c_proj.residual_proj = True

      def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
          batch_size, seq_len, hidden_size = hidden_states.shape
          hidden_states = hidden_states.view(-1, hidden_size)  # (batch_size * seq_len, hidden_size)

          # Router computation
          router_logits = self.gate(hidden_states)  # (batch_size * seq_len, num_experts)
          routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

          # Select top-k experts
          routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
          routing_weights /= routing_weights.sum(dim=-1, keepdim=True)  # Renormalize

          # Initialize output
          final_hidden_states = torch.zeros(
              (batch_size * seq_len, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device
          )

          # Process each expert
          expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

          for expert_idx in range(self.num_experts):
              expert = self.experts[expert_idx]
              idx, top_x = torch.where(expert_mask[expert_idx])

              if idx.shape[0] == 0:
                  continue

              # Get tokens for this expert
              top_x_list = idx.tolist()
              current_state = hidden_states[None, top_x_list].reshape(-1, hidden_size)

              # Forward pass through expert (your squared ReLU activation)
              current_hidden_states = expert.c_fc(current_state)
              current_hidden_states = F.relu(current_hidden_states).square()  # Your activation
              current_hidden_states = expert.c_proj(current_hidden_states)

              # Apply routing weights and accumulate
              current_hidden_states = current_hidden_states * routing_weights[top_x_list, idx, None]
              final_hidden_states.index_add_(0, top_x, current_hidden_states)

          # Auxiliary loss for load balancing
          aux_loss = self.load_balancing_loss(router_logits, selected_experts)

          return final_hidden_states.view(batch_size, seq_len, hidden_size), aux_loss

      def load_balancing_loss(self, router_logits: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
          """Auxiliary loss to encourage balanced expert usage"""
          num_tokens = router_logits.shape[0]

          # Fraction of tokens routed to each expert
          density = torch.mean(torch.nn.functional.one_hot(selected_experts.flatten(), num_classes=self.num_experts).float(), dim=0)

          # Average probability of routing to each expert
          density_prob = torch.mean(F.softmax(router_logits, dim=-1), dim=0)

          # Load balancing loss
          loss = torch.sum(density_prob * density) * self.num_experts
          return loss

2. Updated Block with MoE

Replace your current Block:

class MoEBlock(nn.Module):
def **init**(self, config, num_experts: int = 8, top_k: int = 2):
super().**init**()
self.config = config

          self.attn = CausalSelfAttention(config)  # Keep your existing attention
          self.moe_mlp = MoEMLP(config, num_experts, top_k)  # Replace MLP with MoE

      def forward(self, x, cos_sin, kv_cache=None, layer_idx=0):
          # Attention (unchanged)
          x = x + self.attn(
              norm(x, use_rms=self.config.use_rmsnorm),
              cos_sin,
              kv_cache=kv_cache,
              layer_idx=layer_idx,
          )

          # MoE MLP
          moe_output, aux_loss = self.moe_mlp(norm(x, use_rms=self.config.use_rmsnorm))
          x = x + moe_output

          return x, aux_loss

3. Updated GPTModel with MoE

class MoEGPTModel(nn.Module):
def **init**(self, config, num_experts: int = 8, top_k: int = 2):
super().**init**()
self.config = config
self.num_experts = num_experts
self.top_k = top_k
self.tokenizer = tiktoken.get_encoding("gpt2")

          # ... (same initialization as before)

          # Replace transformer blocks with MoE blocks
          self.transformer = nn.ModuleDict(
              dict(
                  wte=nn.Embedding(config.vocab_size, config.n_emb),
                  wpe=(
                      nn.Embedding(config.block_size, config.n_emb)
                      if not config.use_rope
                      else None
                  ),
                  h=nn.ModuleList([
                      MoEBlock(config, num_experts, top_k) for _ in range(config.n_layer)
                  ]),
              )
          )

          # ... (rest same as before)

      def forward(self, idx, targets=None, kv_cache=None):
          B, T = idx.shape
          cos_sin = self.cos, self.sin

          if self.training:
              kv_cache = None

          x = self.transformer.wte(idx)
          if self.config.use_rope is False:
              positions = torch.arange(0, T, dtype=torch.long, device=idx.device)
              x = x + self.transformer.wpe(positions)

          x = norm(x, use_rms=self.config.use_rmsnorm)

          # Accumulate auxiliary losses
          total_aux_loss = 0.0

          for layer_idx, block in enumerate(self.transformer.h):
              x = x * self.resid_lambdas[layer_idx]
              x, aux_loss = block(x, cos_sin, kv_cache=kv_cache, layer_idx=layer_idx)
              total_aux_loss += aux_loss

          x = norm(x, use_rms=self.config.use_rmsnorm)
          logits = self.lm_head(x)
          logits = self.config.logit_softcap * torch.tanh(logits / self.config.logit_softcap)

          loss = None
          if targets is not None:
              logits_flat = logits.reshape(-1, logits.size(-1))
              targets_flat = targets.reshape(-1)

              # Main loss
              main_loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1)

              # Add auxiliary loss (weighted)
              aux_weight = 0.01  # Hyperparameter to tune
              loss = main_loss + aux_weight * total_aux_loss

          return logits, loss

4. Configuration Updates

Add MoE config to your existing configs:

@dataclass
class MoEGPTConfig: # All your existing config fields
block_size: int = 1024
vocab_size: int = 50304
n_layer: int = 12
n_head: int = 6
n_kv_head: int = 6
n_emb: int = 768
logit_softcap: float = 15.0

      # MoE-specific configs
      num_experts: int = 8
      top_k: int = 2
      aux_loss_weight: float = 0.01

      # ... rest of your config

---

🎯 MoE Strategy for Your 560M Model

Recommended Configuration

# For ~560M total params with MoE

config = MoEGPTConfig(
n_layer=12,
n_emb=768,
num_experts=8, # 8 experts per MoE layer
top_k=2, # Activate top 2 experts
aux_loss_weight=0.01
)

Parameter Analysis:

- Dense model: 560M params
- MoE model: ~600-700M total params, but only ~140-280M active per token
- Expert distribution: Each expert ~70M params, only 2 active = 140M params/token

Training Considerations

1. Load Balancing: Monitor expert usage to ensure balanced training
2. Auxiliary Loss: Start with 0.01 weight, tune based on expert balance
3. Memory: Experts can be stored on different GPUs for scaling
4. Batch Size: May need adjustment due to routing overhead

Expected Benefits

- Capacity: 2-3x model capacity with similar inference cost
- Specialization: Experts may specialize (syntax, reasoning, facts, etc.)
- Quality: Often 10-20% improvement on downstream tasks

This implementation maintains your excellent code structure while adding the MoE capability. The modular design makes it easy to experiment with different expert counts and routing
strategies.
