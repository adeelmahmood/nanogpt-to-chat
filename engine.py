from gpt import KVCache
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sampler:
    def __init__(self, temperature=1.0, top_k=50):
        self.temperature = temperature
        self.top_k = top_k

    def sample(self, logits):
        if self.temperature != 1.0:
            logits = logits / self.temperature

        probs = F.softmax(logits, dim=-1)

        if self.top_k is not None:
            topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            next_id = torch.gather(topk_indices, -1, ix)
        else:
            next_id = torch.multinomial(probs, 1)
        
        return next_id
    

class EngineState:
    def __init__(self, kv_cache=None):
        self.kv_cache = kv_cache
        self.generated = 0
        self.stopped = False
        self.stop_reason = None


class Engine:
    def __init__(self, model, sampler, use_kv_cache=True):
        self.model = model
        self.sampler = sampler
        self.use_kv_cache = use_kv_cache

    @torch.inference_mode()
    def prefill(self, tokens, state):
        logits, _ = self.model(tokens, kv_cache=state.kv_cache)
        return logits[:, -1, :]
    

    @torch.inference_mode()
    def decode_next(self, token, state):
        logits, _ = self.model(token, kv_cache=state.kv_cache)
        return logits[:, -1, :]
    

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, stop_token_id=None):
        B, T = idx.shape
        assert B==1, "Engine.generate supports B=1"

        # prefill sequence length check
        if T > self.model.config.block_size:
            idx = idx[:, -self.model.config.block_size:]
            T = idx.size(1)

        # init state
        kv_cache = KVCache(self.model.config.n_layer) if self.use_kv_cache else None
        state = EngineState(kv_cache=kv_cache)

        # prefill 
        next_logits = self.prefill(idx, state)

        for _ in range(max_new_tokens):
            # RoPE sequence length check
            if state.kv_cache is not None and state.kv_cache.seq_len() >= self.model.config.block_size:
                state.stopped = True
                state.stop_reason = "Context window exceeded"
                break

            # sample
            idx_next = self.sampler.sample(next_logits)

            # check for end token
            if stop_token_id is not None and idx_next.item() == stop_token_id:
                state.stopped = True
                state.stop_reason = "Stop token generated"
                break

            idx = torch.cat([idx, idx_next], dim=1)
            state.generated += 1

            # fast path with cache
            if state.kv_cache is not None:
                next_logits = self.decode_next(idx_next, state)
            else:
                # slow path with full context
                if idx.size(1) > self.model.config.block_size:
                    idx = idx[:, -self.model.config.block_size:]
                logits, _ = self.model(idx)
                next_logits = logits[:, -1, :]

        if not state.stopped:
            state.stop_reason = "completed"

        return idx, state

