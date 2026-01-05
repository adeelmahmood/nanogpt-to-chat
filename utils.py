from chat import decode_with_special_tokens
from engine import Engine, Sampler
from gpt import GPTConfig, GPTModel
import torch
import math

def save_checkpoint(path, model, optimizer, step, config=None):
  checkpoint = {
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "step": step,
      "config": config
  }
  torch.save(checkpoint, path)
  print(f"Checkpoint saved at {path}")


def load_from_checkpoint(
    model,
    ckp_path,
    device,
    optimizer=None,
    strict=True,
):
    # Load checkpoint
    checkpoint = torch.load(
        ckp_path,
        map_location=device,
        weights_only=False,
    )

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model.to(device)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Ensure optimizer tensors are on the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    step = checkpoint.get("step", 0)
    config = checkpoint.get("config", None)

    return model, optimizer, step, config
  

def get_lr(it, max_lr, min_lr, warmup_steps, max_steps):
  # 1) linear warmup for warm_iter steps
  if it < warmup_steps:
    return max_lr * (it+1) / warmup_steps
  # 2) if it > lr_decay_iters, return min learning rate
  if it > max_steps:
    return min_lr
  # 3) in between, use consine decay
  decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # starts at 1 and goes to 0
  return min_lr + coeff * (max_lr - min_lr)



def sample_from_model(model, tokenizer, device, context, max_tokens):
    # generate
    sampler = Sampler(temperature=1.0, top_k=50)
    engine = Engine(model, sampler, use_kv_cache=True)

    idx = torch.tensor([tokenizer.encode(context)], dtype=torch.long, device=device)

    token_ids, state = engine.generate(idx, max_new_tokens=max_tokens)
    print(">>>")
    print(decode_with_special_tokens(token_ids.squeeze(0).tolist(), tokenizer))
    print(">>>")