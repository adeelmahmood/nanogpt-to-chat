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

