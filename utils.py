import os
from chat import decode_with_special_tokens
from engine import Engine, Sampler
from gpt import GPTConfig, GPTModel
import torch
import math

def save_checkpoint(path, model, optimizer, step, config=None):
  # make the directory if it doesn't exist
  os.makedirs(os.path.dirname(path), exist_ok=True)

  # save the checkpoint
  checkpoint = {
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "step": step,
      "config": config
  }
  torch.save(checkpoint, path)
  print0(f"Checkpoint saved at {path}")


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


def get_lr_multiplier(step, warmup_steps, max_steps, min_lr_frac):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)

    if step >= max_steps:
        return min_lr_frac

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr_frac + cosine * (1 - min_lr_frac)



def sample_from_model(model, tokenizer, device, context, max_tokens):
    # generate
    sampler = Sampler(temperature=1.0, top_k=50)
    engine = Engine(model, sampler, use_kv_cache=True)

    idx = torch.tensor([tokenizer.encode(context)], dtype=torch.long, device=device)

    token_ids, state = engine.generate(idx, max_new_tokens=max_tokens)
    print0(">>>")
    print0(decode_with_special_tokens(token_ids.squeeze(0).tolist(), tokenizer))
    print0(">>>")


def render_mcq(question, letters, choices):
   str = f"Multiple choice question: {question}\n"
   str += "".join([f"- {c}={l}\n" for l, c in zip(letters, choices)])
   str += f"\nRespond only with the letter of the correct answer."
   return str


def print0(s="", **kwargs):
  ddp_rank = int(os.environ.get("RANK", 0))
  if ddp_rank == 0:
    print(s, **kwargs)


def dataset_defaults(name):
  if name == "fw":
      return dict(
          data_root="download/edu_fineweb10B",
          max_steps=10_000,
      )
  elif name == "ts":
      return dict(
          data_root="download/tinystories",
          max_steps=1_000,
      )
  elif name == "tsk":
      return dict(
          data_root="download/tinysk",
          max_steps=500,
      )
  else:
      raise ValueError(name)
