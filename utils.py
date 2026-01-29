import os
from random import random
import re

import numpy as np
from chat import decode_with_special_tokens
from engine import Engine, Sampler
import torch
import math
from torch.distributed import init_process_group


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config=None,
    train_loader=None,
    val_loader=None,
    rank=0,
):
    if rank == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # save the model state
        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": int(step),
            "config": config,
            # Data loader state
            "train_loader_state": (
                train_loader.state_dict() if train_loader is not None else None
            ),
            "val_loader_state": (
                val_loader.state_dict() if val_loader is not None else None
            ),
        }
        torch.save(ckpt, path + ".tmp")
        os.replace(path + ".tmp", path)
        print0(f"Checkpoint (model) saved at {path}")

    # per rank rng state
    rng_state = get_rng_state()
    rng_path = path.replace(".pt", f".rank{rank}.rng.pt")
    torch.save(rng_state, rng_path)
    print(f"RNG state saved at {rng_path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    strict: bool = True,
    rank: int = 0,
):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Model
    state = ckpt["model_state_dict"]
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state, strict=strict)
    model.to(device)

    # Optimizer
    if (
        optimizer is not None
        and "optimizer_state_dict" in ckpt
        and ckpt["optimizer_state_dict"] is not None
    ):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Move optimizer state tensors to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    step = int(ckpt.get("step", -1))

    # load per rank rng
    rng_path = path.replace(".pt", f".rank{rank}.rng.pt")
    rng_state = torch.load(rng_path, map_location=device, weights_only=False)
    set_rng_state(rng_state)
    print(f"RNG state restored from {rng_path}")

    # return the checkpoint and step
    return ckpt, step


def get_rng_state():
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])


def get_lr(it, max_lr, min_lr, warmup_steps, max_steps):
    # 1) linear warmup for warm_iter steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use consine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # starts at 1 and goes to 0
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


FINAL_NUM_RE = re.compile(r"####\s*([\-0-9\.,]+)")


# used for gsm8k dataset
def extract_final_number(answer: str) -> str:
    match = FINAL_NUM_RE.search(answer)
    if not match:
        return None
    return match.group(1).replace(",", "").strip()


def env_info():
    # distributed data parallel setup
    ddp = int(os.environ.get("RANK", -1)) != -1
    using_cuda = torch.cuda.is_available()

    if ddp:
        # backend from env variable (override for testing)
        backend = os.environ.get("DDP_BACKEND", "nccl" if using_cuda else "gloo")
        init_process_group(backend=backend)

        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])

        if using_cuda:
            device = torch.device(f"cuda:{ddp_local_rank}")
            torch.cuda.set_device(device)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if using_cuda:
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"

    return (
        ddp,
        ddp_rank,
        ddp_local_rank,
        ddp_world_size,
        device,
        master_process,
        using_cuda,
    )
