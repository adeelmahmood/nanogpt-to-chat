import argparse
import os
from chat import decode_with_special_tokens
from engine import Engine, Sampler
import torch
import math


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument(
        "--dataset", type=str, choices=["fw", "ts", "tsk"], default="tsk"
    )

    # model
    parser.add_argument(
        "--model_depth", type=str, choices=["d12", "d20"], default="d12"
    )

    # batch
    parser.add_argument("--batch_size", type=int, choices=[4, 8, 16, 32], default=4)
    parser.add_argument("--total_batch_size", type=int, default=524288)
    parser.add_argument("--compile_model", type=bool, default=True)

    # training
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)

    # paths
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--ckpt_out", type=str, default="./ckps")
    parser.add_argument("--resume_ckpt", type=str, default=None)

    return parser.parse_args()


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config=None,
    train_loader=None,
    val_loader=None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(step),
        "config": config,
        # RNG
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        # Data loader state
        "train_loader_state": (
            train_loader.state_dict() if train_loader is not None else None
        ),
        "val_loader_state": val_loader.state_dict() if val_loader is not None else None,
    }

    torch.save(ckpt, path)
    print0(f"Checkpoint saved at {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    strict: bool = True,
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
    return ckpt, step


def restore_rng(ckpt: dict):
    rng = ckpt.get("rng_state", None)
    if rng is None:
        return

    try:
        torch.set_rng_state(rng["torch"])
    except Exception:
        pass
    if torch.cuda.is_available() and rng.get("cuda", None) is not None:
        try:
            torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception:
            pass


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
