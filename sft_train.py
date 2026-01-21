import argparse
from contextlib import nullcontext
from datetime import datetime
from dataloader_sft import sft_loader
from gpt import GPTConfig, GPTConfigD20, GPTModel, configure_optimizer
from tasks import GSM8K, MMLU, Arc, SmolTalkTask, TaskMixture
import torch
import time
import os
import tiktoken

import wandb

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from utils import load_checkpoint, print0, sample_from_model, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument(
        "--model_depth", type=str, choices=["d12", "d20"], default="d12"
    )

    # batch
    parser.add_argument("--batch_size", type=int, choices=[4, 8, 16, 32], default=4)
    parser.add_argument("--target_examples_per_step", type=int, default=32)

    # training
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)

    # paths
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--ckpt_out", type=str, default="./ckps")

    # SFT training only
    parser.add_argument("--resume_ckpt", type=str, default=None)

    return parser.parse_args()


args = parse_args()

# distributed data parallel setup
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

send_to_wandb = int(os.environ.get("SEND_TO_WANDB", -1)) != -1
device_type = "cuda" if torch.cuda.is_available() else "cpu"
print0(f"using device: {device} type {device_type}")

if device_type == "cuda":
    torch.set_float32_matmul_precision("high")

autocast_ctx = (
    torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    if device_type == "cuda"
    else nullcontext()
)
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

# log
if send_to_wandb and master_process:
    wandb_run = wandb.init(project="nano-chat", name="sft-train")

# set seeds
torch.manual_seed(1337 + ddp_rank)
torch.cuda.manual_seed(1337 + ddp_rank)

# define the tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# initialize the model
if args.model_depth == "d12":
    config = GPTConfig()
elif args.model_depth == "d20":
    config = GPTConfigD20()
else:
    raise ValueError(f"Unknown model depth: {args.model_depth}")

# Initialize the model
model = GPTModel(config).to(device)

# Load pretrained state for model and optimizer
checkpoint = (
    args.resume_ckpt
    or f"./{args.ckpt_out}/midtrain_{args.dataset}_{args.model_depth}.pt"
)
ckpt, _ = load_checkpoint(path=checkpoint, model=model, optimizer=None, device=device)

# orig_model = model # for saving checkpoints and sampling


# initiatlize the optimizer
optimizer = configure_optimizer(model)

# not compiling because of variable sequence length
# if device_type == "cuda":
#   print0("Compiling model")
#   model = torch.compile(model, dynamic=False)

# wrap the model in ddp
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# scale down learning rates for sft training
for pg in optimizer.param_groups:
    pg["lr"] *= 0.05
    pg["initial_lr"] = pg["lr"]


if master_process:
    print(f"Model parameters: {sum(p.nelement() for p in model.parameters())/1e6:.2f}M")
    print("Loaded midtrained model. Sampling...")
    sample_from_model(
        model.module if ddp else model,
        tokenizer,
        device,
        "Why is sky blue?",
        max_tokens=100,
    )


# Hyper parameters
max_steps = args.max_steps or 900
B = args.batch_size
target_examples_per_step = 32

gradient_accum_steps = target_examples_per_step // (B * ddp_world_size)

print0(f"\nB = {B}, target_examples_per_step = {target_examples_per_step}")
print0(f"Using gradient accum steps: {gradient_accum_steps}")

if master_process:
    print("\n======== RUN CONFIG ========")
    print(f"model_depth    : {args.model_depth}")
    print(f"batch_size     : {B}")
    print(f"block_size     : {config.block_size}")
    print(f"max_steps      : {max_steps}")
    print(f"eval_every     : {args.eval_every}")
    if hasattr(args, "resume_ckpt"):
        print(f"resume_ckpt    : {args.resume_ckpt}")
    print("============================\n")


# SFT datasets
train_task = TaskMixture(
    [
        SmolTalkTask(stop=64),
        MMLU(stop=2_000),
        GSM8K(stop=2_000),
        Arc(stop=2_000),
    ]
)

train_loader = sft_loader(
    model=model,
    dataset=train_task,
    batch_size=B,
    tokenizer=tokenizer,
    device=device,
    ddp_rank=ddp_rank,
    ddp_world_size=ddp_world_size,
)

val_loaders = {
    "smoltalk": sft_loader(
        model=model,
        dataset=TaskMixture([SmolTalkTask(split="test")]),
        batch_size=B,
        tokenizer=tokenizer,
        device=device,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    ),
    "mmlu": sft_loader(
        model=model,
        dataset=TaskMixture([MMLU(subset="all", split="test", stop=5200)]),
        batch_size=B,
        tokenizer=tokenizer,
        device=device,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    ),
    "gsm8k": sft_loader(
        model=model,
        dataset=TaskMixture([GSM8K(split="test", stop=420)]),
        batch_size=B,
        tokenizer=tokenizer,
        device=device,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    ),
}

total_time = 0
total_examples = 0
step = 0

ema_loss = 0.0
ema_initialized = False
alpha = 0.98


def get_lr_multiplier(progress):
    return max(1.0 - progress, 0.0)


print0(f"\nStarting SFT-Training ({datetime.now()})")

for step in range(max_steps):
    last_step = step == max_steps - 1

    # validation loss
    if args.eval_every and step > 0 and (step % args.eval_every == 0 or last_step):
        model.eval()
        with torch.no_grad():
            val_loss_steps = 10
            val_metrics = {}
            for name, loader in val_loaders.items():
                val_loss_accum = 0.0
                for _ in range(val_loss_steps):
                    x, y = next(loader)
                    with autocast_ctx:
                        _, loss = model(x, y)
                    val_loss_accum += loss.detach()
                val_loss_accum /= val_loss_steps

                if ddp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

                val_metrics[f"val/{name}_loss"] = val_loss_accum.item()
                print0(f"val/{name}_loss: {val_loss_accum.item():.4f}")

        if master_process and send_to_wandb:
            wandb_run.log(val_metrics, step=step)
        model.train()

    # save checkpoint
    if master_process and last_step:
        ckpt_path = os.path.join(args.ckpt_out, f"sft-train_{args.model_depth}.pt")
        save_checkpoint(ckpt_path, model, optimizer, step=step, config=config)

    if last_step:
        break

    # train
    synchronize()
    st = time.time()
    optimizer.zero_grad(set_to_none=True)
    loss_accum = torch.zeros(1, device=device)

    for grad_step in range(gradient_accum_steps):
        # get a batch
        x, y = next(train_loader)

        # this prevents synching of gradients across ranks until grad accumulation is done
        if ddp:
            model.require_backward_grad_sync = grad_step == gradient_accum_steps - 1

        # forward pass
        with autocast_ctx:
            _, loss = model(x, y)

        # average the loss across grad accum batch
        loss = loss / gradient_accum_steps
        loss_accum += loss.detach()

        # update EMA of loss
        step_loss = loss_accum.item()
        if step == 0:
            ema_loss = step_loss
        else:
            ema_loss = alpha * ema_loss + (1 - alpha) * step_loss

        # backward pass
        loss.backward()

    # gather loss from all ranks
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # clip gradients
    clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # get learning rate
    progress = step / max_steps
    lrm = get_lr_multiplier(progress)
    lr_logs = {
        "lr/multiplier": lrm,
    }
    for pg in optimizer.param_groups:
        pg["lr"] = pg["initial_lr"] * lrm
        # for logging
        name = pg.get("name", "unknown")
        lr_logs[f"lr/{name}"] = pg["lr"]

    # update
    optimizer.step()

    # logging
    synchronize()
    et = time.time()
    examples_per_sec = (B * gradient_accum_steps * ddp_world_size) / (et - st)
    total_time += et - st
    total_examples += B * gradient_accum_steps * ddp_world_size
    print0(
        f"step: {step:05d}/{max_steps:05d} | step loss: {step_loss:.4f} | ema loss: {ema_loss:.4f} | norm {clipped_norm:.4f} | time: {(et-st)*1000:.2f}ms | examples/sec: {examples_per_sec:.2f} | total time: {total_time/60:.2f}m | total examples: {total_examples:,}"
    )
    if master_process and send_to_wandb:
        wandb_run.log(
            {
                "train/loss": loss_accum.item(),
                "train/grad_norm_clipped": clipped_norm.item(),
                **lr_logs,
                "perf/examples_per_sec": examples_per_sec,
                "progress/total_time": total_time,
                "progress/total_examples": total_examples,
            },
            step=step,
        )


if master_process:
    model.eval()
    print("Finished SFT-Training. Sampling...")
    sample_from_model(
        model.module if ddp else model,
        tokenizer,
        device,
        "Why is sky blue?",
        max_tokens=100,
    )

if send_to_wandb and master_process:
    wandb_run.finish()

if ddp:
    destroy_process_group()
