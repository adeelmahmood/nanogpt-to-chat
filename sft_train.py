import argparse
from contextlib import nullcontext
from datetime import datetime
import math
from dataloader_sft import sft_loader
from gpt import GPTModel, configure_optimizer, get_gpt_config
from tasks import GSM8K, Arc, SmolTalkTask, SpellingTask, TaskMixture
import torch
import time
import os
import tiktoken

import wandb

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from utils import (
    load_checkpoint,
    print0,
    save_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument(
        "--dataset", type=str, choices=["fw", "ts", "tsk"], default="fw"
    )

    # model
    parser.add_argument(
        "--model_depth", type=str, choices=["d12", "d20", "d2"], default="d12"
    )

    # batch
    parser.add_argument("--batch_size", type=int, choices=[4, 8, 16, 32], default=4)
    parser.add_argument("--total_batch_size", type=int, default=524288)
    parser.add_argument("--target_examples_per_step", type=int, default=32)

    # training
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)

    # paths
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--ckpt_out", type=str, default="./ckps")
    parser.add_argument("--resume_ckpt", type=str, default=None)

    return parser.parse_args()


def main():
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
        # torch.backends.cuda.matmul.fp32_precision = "tf32"

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
    config = get_gpt_config(args.model_depth)

    # Initialize the model
    model = GPTModel(config).to(device)

    # Load pretrained state for model and optimizer
    checkpoint = (
        args.resume_ckpt
        or f"{args.ckpt_out}/midtrain_{args.dataset}_{args.model_depth}.pt"
    )
    ckpt, _ = load_checkpoint(
        path=checkpoint, model=model, optimizer=None, device=device
    )

    # initiatlize the optimizer
    optimizer = configure_optimizer(
        model, total_batch_size_tokens=args.total_batch_size, stage="sft"
    )
    for pg in optimizer.param_groups:
        print0(f"{pg['name']}: lr={pg['lr']:.6f}, weight_decay={pg['weight_decay']}")

    # wrap the model in ddp
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    print0(
        f"Model parameters: {sum(p.nelement() for p in model.parameters())/1e6:.2f}M"
    )

    # Hyper parameters
    max_steps = args.max_steps or 900
    B = args.batch_size
    target_examples_per_step = args.target_examples_per_step or 32
    gradient_accum_steps = max(
        1, math.ceil(target_examples_per_step / (B * ddp_world_size))
    )

    print0(f"\nB = {B}, target_examples_per_step = {target_examples_per_step}")
    print0(f"Using gradient accum steps: {gradient_accum_steps}")

    eval_every = args.eval_every or (max_steps // 10)

    if master_process:
        print("\n======== RUN CONFIG ========")
        print(f"model_depth    : {args.model_depth}")
        print(f"batch_size     : {B}")
        print(f"block_size     : {config.block_size}")
        print(f"max_steps      : {max_steps}")
        if hasattr(args, "resume_ckpt"):
            print(f"resume_ckpt    : {args.resume_ckpt}")
        print("============================\n")

    # SFT datasets
    train_task = TaskMixture(
        [
            Arc(subset="ARC-Easy"),  # 2.3k
            Arc(subset="ARC-Challenge"),  # 1.1k
            GSM8K(),  # 8k
            SmolTalkTask(stop=10_000),  # 10k
            SpellingTask(size=500),  # # 500
        ]
    )  # Total = 2.3k + 1.1k + 8k + 10k + 0.5k = 21.9k

    train_loader = sft_loader(
        dataset=train_task,
        batch_size=B,
        max_seq_len=config.block_size,
        tokenizer=tokenizer,
        device=device,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )

    val_loaders = {
        "smoltalk": sft_loader(
            dataset=TaskMixture([SmolTalkTask(split="test")]),
            batch_size=B,
            max_seq_len=config.block_size,
            tokenizer=tokenizer,
            device=device,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        ),
    }

    total_examples = 0
    step = 0

    def get_lr_multiplier(progress):
        return max(1.0 - progress, 0.0)

    print0(f"\nStarting SFT-Training ({datetime.now()})")

    for step in range(max_steps):
        last_step = step == max_steps - 1

        # validation loss (just do it twice)
        if step > 0 and eval_every != -1 and (last_step or step % eval_every == 0):
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
            ckpt_path = os.path.join(
                args.ckpt_out, f"sfttrain_{args.dataset}_{args.model_depth}.pt"
            )
            save_checkpoint(
                ckpt_path,
                model.module if ddp else model,
                optimizer,
                step=step,
                config=config,
            )

        if last_step:
            break

        # train
        synchronize()
        st = time.time()
        loss_accum = torch.zeros(1, device=device)
        num_tokens = torch.tensor(0, device=device)  # number of active tokens

        for grad_step in range(gradient_accum_steps):
            # get a batch
            x, y = next(train_loader)

            # forward pass
            with autocast_ctx:
                _, loss = model(x, y)

            # average the loss across grad accum batch
            loss = loss / gradient_accum_steps
            loss_accum += loss.detach()

            # backward pass
            loss.backward()
            num_tokens += (y >= 0).sum()

        # gather loss from all ranks
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)

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
        optimizer.zero_grad(set_to_none=True)

        # logging
        synchronize()
        et = time.time()
        examples_per_sec = (B * gradient_accum_steps * ddp_world_size) / (et - st)
        total_examples += B * gradient_accum_steps * ddp_world_size
        num_tokens_item = num_tokens.item()
        step_loss = loss_accum.item()
        print0(
            f"step: {step:05d}/{max_steps:05d} | step loss: {step_loss:.4f} | time: {(et-st)*1000:.2f}ms | examples/sec: {examples_per_sec:.2f} | total examples: {total_examples:,} | num_tokens: {num_tokens_item:,}"
        )
        if master_process and send_to_wandb:
            wandb_run.log(
                {
                    "train/loss": step_loss,
                    **lr_logs,
                    "perf/examples_per_sec": examples_per_sec,
                    "progress/total_examples": total_examples,
                    "progress/num_tokens": num_tokens_item,
                },
                step=step,
            )

    if send_to_wandb and master_process:
        wandb_run.finish()

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
