import argparse
from contextlib import nullcontext
from datetime import datetime
import math
from dataloader import DataLoaderLite
from gpt import (
    GPTModel,
    configure_optimizer,
    get_gpt_config,
)
import torch
import time
import os

import wandb

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from logger import MetricLogger
from utils import (
    env_info,
    get_lr_multiplier,
    load_checkpoint,
    print0,
    save_checkpoint,
)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument(
        "--dataset", type=str, choices=["fw", "ts", "tsk"], default="fw"
    )

    # model
    parser.add_argument(
        "--model_depth", type=str, choices=["d12", "d20"], default="d12"
    )

    # batch
    parser.add_argument("--batch_size", type=int, choices=[4, 8, 16, 32], default=16)
    parser.add_argument("--total_batch_size", type=int, default=524288)
    parser.add_argument("--compile_model", type=bool, default=False)

    # training
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=-1)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)

    # paths
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--ckpt_out", type=str, default="./ckps")
    parser.add_argument("--resume_ckpt", type=str, default=None)

    # logging
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--log_file", type=str, default=None)

    # architecture params (only for ablation experiments)
    # (keeping None as default will use config defaults based on model depth)
    parser.add_argument("--block_size", type=int, default=None)
    # parser.add_argument("--vocab_size", type=int, default=50304)
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_kv_head", type=int, default=None)
    parser.add_argument("--n_emb", type=int, default=None)
    parser.add_argument("--logit_softcap", type=float, default=None)

    parser.add_argument(
        "--pos_emb_type", type=str, choices=["rope", "absolute", "none"], default=None
    )
    parser.add_argument("--use_rmsnorm", type=str2bool, default=True)
    parser.add_argument("--use_qk_norm", type=str2bool, default=True)
    parser.add_argument(
        "--attn_type", type=str, choices=["mha", "gqa", "mqa"], default="mha"
    )
    parser.add_argument("--use_kv_cache", type=str2bool, default=True)

    parser.add_argument("--lr_alpha", type=float, default=0.55)
    parser.add_argument("--matrix_lr_alpha", type=float, default=0.16)
    parser.add_argument("--embed_lr_alpha", type=float, default=1.0)
    parser.add_argument("--resid_lambda_alpha", type=float, default=1.0)

    return parser.parse_args()


def main():

    args = parse_args()

    logger = MetricLogger(args.log_dir, file_name=args.log_file)

    # env setup
    (
        ddp,
        ddp_rank,
        ddp_local_rank,
        ddp_world_size,
        device,
        master_process,
        using_cuda,
    ) = env_info()

    send_to_wandb = int(os.environ.get("SEND_TO_WANDB", -1)) != -1
    device_type = "cuda" if using_cuda else "cpu"
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
        # grab RUN_ID env variable
        run_id = os.environ.get("RUN_ID", "")
        resume = "allow" if run_id != "" else None
        print(f"WandB run id: {run_id}, resume: {resume}")
        wandb_run = wandb.init(
            project="nano-chat", name="pretrain" + str(run_id), resume=resume
        )

    # set seeds
    base_seed = 1337 + ddp_rank
    torch.manual_seed(base_seed)
    if using_cuda:
        torch.cuda.manual_seed(base_seed)

    # initialize the model
    config = get_gpt_config(args.model_depth, args=args)
    if master_process:
        print("\n======== MODEL CONFIG ========")
        for k in sorted(vars(config).keys()):
            print(f"{k}: {getattr(config, k)}")
        print("============================\n")

    # Hyper parameters
    max_steps = args.max_steps  # 19073 # 10B / 524288
    warmup_steps = int(0.01 * max_steps)  # 1% of max steps

    B = args.batch_size
    T = config.block_size
    total_batch_size = args.total_batch_size or (B * T * ddp_world_size)
    gradient_accum_steps = max(
        1, math.ceil(total_batch_size / (B * T * ddp_world_size))
    )

    # if max token limit is set, adjust max steps to be large but not infinite
    if args.max_tokens != -1:
        orig_max_steps = max_steps
        max_steps = min(max_steps, math.ceil(args.max_tokens / total_batch_size))

    model = GPTModel(config).to(device)

    # initiatlize the optimizer
    optimizer = configure_optimizer(model, total_batch_size_tokens=total_batch_size)
    for pg in optimizer.param_groups:
        print0(f"{pg['name']}: lr={pg['lr']:.6f}, weight_decay={pg['weight_decay']}")
        # fixed alpha
        if args.lr_alpha != 1.0:
            pg["initial_lr"] *= args.lr_alpha
            print0(f">Applied lr_alpha={args.lr_alpha}, new ilr={pg['initial_lr']:.6f}")
        # layer level alpha
        if args.matrix_lr_alpha != 1.0 and pg.get("name") == "matrix":
            pg["initial_lr"] *= args.matrix_lr_alpha
            print0(
                f">>Applied matrix_lr_alpha={args.matrix_lr_alpha}, new ilr={pg['initial_lr']:.6f}"
            )
        if args.embed_lr_alpha != 1.0 and pg.get("name") == "embed":
            pg["initial_lr"] *= args.embed_lr_alpha
            print0(
                f">>Applied embed_lr_alpha={args.embed_lr_alpha}, new ilr={pg['initial_lr']:.6f}"
            )
        if args.resid_lambda_alpha != 1.0 and pg.get("name") == "resid_lambda":
            pg["initial_lr"] *= args.resid_lambda_alpha
            print0(
                f">>Applied resid_lambda_alpha={args.resid_lambda_alpha}, new ilr={pg['initial_lr']:.6f}"
            )

    num_params = sum(p.nelement() for p in model.parameters())
    print0(f"\nModel parameters: {num_params/1e6:.2f}M")
    target_total_tokens = max_steps * total_batch_size

    print0(f"B = {B}, T = {T}")
    print0(f"Using gradient accum steps: {gradient_accum_steps}")
    print0(f"Total batch size: {total_batch_size}")
    print0(f"Dataset folder: {args.dataset_dir}")
    print0(f"Target training tokens: {target_total_tokens:,}")
    print0(f"Tokens to params ration: {target_total_tokens / num_params:.2f}")
    print0(
        f"Based on ~20 T-to-P ratio, the steps should be {(20 * num_params) / total_batch_size:.2f}"
    )

    # data loader
    train_loader = DataLoaderLite(
        B=B,
        T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="train",
        data_root=args.dataset_dir,
        master_process=master_process,
    )
    val_loader = DataLoaderLite(
        B=B,
        T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="val",
        data_root=args.dataset_dir,
        master_process=master_process,
    )

    loaded_step = -1
    # Resume from checkpoint if specified
    if args.resume_ckpt is not None:
        if master_process:
            print0(f"Resuming from checkpoint: {args.resume_ckpt}")
        # Ensure all ranks wait for rank0 message
        if ddp:
            dist.barrier()

        ckpt, loaded_step = load_checkpoint(
            path=args.resume_ckpt,
            model=model.module if ddp and hasattr(model, "module") else model,
            optimizer=optimizer,
            device=device,
            strict=True,
            rank=ddp_rank,
        )

        if ckpt.get("train_loader_state", None) is not None:
            train_loader.load_state_dict(ckpt["train_loader_state"])
        if ckpt.get("val_loader_state", None) is not None:
            val_loader.load_state_dict(ckpt["val_loader_state"])

        # Optional: replace config from checkpoint to be safe
        if ckpt.get("config", None) is not None:
            config = ckpt["config"]

        if ddp:
            dist.barrier()

        if master_process:
            print0(f"Loaded checkpoint at step={loaded_step}")

    # compile
    if args.compile_model or device_type == "cuda":
        print0("Compiling model")
        model = torch.compile(model, dynamic=False)

    # wrap ddp
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank] if using_cuda else None)

    eval_every = args.eval_every or (max_steps // 10)
    save_every = args.save_every or (max_steps // 10)

    if master_process:
        print("\n======== RUN CONFIG ========")
        print(f"dataset        : {args.dataset}")
        print(f"dataset_dir    : {args.dataset_dir}")
        print(f"model_depth    : {args.model_depth}")
        print(f"batch_size     : {B}")
        print(f"block_size     : {config.block_size}")
        print(
            f"max_steps      : {max_steps if args.max_tokens == -1 else f'{orig_max_steps} (adjusted to {max_steps} due to max_tokens: {args.max_tokens:,})'}"
        )
        print(f"eval_every     : {eval_every}")
        print(f"save_every     : {save_every}")
        if hasattr(args, "resume_ckpt") and args.resume_ckpt is not None:
            print(f"resume_ckpt    : {args.resume_ckpt}")
            print(f"Resuming from step={loaded_step}")
        print("============================\n")

    total_time = 0.0
    total_tokens = 0
    print0(f"\nStarting Training ({datetime.now()})")

    start_step = loaded_step if loaded_step >= 0 else 0
    for step in range(start_step, max_steps):
        last_step = step == max_steps - 1

        # validation loss
        if (
            step > start_step
            and eval_every != -1
            and (step % eval_every == 0 or last_step)
        ):
            model.eval()
            # val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with autocast_ctx:
                        _, loss = model(x, y)
                    val_loss_accum += loss.detach()
                val_loss_accum /= val_loss_steps

            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            print0(f"Validation loss: {val_loss_accum.item():.4f}")
            if master_process and send_to_wandb:
                wandb_run.log(
                    {"val/loss": val_loss_accum.item()},
                    step=step,
                )
            model.train()

        # save checkpoint
        if last_step or (
            save_every != -1 and step > start_step and step % save_every == 0
        ):
            ckp_prefix = f"pretrain_{args.dataset}_{args.model_depth}"
            ckpt_name = (
                f"{ckp_prefix}.pt" if last_step else f"{ckp_prefix}_{step:05d}.pt"
            )
            ckpt_path = os.path.join(args.ckpt_out, ckpt_name)
            save_checkpoint(
                ckpt_path,
                model.module if ddp and hasattr(model, "module") else model,
                optimizer,
                step=step,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                rank=ddp_rank,
            )

        if last_step:
            break

        # training
        synchronize()
        st = time.time()
        loss_accum = torch.zeros(1, device=device)

        for grad_step in range(gradient_accum_steps):
            # get a batch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # this prevents synching of gradients across ranks until grad accumulation is done
            if ddp:
                model.require_backward_grad_sync = grad_step == gradient_accum_steps - 1

            # forward pass
            with autocast_ctx:
                _, loss = model(x, y)

            # average the loss across grad accum batch
            loss = loss / gradient_accum_steps
            loss_accum += loss.detach()

            # backward pass
            loss.backward()

        # ---- grad norm (no clipping) ----
        total_grad_sq = torch.zeros(1, device=device)
        for p in model.parameters():
            if p.grad is not None:
                total_grad_sq += p.grad.detach().pow(2).sum()
        grad_norm = total_grad_sq.sqrt()

        # gather loss from all ranks
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # get learning rate multiplier
        lrm = get_lr_multiplier(step, warmup_steps, max_steps, 0.0)
        lr_logs = {
            "lr/multiplier": lrm,
        }
        for pg in optimizer.param_groups:
            pg["lr"] = pg["initial_lr"] * lrm
            # for logging
            name = pg.get("name", "unknown")
            lr_logs[f"lr/{name}"] = pg["lr"]

        # ---- matrix update ratio ----
        matrix_update_sq = torch.zeros(1, device=device)
        matrix_weight_sq = torch.zeros(1, device=device)

        for pg in optimizer.param_groups:
            if pg.get("name") == "matrix":
                lr = pg["initial_lr"] * lrm  # current effective LR
                for p in pg["params"]:
                    if p.grad is not None:
                        matrix_update_sq += (lr * p.grad).detach().pow(2).sum()
                        matrix_weight_sq += p.detach().pow(2).sum()

        matrix_update_ratio = matrix_update_sq.sqrt() / (
            matrix_weight_sq.sqrt() + 1e-12
        )

        if ddp:
            dist.all_reduce(matrix_update_ratio, op=dist.ReduceOp.AVG)

        # update
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # gather stats
        synchronize()
        et = time.time()
        step_tokens = (
            train_loader.B * train_loader.T * gradient_accum_steps * ddp_world_size
        )
        tok_sec = step_tokens / (et - st)
        total_time += et - st
        total_tokens += step_tokens
        avg_time_per_step = total_time / max(1, step - start_step + 1)
        remaining_steps = max_steps - step - 1
        eta_seconds = remaining_steps * avg_time_per_step
        matrix_lr = lr_logs["lr/matrix"]

        # logging
        if master_process:
            print0(
                f"step: {step:05d}/{max_steps:05d} | loss: {loss_accum.item():.4f} | matrix lr {matrix_lr:.4e} | time: {(et-st)*1000:.2f}ms | tok-sec: {tok_sec:.2f} | total time: {total_time/60:.2f}m | eta: {eta_seconds/60:.1f}m | total tokens: {total_tokens:,}"
            )
            logger.log(
                step=step,
                train_loss=loss_accum.item(),
                lr_matrix=matrix_lr,
                tok_per_sec=tok_sec,
                total_time=total_time,
                eta_seconds=eta_seconds,
                total_tokens=total_tokens,
                grad_norm=grad_norm.item(),
                update_ratio_matrix=matrix_update_ratio.item(),
            )

        if master_process and send_to_wandb:
            wandb_run.log(
                {
                    "train/loss": loss_accum.item(),
                    **lr_logs,
                    "perf/tok_per_sec": tok_sec,
                    "progress/total_time": total_time,
                    "progress/total_tokens": total_tokens,
                },
                step=step,
            )

    print0(f"Training completed {datetime.now()}")

    # Delete all step checkpoints
    # if master_process:
    #     print0("Deleting step checkpoints as training has completed")
    #     for f in os.listdir(args.ckpt_out):
    #         if f.startswith(
    #             f"pretrain_{args.dataset}_{args.model_depth}_"
    #         ) and f.endswith(".pt"):
    #             os.remove(os.path.join(args.ckpt_out, f))

    if send_to_wandb and master_process:
        wandb_run.finish()

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
