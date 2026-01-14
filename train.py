from contextlib import nullcontext
from datetime import datetime
from dataloader import DataLoaderLite
from gpt import GPTConfig, GPTModel, configure_optimizer
import torch
import time
import os
import tiktoken
from tqdm import tqdm

import wandb

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from utils import get_lr, print0, save_checkpoint


# distributed data parallel setup
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
  init_process_group(backend='nccl')
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
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
print0(f"using device: {device} type {device_type}")

if device_type == "cuda":
  torch.set_float32_matmul_precision('high')

autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

# log
if send_to_wandb:
  wandb_run = wandb.init(project="nano-chat", name="pre-train")

# set seeds
torch.manual_seed(1337 + ddp_rank)
torch.cuda.manual_seed(1337 + ddp_rank)

# define the tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# initialize the model
model = GPTModel(GPTConfig(vocab_size=50304))
model = model.to(device)
orig_model = model # for saving checkpoints and sampling
# model = torch.compile(model, dynamic=False)

# wrap the model in ddp
if ddp:
  model = DDP(model, device_ids=[ddp_local_rank])


# initiatlize the optimizer
optimizer = configure_optimizer(model, lr=3e-4)
print0(f"Model parameters: {sum(p.nelement() for p in model.parameters())/1e6:.2f}M")


# Hyper parameters
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 50
max_steps = 100 # 19073 # 10B / 524288

B = 8
T = 1024
total_batch_size = 1*B*T # 524288
gradient_accum_steps = total_batch_size // (B*T*ddp_world_size) # 128 or 32
data_set_folder = "files/tinystories"
sg = False

print0(f"\nB = {B}, T = {T}")
print0(f"Using gradient accum steps: {gradient_accum_steps}")
print0(f"Total batch size: {total_batch_size}")
print0(f"Dataset folder: {data_set_folder}")

# data loader
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", data_root=data_set_folder, master_process=master_process)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", data_root=data_set_folder, master_process=master_process)

total_time = 0
total_tokens = 0
print0(f"\nStarting Training ({datetime.now()})")

for step in range(1, max_steps):
  last_step = (step == max_steps -1)

  # validation loss
  if step > 0 and (step % 100 == 0 or last_step):
    model.eval()
    val_loader.reset()
    with torch.no_grad():
      val_loss_accum = 0.0
      val_loss_steps = 20
      for _ in range(val_loss_steps):
        x, y = val_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
          _, loss = model(x, y)
        loss = loss / val_loss_steps
        val_loss_accum += loss.detach()
    
    if ddp:
      dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    print0(f"Validation loss: {val_loss_accum.item():.4f}")
    model.train()
  
  # save checkpoint
  if last_step and master_process:
    save_checkpoint(
        f"./ckps/model_{step:05d}_{time.time()}.pt",
        orig_model,
        optimizer,
        step=step,
    )
    print(f"Checkpoint saved at step {step}")


  # training
  synchronize()
  st = time.time()
  optimizer.zero_grad(set_to_none=True)
  loss_accum = torch.zeros(1, device=device)

  for grad_step in (tqdm(range(gradient_accum_steps), desc="Grad Steps", leave=False) if sg else range(gradient_accum_steps)):
    # get a batch
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)

    # this prevents synching of gradients across ranks until grad accumulation is done
    if ddp:
      model.require_backward_grad_sync = (grad_step == gradient_accum_steps-1)

    # forward pass
    with autocast_ctx:
      _, loss = model(x, y)

    # average the loss across grad accum batch
    loss = loss / gradient_accum_steps
    loss_accum += loss.detach()

    # backward pass
    loss.backward()

  # gather loss from all ranks
  if ddp:
    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

  # clip gradients
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

  # get learning rate
  lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps)
  for pg in optimizer.param_groups:
    pg["lr"] = lr

  # update
  optimizer.step()

  # logging
  synchronize()
  et = time.time()
  step_tokens = train_loader.B * train_loader.T * gradient_accum_steps * ddp_world_size
  tok_sec = step_tokens / (et-st)
  total_time += (et-st)
  total_tokens += step_tokens
  avg_time_per_step = total_time / step
  remaining_steps = max_steps - step
  eta_seconds = remaining_steps * avg_time_per_step
  print0(f"step: {step:05d}/{max_steps:05d} | loss: {loss_accum.item():.4f} | lr {lr:.4e} | norm {norm:.4f} | time: {(et-st)*1000:.2f}ms | tok-sec: {tok_sec:.2f} | total time: {total_time/60:.2f}m | eta: {eta_seconds/60:.1f}m | total tokens: {total_tokens:,}")

  if master_process and send_to_wandb:
    wandb_run.log({
        "step": step,
        "train_loss": loss_accum.item(),
        "grad_norm": norm.item(),
        "lr": lr,
        "tok_per_sec": tok_sec,
        "total_time": total_time,
        "total_tokens": total_tokens,
    })

print("Training completed", datetime.now())

if send_to_wandb: 
  wandb_run.finish()
  
if ddp:
  destroy_process_group()