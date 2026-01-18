from contextlib import nullcontext
from datetime import datetime
from dataloader_midtrain_bos import midtraining_loader_bos
from gpt import GPTConfig, GPTConfigD20, GPTModel, configure_optimizer
from tasks import GSM8K, MMLU, Arc, SmolTalkTask, TaskMixture
import torch
import time
import os
import tiktoken
from tqdm import tqdm

import wandb

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from utils import load_from_checkpoint, print0, sample_from_model, save_checkpoint


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
if send_to_wandb and master_process:
  wandb_run = wandb.init(project="nano-chat", name="mid-train")

# set seeds
torch.manual_seed(1337 + ddp_rank)
torch.cuda.manual_seed(1337 + ddp_rank)

# define the tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# initialize the model
config = GPTConfigD20() 
model = GPTModel(config)
model = model.to(device)

# load pretrained state for model and optimizer
checkpoint = "./ckps/model_00499_1768513334.5296981.pt"
model, _, _, _ = load_from_checkpoint(model, checkpoint, device)
orig_model = model # for saving checkpoints and sampling
# model = torch.compile(model, dynamic=False)

# initiatlize the optimizer
optimizer = configure_optimizer(model)

# scale down learning rates for midtraining
for pg in optimizer.param_groups:
    pg["lr"] *= 0.5
    pg["initial_lr"] *= 0.5

# wrap the model in ddp
if ddp:
  model = DDP(model, device_ids=[ddp_local_rank])

if master_process:
  print(f"Model parameters: {sum(p.nelement() for p in model.parameters())/1e6:.2f}M")
  print("Loaded pretrained model. Sampling...")
  sample_from_model(orig_model, tokenizer, device, "Sam", max_tokens=100)


# Hyper parameters
max_steps = 100 # 5_000
B = 4
T = config.block_size
total_batch_size = 1*B*T # 524288
gradient_accum_steps = total_batch_size // (B*T*ddp_world_size) # 128 or 32
sg = False

print0(f"\nB = {B}, T = {T}")
print0(f"Using gradient accum steps: {gradient_accum_steps}")
print0(f"Total batch size: {total_batch_size}")

# Midtraining datasets
train_task = TaskMixture([
    SmolTalkTask(), 
    MMLU(), MMLU(),
    GSM8K(), GSM8K(),
    Arc(), Arc()
])

train_loader = midtraining_loader_bos(
    tokenizer,
    train_task,
    batch_size=B, seq_len=T,
    device=device, ddp_rank=ddp_rank, ddp_world_size=ddp_world_size
)

val_loaders = {
  "smoltalk": midtraining_loader_bos(
    tokenizer,
    TaskMixture([SmolTalkTask(split="test")]),
    batch_size=B,seq_len=T, 
    device=device, ddp_rank=ddp_rank,ddp_world_size=ddp_world_size
  ),
  "mmlu": midtraining_loader_bos(
    tokenizer,
    TaskMixture([MMLU(subset="all", split="test")]),
    batch_size=B,seq_len=T, 
    device=device, ddp_rank=ddp_rank,ddp_world_size=ddp_world_size
  ),
  "gsm8k": midtraining_loader_bos(
    tokenizer,
    TaskMixture([GSM8K(split="test")]),
    batch_size=B,seq_len=T, 
    device=device, ddp_rank=ddp_rank,ddp_world_size=ddp_world_size
  ),
}

total_time = 0
total_tokens = 0
step = 0

def get_lr_multiplier(progress):
    # clamp progress to [0, 1]
    progress = min(max(progress, 0.0), 1.0)

    # first 80%: flat LR, last 20%: linear decay to zero
    if progress < 0.8:
        return 1.0
    else:
        return max(1.0 - (progress - 0.8) / 0.2, 0.0)

print0(f"\nStarting Mid-Training ({datetime.now()})")

for step in range(max_steps):
  last_step = step == max_steps - 1

  # validation loss
  if step > 0 and (step % 10 == 0 or last_step):
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
  if last_step and master_process:
    save_checkpoint(
        f"./ckps/midtrain_model_{step:05d}.pt",
        orig_model,
        optimizer,
        step=step
    )
    print(f"Checkpoint saved at step {step}")

  if last_step:
    break

  # train
  synchronize()
  st = time.time()
  optimizer.zero_grad(set_to_none=True)
  loss_accum = torch.zeros(1, device=device)

  for grad_step in (tqdm(range(gradient_accum_steps), desc="Grad Steps", leave=False) if sg else range(gradient_accum_steps)):
    # get a batch
    x, y = next(train_loader)

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
  clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

  # get learning rate
  progress = step / max_steps
  lrm = get_lr_multiplier(progress)
  lr_logs = { "lr/multiplier": lrm, }
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
  tok_sec = (B * T * gradient_accum_steps * ddp_world_size) / (et-st)
  total_time += (et-st)
  total_tokens += (B * T * gradient_accum_steps * ddp_world_size)
  print0(f"step: {step:05d}/{max_steps:05d} | loss: {loss_accum.item():.4f} | norm {clipped_norm:.4f} | time: {(et-st)*1000:.2f}ms | tok-sec: {tok_sec:.2f} | total time: {total_time/60:.2f}m | total tokens: {total_tokens:,}")

  if master_process and send_to_wandb:
    wandb_run.log(
        {
            "train/loss": loss_accum.item(),
            "train/grad_norm_clipped": clipped_norm.item(),
            
            **lr_logs,

            "perf/tok_per_sec": tok_sec,

            "progress/total_time": total_time,
            "progress/total_tokens": total_tokens,
        },
        step=step,
    )


if master_process:
  model.eval()
  print("Finished mitraining. Sampling...")
  sample_from_model(orig_model, tokenizer, device, "Sam", max_tokens=100)

if send_to_wandb and master_process: 
  wandb_run.finish()

if ddp:
  destroy_process_group()

