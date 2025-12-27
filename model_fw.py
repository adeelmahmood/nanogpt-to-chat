from dataloader import DataLoaderLite
from gpt import GPTConfig, GPTModel, configure_optimizer
import torch
import time

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import tiktoken

import os

from utils import get_lr, save_checkpoint

ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
  init_process_group(backend='nccl')
  ddp_rank = int(os.environ["RANK"])
  ddp_local_rank = int(os.environ["LOCAL_RANK"])
  ddp_world_size = int(os.environ["WORLD_SIZE"])
  device = f"cuda:{ddp_local_rank}"
  torch.cuda.set_device(device)
  master_process = ddp_local_rank == 0
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
  

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
if master_process:
  print(f"using device: {device} type {device_type}")

# define the tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

torch.set_float32_matmul_precision('high')

# initialize the model
model = GPTModel(GPTConfig(vocab_size=50304)) 
model = model.to(device)
# model = torch.compile(model)

# wrap the model in ddp
if ddp:
  model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# initiatlize the optimizer
optimizer = configure_optimizer(raw_model, lr=3e-4)
if master_process:
  print(f"Model parameters: {sum(p.nelement() for p in model.parameters())/1e6:.2f}M")


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 5 # 19073 # 10B / 524288


# Training Loop
ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16) # if use_bf16 else nullcontext()

torch.manual_seed(1337 + ddp_rank)
torch.cuda.manual_seed(1337 + ddp_rank)

# Batch parameters
B = 4
T = 1024
total_batch_size = B*T # 524288
gradient_accum_steps = total_batch_size // (B*T*ddp_world_size) # 128 or 32
if master_process:
  print(f"B = {B}, T = {T}")
  print(f"Using gradient accum steps: {gradient_accum_steps}")
  print(f"Total batch size: {total_batch_size}")

# data loader
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", data_root="files/tinysk", master_process=master_process)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", data_root="files/tinysk", master_process=master_process)

for i in range(max_steps):
  st = time.time()
  last_step = (i == max_steps -1)

  # validation loss
  if i > 0 and (i % 250 == 0 or last_step):
    model.eval()
    val_loader.reset()
    with torch.no_grad():
      val_loss_accum = 0.0
      val_loss_steps = 20
      for _ in range(val_loss_steps):
        x, y = val_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with ctx:
          _, loss = model(x, y)
        loss = loss / val_loss_steps
        val_loss_accum += loss.detach()
    
    if ddp:
      dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    if master_process:
      print(f"Validation loss: {val_loss_accum.item():.4f}")
  
  # save checkpoint
  if i > 0 and (i % 5000 == 0 or last_step) and master_process and False:
    save_checkpoint(
        f"./ckps/fw_model_{i:05d}.pt",
        raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model,
        optimizer,
        step=i,
        config=raw_model.config
    )
    print(f"Checkpoint saved at step {i}")


  # training
  model.train()
  optimizer.zero_grad()
  loss_accum = torch.zeros(1, device=device)

  for grad_step in range(gradient_accum_steps):
    # get a batch
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)

    # this prevents synching of gradients across ranks until grad accumulation is done
    if ddp:
      model.require_backward_grad_sync = (grad_step == gradient_accum_steps-1)

    # forward pass
    with ctx:
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
  lr = get_lr(i, max_lr, min_lr, warmup_steps, max_steps)
  for pg in optimizer.param_groups:
    pg["lr"] = lr

  # update
  optimizer.step()

  # synchronize to time
  if 'cuda' in device:
    torch.cuda.synchronize()

  et = time.time()
  tok_sec = (train_loader.B * train_loader.T * gradient_accum_steps * ddp_world_size) / (et-st)
  # if i % (max_iter*0.1) == 0 or i == max_iter-1:
  if master_process:
    print(f"step: {i} | loss: {loss_accum.item():.4f} | lr {lr:.4e} | norm {norm:.4f} | time: {(et-st)*1000:.2f}ms | tok-sec: {tok_sec:.2f}")


if ddp:
  destroy_process_group()