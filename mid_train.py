from contextlib import nullcontext
from gpt import GPTConfig, GPTModel, configure_optimizer
from logger import MetricLogger
from dataloader_midtrain import midtraining_loader
from tasks import SmolTalkTask, TaskMixture
import torch
import time

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import tiktoken

import os

from utils import load_from_checkpoint, sample_from_model, save_checkpoint


# set up logger
logger = MetricLogger("runs", "midtraining")

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
  

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
if master_process:
  print(f"using device: {device} type {device_type}")

# set seeds
torch.manual_seed(1337 + ddp_rank)
torch.cuda.manual_seed(1337 + ddp_rank)

# define the tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

torch.set_float32_matmul_precision('high')

# initialize the model
model = GPTModel(GPTConfig(vocab_size=50304))
model = model.to(device)
# initiatlize the optimizer
optimizer = configure_optimizer(model, lr=1e-4)

# load pretrained state for model and optimizer
model, optimizer, _, _ = load_from_checkpoint(model, "./ckps/localmodel/pretrain_model_00487.pt", device, optimizer)
raw_model = model
# model = torch.compile(model)

print("Loaded pretrained model. Sampling...")
sample_from_model(model, tokenizer, device, "Sam", max_tokens=100)

# wrap the model in ddp
if ddp:
  model = DDP(model, device_ids=[ddp_local_rank])

if master_process:
  print(f"Model parameters: {sum(p.nelement() for p in model.parameters())/1e6:.2f}M")

# Training Loop
ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16) # if use_bf16 else nullcontext()

# Batch parameters
B = 4
T = 1024

# Midtraining dataset
task = TaskMixture([
    SmolTalkTask(),
])

train_loader = midtraining_loader(
    tokenizer,
    task,
    batch_size=B,
    seq_len=T,
    device=device,
    ddp_rank=ddp_rank,
    ddp_world_size=ddp_world_size
)

max_steps = 50 # 5_000
print(f"Starting midtraining for {max_steps} steps with B={B} and T={T}")

model.train()
for i in range(max_steps):
  st = time.time()
  last_step = (i == max_steps - 1)
  
  # save checkpoint
  if last_step and master_process:
    save_checkpoint(
        f"./ckps/localmodel/midtrain_model_{i:05d}.pt",
        raw_model,
        optimizer,
        step=i,
        config=raw_model.config
    )
    print(f"Checkpoint saved at step {i}")


  optimizer.zero_grad()
  
  # get a batch
  x, y = next(train_loader)
  
  # forward pass
  with ctx:
    _, loss = model(x, y)

  loss_accum = loss.detach()

  # backward pass
  loss.backward()

  # gather loss from all ranks
  if ddp:
    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

  # clip gradients
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

  # update
  optimizer.step()

  # synchronize to time
  if 'cuda' in device:
    torch.cuda.synchronize()

  et = time.time()
  tok_sec = (B * T * ddp_world_size) / (et-st)
  # if i % (max_iter*0.1) == 0 or i == max_iter-1:
  if master_process:
    print(f"step: {i} | loss: {loss_accum.item():.4f} | norm {norm:.4f} | time: {(et-st)*1000:.2f}ms | tok-sec: {tok_sec:.2f}")
    logger.log(
        step=i,
        train_loss=loss_accum.item(),
        grad_norm=norm.item(),
        tok_per_sec=tok_sec
    )


model.eval()
print("Finished mitraining. Sampling...")
sample_from_model(model.module if ddp else model, tokenizer, device, "Sam", max_tokens=100)

if ddp:
  destroy_process_group()

