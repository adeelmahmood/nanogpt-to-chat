from dataclasses import dataclass
from chat import decode_with_special_tokens
from engine import Engine, Sampler
from gpt import GPTConfig, GPTModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
tokenizer = tiktoken.get_encoding("gpt2")

# Load checkpoint
checkpoint = torch.load("./ckps/localmodel/midtrain_model_01999.pt", map_location=device, weights_only=False)

# --- FIX: strip `_orig_mod.` keys if present ---
state = checkpoint["model_state_dict"]
fixed_state = {}

for k, v in state.items():
    if k.startswith("_orig_mod."):
        fixed_state[k.replace("_orig_mod.", "", 1)] = v
    else:
        fixed_state[k] = v

# Initialize model
model = GPTModel(GPTConfig(vocab_size=50304))
model.load_state_dict(fixed_state, strict=True)
model.to(device)
model.eval()

print("Model loaded for inference")

torch.manual_seed(1)
torch.cuda.manual_seed(1)


# generate
sampler = Sampler(temperature=1.0, top_k=50)
engine = Engine(model, sampler, use_kv_cache=True)

text = "Who is Max?"
idx = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)

token_ids, state = engine.generate(idx, max_new_tokens=50)
print(decode_with_special_tokens(token_ids.squeeze(0).tolist(), tokenizer))

# print(tokenizer.decode(token_ids))



# step: 19068 | loss: 3.0779 | lr 6.0000e-05 | norm 0.2802 | time: 358.64ms | tok-sec: 1461880.33
# step: 19069 | loss: 3.0880 | lr 6.0000e-05 | norm 0.2880 | time: 359.01ms | tok-sec: 1460383.29
# step: 19070 | loss: 3.0448 | lr 6.0000e-05 | norm 0.2921 | time: 359.05ms | tok-sec: 1460215.53
# step: 19071 | loss: 3.0554 | lr 6.0000e-05 | norm 0.2876 | time: 358.64ms | tok-sec: 1461882.28
# Validation loss: 3.0655
# Checkpoint saved at ./ckps/fw_model_19072.pt
# Checkpoint saved at step 19072
# step: 19072 | loss: 3.0833 | lr 6.0000e-05 | norm 0.3028 | time: 3540.07ms | tok-sec: 148100.82

