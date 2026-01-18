from dataclasses import dataclass
from chat import decode_with_special_tokens, get_special_tokens
from engine import Engine, Sampler
from gpt import GPTConfig, GPTModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken

# ANSI color codes for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
tokenizer = tiktoken.get_encoding("gpt2")

# Load checkpoint
checkpoint = torch.load("./ckps/midtrain_model_00099.pt", map_location=device, weights_only=False)

# Strip `_orig_mod.` prefix from state dict keys if present
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

print(f"{Colors.GREEN}{Colors.BOLD}Model loaded for inference{Colors.END}")

special = get_special_tokens()

# Configuration for sampling
prompts = [
    "Why is sky blue?",
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "Write a short poem about winter.",
    "What are the benefits of exercise?"
]

max_new_tokens = 200
temperature = 1.0
top_k = 50

print(f"\n{Colors.CYAN}{Colors.BOLD}Generating samples for {len(prompts)} prompts{Colors.END}\n")
print(f"{Colors.YELLOW}{'=' * 80}{Colors.END}")

# Generate sample for each prompt
for i, text in enumerate(prompts):
    # Set seed for reproducibility
    torch.manual_seed(1337 + i)
    torch.cuda.manual_seed(1337 + i)
    
    # Create sampler and engine
    sampler = Sampler(temperature=temperature, top_k=top_k)
    engine = Engine(model, sampler, use_kv_cache=True)
    
    # Encode prompt
    idx = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)
    
    # Generate tokens
    token_ids, state = engine.generate(idx, max_new_tokens=max_new_tokens, stop_token_id=special.assistant_end)
    
    # Decode and print result
    decoded_text = decode_with_special_tokens(token_ids.squeeze(0).tolist(), tokenizer)
    
    print(f"\n{Colors.BLUE}{Colors.BOLD}Prompt {i + 1}:{Colors.END} {Colors.HEADER}{text}{Colors.END}")
    print(f"{Colors.CYAN}{'-' * 80}{Colors.END}")
    print(decoded_text)
    print(f"{Colors.CYAN}{'-' * 80}{Colors.END}")

print(f"\n{Colors.YELLOW}{'=' * 80}{Colors.END}")
print(f"{Colors.GREEN}{Colors.BOLD}All samples generated{Colors.END}")