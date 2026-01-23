import argparse
from chat import decode_with_special_tokens, get_special_tokens
from engine import Engine, Sampler
from gpt import GPTConfig, GPTModel
import torch
import tiktoken

from utils import load_checkpoint


# ANSI color codes for terminal
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def parse_args():
    parser = argparse.ArgumentParser()

    # model file
    parser.add_argument("--model_file", type=str, required=True)

    return parser.parse_args()


def main():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    tokenizer = tiktoken.get_encoding("gpt2")

    args = parse_args()

    # Load checkpoint
    model = GPTModel(GPTConfig(vocab_size=50304))
    model = model.to(device)
    load_checkpoint(path=args.model_file, model=model, device=device, optimizer=None)

    model.eval()

    print(f"{Colors.GREEN}{Colors.BOLD}Model loaded for inference{Colors.END}")

    special = get_special_tokens()

    # Configuration for sampling
    prompts = [
        "Why is sky blue?",
        "what is 2+2?",
        "What is the capital of France?",
        # "Why is sky blue?",
        # "Explain how photosynthesis works.",
        # "Write a short poem about winter.",
        # "What are the benefits of exercise?",
    ]

    max_new_tokens = 50
    temperature = 0.7
    top_k = 50

    print(
        f"{Colors.CYAN}{Colors.BOLD}Generating samples for {len(prompts)} prompts{Colors.END}"
    )

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
        token_ids, state = engine.generate(
            idx, max_new_tokens=max_new_tokens, stop_token_id=special.assistant_end
        )

        token_ids_list = token_ids.squeeze(0).tolist()

        # check if assistant start is present in the generated tokens
        if special.assistant_start in token_ids_list:
            # start from here
            start_idx = token_ids_list.index(special.assistant_start)
            token_ids_list = token_ids_list[start_idx + 1 :]

        # Decode and print result
        decoded_text = decode_with_special_tokens(token_ids_list, tokenizer)

        print(
            f"\n{Colors.BLUE}{Colors.BOLD}Prompt {i + 1}:{Colors.END} {Colors.HEADER}{text}{Colors.END}"
        )
        print(decoded_text)

    print(f"{Colors.GREEN}{Colors.BOLD}All samples generated{Colors.END}")


if __name__ == "__main__":
    main()
