"""
Interactive chat CLI using GPTModel + Engine + Sampler

Run:
python chat_cli.py --model_file path/to/checkpoint.pt
"""

import argparse
import torch
import tiktoken

from chat import decode_with_special_tokens, get_special_tokens
from engine import Engine, Sampler
from gpt import GPTConfig, GPTModel
from utils import load_checkpoint


# ANSI color codes for terminal
class Colors:
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with the model")
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser.parse_args()


def autodetect_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()
    device = autodetect_device()

    tokenizer = tiktoken.get_encoding("gpt2")
    special = get_special_tokens()

    # Load model
    model = GPTModel(GPTConfig(vocab_size=50304))
    model.to(device)
    load_checkpoint(
        path=args.model_file,
        model=model,
        device=device,
        optimizer=None,
    )
    model.eval()

    print(f"{Colors.GREEN}{Colors.BOLD}Model loaded on {device}{Colors.END}")

    sampler = Sampler(
        temperature=args.temperature,
        top_k=args.top_k,
    )
    engine = Engine(
        model,
        sampler,
        use_kv_cache=True,
    )

    print("\nInteractive Chat Mode")
    print("-" * 50)
    print("Type 'quit' or 'exit' to end")
    print("Type 'clear' to reset conversation")
    print("-" * 50)

    # Initialize conversation with BOS
    conversation_tokens = [special.bos]

    while True:
        try:
            user_input = input(f"\n{Colors.BLUE}User:{Colors.END} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            conversation_tokens = [special.bos]
            print("Conversation cleared.")
            continue

        # Append user message
        conversation_tokens.append(special.user_start)
        conversation_tokens.extend(tokenizer.encode(user_input))
        conversation_tokens.append(special.user_end)

        # Append assistant start
        conversation_tokens.append(special.assistant_start)

        # Convert to tensor
        idx = torch.tensor(
            [conversation_tokens],
            dtype=torch.long,
            device=device,
        )

        print(f"\n{Colors.CYAN}Assistant:{Colors.END} ", end="", flush=True)

        # Generate
        token_ids, _ = engine.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            stop_token_id=special.assistant_end,
        )

        # Extract newly generated tokens
        generated = token_ids[0, len(conversation_tokens) :].tolist()

        # Ensure assistant_end exists
        if not generated or generated[-1] != special.assistant_end:
            generated.append(special.assistant_end)

        # Decode + stream-print
        for tok in generated:
            text = decode_with_special_tokens([tok], tokenizer)
            print(text, end="", flush=True)

        print()

        # Update conversation state
        conversation_tokens.extend(generated)


if __name__ == "__main__":
    main()
