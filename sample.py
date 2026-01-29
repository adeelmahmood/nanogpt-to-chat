import argparse
import torch
import tiktoken

from chat import decode_with_special_tokens, get_special_tokens
from engine import Engine, Sampler
from gpt import GPTModel, get_gpt_config
from utils import load_checkpoint


# ANSI color codes for terminal
class Colors:
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument(
        "--model_depth",
        type=str,
        choices=["d12", "d20", "d2"],
        default="d12",
        help="Model depth configuration",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming token generation instead of generate()",
    )
    return parser.parse_args()


def main():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    tokenizer = tiktoken.get_encoding("gpt2")
    special = get_special_tokens()

    args = parse_args()

    # Load model
    model = GPTModel(get_gpt_config(args.model_depth))
    model.to(device)
    load_checkpoint(
        path=args.model_file,
        model=model,
        device=device,
        optimizer=None,
    )
    model.eval()

    print(f"{Colors.GREEN}{Colors.BOLD}Model loaded for inference{Colors.END}")

    # Sampling config
    prompts = [
        "Who are you?",
        "Why is the sky blue?",
        "What is 2 + 2?",
        "Which city is the capital of France?",
        "Is ice cold or hot?",
        "Write a poem about cats",
        # "Write a story about a cat who loved to eat",
    ]

    max_new_tokens = 50
    temperature = 0.7
    top_k = 50

    print(f"{Colors.CYAN}{Colors.BOLD}Generating chat-formatted samples{Colors.END}")

    for i, text in enumerate(prompts):
        torch.manual_seed(1337 + i)
        if device == "cuda":
            torch.cuda.manual_seed(1337 + i)

        sampler = Sampler(temperature=temperature, top_k=top_k)
        engine = Engine(model, sampler, use_kv_cache=True)

        # ---- CHAT FORMATTING ----
        conversation = [
            special.bos,
            special.user_start,
            *tokenizer.encode(text),
            special.user_end,
            special.assistant_start,
        ]

        idx = torch.tensor([conversation], dtype=torch.long, device=device)
        print(f"\n{Colors.BLUE}{Colors.BOLD}Prompt {i + 1}:{Colors.END} {text}")

        if args.stream:
            # -------- STREAMING PATH --------
            print(
                f"{Colors.CYAN}{Colors.BOLD}Assistant:{Colors.END} ", end="", flush=True
            )

            generated = []
            for token_id, token_text in engine.stream(
                idx,
                max_new_tokens=max_new_tokens,
                stop_token_id=special.assistant_end,
            ):
                generated.append(token_id)
                print(token_text, end="", flush=True)

            # ensure assistant_end exists for consistency
            # if not generated or generated[-1] != special.assistant_end:
            #     generated.append(special.assistant_end)

            print()  # newline after streaming

        else:
            # -------- NON-STREAMING PATH --------
            token_ids, _ = engine.generate(
                idx,
                max_new_tokens=max_new_tokens,
                stop_token_id=special.assistant_end,
            )

            generated = token_ids[0, len(conversation) :].tolist()

            # if not generated or generated[-1] != special.assistant_end:
            #     generated.append(special.assistant_end)

            decoded = decode_with_special_tokens(generated, tokenizer)

            print(decoded)

    print(f"\n{Colors.GREEN}{Colors.BOLD}All samples generated{Colors.END}")


if __name__ == "__main__":
    main()
