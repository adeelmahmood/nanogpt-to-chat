import os
import multiprocessing as mp
import tiktoken
from datasets import load_dataset
import numpy as np

# tokenizer init must be top-level
tokenizer = tiktoken.get_encoding("gpt2")
eot = tokenizer.eot_token

def tokenize(doc):
    # tokens = [eot]
    tokens = tokenizer.encode_ordinary(doc["text"])
    tokens_np = np.array(tokens, dtype=np.uint16)
    return tokens_np


def main():

    with open("/Users/adeelqureshi/Downloads/input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenize({"text": text})

    # split 90 - 10 train val
    n = int(0.9 * len(tokens))
    train_tokens = tokens[:n]
    val_tokens = tokens[n:] 

    # save tokens to a numpy file
    np.save("train_000000.npy", train_tokens)
    np.save("val_000000.npy", val_tokens)  

    # download dataset inside main
if __name__ == "__main__":
    mp.freeze_support()  # optional on macOS, but safe
    main()
