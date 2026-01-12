import os
import multiprocessing as mp
import tiktoken
from datasets import load_dataset
import numpy as np

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--ds', type=str, choices=["fw", "ts"], help='Dataset to download: fw (FineWeb Edu) | ts (The Stack)')

args = parser.parse_args()
if args.ds == "fw":
    dataset_name = "HuggingFaceFW/fineweb-edu"
    remote_name = "sample-10BT"
    local_dir = "edu_fineweb10B"
elif args.ds == "ts":
    dataset_name = "roneneldan/TinyStories"
    remote_name = ""
    local_dir = "tinystories"
else:
    raise ValueError("Invalid dataset choice. Use --ds fw or --ds ts")

shard_size = int(1e8) # 100M tokens per shard

# create the dir
os.makedirs(local_dir, exist_ok=True)

# tokenizer init must be top-level
tokenizer = tiktoken.get_encoding("gpt2")
eot = tokenizer.eot_token

def tokenize(doc):
    tokens = [eot]
    tokens.extend(tokenizer.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.uint16)
    return tokens_np

def main():
    # download dataset inside main
    ds = load_dataset(dataset_name, name=remote_name, split="train")

    nproc = max(1, os.cpu_count()//2)
    print(f"num of processes={nproc}")

    with mp.Pool(nproc) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0

        for tokens in pool.imap(tokenize, ds, chunksize=16):

            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)

            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(local_dir, f"{split}_{shard_index:06d}")

                remainder = shard_size - token_count
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                np.save(filename, all_tokens_np)
                print(f"Saved shard {filename} with {shard_size:,} tokens")

                shard_index += 1

                # leftovers
                leftover = len(tokens) - remainder
                all_tokens_np[0:leftover] = tokens[remainder:]
                token_count = leftover


        # final shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(local_dir, f"{split}_{shard_index:06d}")
            np.save(filename, all_tokens_np[:token_count])
            print(f"Saved final shard {filename} with {token_count:,} tokens")

if __name__ == "__main__":
    main()
    print("Download and tokenization complete.")


