from dataloader import DataLoaderLite
import torch
import tiktoken
import numpy as np
import os


if __name__ == "__main__":
    ds = DataLoaderLite(
        B=5,
        T=10,
        process_rank=0,
        num_processes=1,
        split="train",
        data_root="tinysk",
        master_process=True,
    )
    for _ in range(5):
        xb, yb = ds.next_batch()
        print(xb.shape, yb.shape)
