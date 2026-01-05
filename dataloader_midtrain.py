from collections import deque
from chat import render_conversation
import torch

def midtraining_loader(tokenizer, task, batch_size, seq_len, device, ddp_rank=0, ddp_world_size=1):
    token_buffer = deque()
    mask_buffer = deque()
    
    needed_tokens = batch_size * seq_len + 1

    # buffers
    scratch_ids = torch.empty(needed_tokens, dtype=torch.long)
    scratch_mask = torch.empty(needed_tokens, dtype=torch.long)

    pointer = ddp_rank
    task_size = len(task)
    
    while True:
        # fill stream
        while len(token_buffer) < needed_tokens:
            conv = task.get(pointer)
            ids, mask = render_conversation(conv, tokenizer)

            token_buffer.extend(ids)
            mask_buffer.extend(mask)

            pointer += ddp_world_size
            if pointer >= task_size:
                pointer -= task_size

        
        # pack into blocks
        for i in range(needed_tokens):
            scratch_ids[i] = token_buffer.popleft()
            scratch_mask[i] = mask_buffer.popleft()


        # construct batch
        x = scratch_ids[:-1].view(batch_size, seq_len)
        y = scratch_ids[1:].view(batch_size, seq_len)
        m = scratch_mask[1:].view(batch_size, seq_len)

        # apply mask
        y = torch.where(m == 1, torch.full_like(y, -1), y)

        yield (
            x.to(device, non_blocking=True),
            y.to(device, non_blocking=True)
        )