from collections import deque
from chat import decode_with_special_tokens, render_conversation
from tasks import TaskMixture
import torch

def midtraining_loader(tokenizer, task, batch_size, seq_len, device, ddp_rank=0, ddp_world_size=1): 
    token_buffer = deque() 
    # mask_buffer = deque() 
    
    needed_tokens = batch_size * seq_len + 1 
    
    # buffers 
    scratch_ids = torch.empty(needed_tokens, dtype=torch.long) 
    # scratch_mask = torch.empty(needed_tokens, dtype=torch.long) 
    
    pointer = ddp_rank 
    task_size = len(task) 
    
    while True: 
        # fill stream 
        while len(token_buffer) < needed_tokens: 
            conv = task.get(pointer) 
            ids, _ = render_conversation(conv, tokenizer) 
            
            token_buffer.extend(ids) 
            # mask_buffer.extend(mask) 
            
            pointer += ddp_world_size 
            if pointer >= task_size: 
                pointer -= task_size 
            
        # pack into blocks 
        for i in range(needed_tokens): 
            scratch_ids[i] = token_buffer.popleft() 
            # scratch_mask[i] = mask_buffer.popleft()
       
        # construct batch 
        x = scratch_ids[:-1].view(batch_size, seq_len) 
        y = scratch_ids[1:].view(batch_size, seq_len) 
        # m = scratch_mask[1:].view(batch_size, seq_len) 
        
        # apply mask
        # y = y.masked_fill(m == 0, -1) 
        
        # if (y != -1).any(): 
        yield ( x.to(device, non_blocking=True), y.to(device, non_blocking=True) )


if __name__ == "__main__":
    import tiktoken
    import torch
    from tasks import SmolTalkTask

    tokenizer = tiktoken.get_encoding("gpt2")
    task = TaskMixture([SmolTalkTask()])

    loader = midtraining_loader(
        tokenizer=tokenizer,
        task=task,
        batch_size=1,
        seq_len=1024,   # small for readability
        device="cpu",
    )

    x, y = next(loader)

    x = x[0]
    y = y[0]

    print("\n=== BATCH VERIFICATION (x -> y) ===")
    for i in range(len(x)):
        x_tok = decode_with_special_tokens([x[i].item()], tokenizer).replace("\n", "\\n")

        if y[i].item() == -1:
            y_tok = "<IGNORED>"
            tag = "-----"
        else:
            y_tok = decode_with_special_tokens([y[i].item()], tokenizer).replace("\n", "\\n")
            tag = "TRAIN"

        print(
            f"{i:3d} | x='{x_tok:>10}' "
            f"-> y='{y_tok:>10}' | {tag}"
        )
