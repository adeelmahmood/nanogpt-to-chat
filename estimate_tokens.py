from chat import render_conversation
from dataloader_midtrain import midtraining_loader
from tasks import SmolTalkTask, TaskMixture
import tiktoken

if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding('gpt2')

    task = TaskMixture([SmolTalkTask()])
    loader = midtraining_loader(tokenizer, task, 16, 1024, 'cpu', 0, 1)

    total_tokens = 0
    idx = 0

    while True:
        x, y, progress = next(loader)
        
        total_tokens += x.size(0) * x.size(1)
        pct_done = 100 * progress
        idx += 1

        if idx % 100 == 0:
            print(f"idx {idx} progress {progress:.2f} token {total_tokens:,}")

        if progress > 0.2:
            break
