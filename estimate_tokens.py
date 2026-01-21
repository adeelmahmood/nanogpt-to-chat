from tasks import MMLU, Arc, SmolTalkTask
from transformers import AutoTokenizer
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("gpt2")


def count_tokens(task, max_samples=None):
    total = 0
    n = len(task)
    if max_samples:
        n = min(n, max_samples)

    for i in tqdm(range(n)):
        ex = task.get_example(i)
        text = ""
        for m in ex["messages"]:
            text += m["role"] + ": " + m["content"] + "\n"
        total += len(tokenizer(text).input_ids)

    return total, total / n


# Example usage
smol = SmolTalkTask()
mmlu = MMLU()
arc = Arc()

print("SmolTalk:", count_tokens(smol, max_samples=1000))
print("MMLU:", count_tokens(mmlu, max_samples=1000))
print("ARC:", count_tokens(arc, max_samples=1000))
