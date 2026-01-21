from chat import render_conversation
from tasks import GSM8K, MMLU, Arc, SmolTalkTask, TaskMixture
from transformers import AutoTokenizer
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("gpt2")


def count_tokens(task, max_samples=None):
    total = 0

    for i in range(len(task)):
        ex = task.get_example(i)
        ids, _ = render_conversation(ex, tokenizer=tokenizer)
        total += len(ids)

    return total


train_task = TaskMixture(
    [
        SmolTalkTask(stop=10_000),
        MMLU(stop=2_000),
        GSM8K(stop=2_000),
        Arc(stop=2_000),
    ]
)
print(f"Train task token estimate: {count_tokens(train_task):,}")
