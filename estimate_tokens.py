from chat import render_conversation
from tasks import GSM8K, MMLU, Arc, SmolTalkTask, SpellingTask, TaskMixture
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
        SmolTalkTask(),  # 460k
        MMLU(),  # 100k
        GSM8K(),  # 8k
        SpellingTask(size=200_000),  # 200k
    ]
)
print(f"Train task token estimate: {count_tokens(train_task):,}")

# midtraining
# token estimate: 457,865,025
