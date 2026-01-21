import random
from datasets import load_dataset

from utils import print0, render_mcq


class Task:
    def __init__(self, start=0, stop=None):
        self.start = start
        self.stop = stop

    def num_examples(self):
        raise NotImplementedError()

    def get_example(self, idx):
        raise NotImplementedError()

    def __len__(self):
        start = self.start
        stop = self.stop if self.stop is not None else self.num_examples()
        return max(0, stop - start)

    def __getitem__(self, idx):
        physical_idx = self.start + idx
        return self.get_example(physical_idx)


class TaskMixture(Task):
    def __init__(self, tasks, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.tasks = tasks
        self.indices = []
        self.num_conversations = 0

        for idx, task in enumerate(tasks):
            for i in range(len(task)):
                self.indices.append((idx, i))
                self.num_conversations += 1

        rng = random.Random(seed)
        rng.shuffle(self.indices)

    def num_examples(self):
        return self.num_conversations

    def get_example(self, idx):
        assert (
            0 <= idx < self.num_conversations
        ), f"Index {idx} out of range in {self.num_conversations} conversations"
        ti, i = self.indices[idx]
        return self.tasks[ti][i]


class SmolTalkTask(Task):
    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "split must be train|test"

        print0(f"Loading SmolTalk {split}")
        self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=f"{split}")
        self.ds = self.ds.shuffle(seed=42)
        self.length = len(self.ds)
        print0(f"Loaded {self.length:,} conversations")

    def num_examples(self):
        return self.length

    def get_example(self, idx):
        # print(f"get_example {idx}")
        row = self.ds[idx]
        messages = row["messages"]

        return {"messages": messages}


class MMLU(Task):

    letters = ("A", "B", "C", "D")

    def __init__(self, split="train", subset="auxiliary_train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "split must be train|test"

        print0(f"Loading MMLU {split} {subset}")
        self.ds = load_dataset("cais/mmlu", subset, split=split)
        if subset == "auxiliary_train":
            self.ds = self.ds.map(lambda row: row["train"], remove_columns=["train"])
        self.ds = self.ds.shuffle(seed=42)
        self.length = len(self.ds)
        print0(f"Loaded {self.length:,} questions")

    def num_examples(self):
        return self.length

    def get_example(self, idx):
        row = self.ds[idx]
        question = row["question"]
        choices = row["choices"]
        answer = row["answer"]
        subject = row["subject"]

        user_message = render_mcq(question, self.letters, choices)
        assistant_message = self.letters[answer]

        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]

        conversation = {
            "messages": messages,
            "subject": subject,
            "letters": self.letters,
        }

        return conversation


class Arc(Task):

    def __init__(self, split="train", subset="ARC-Easy", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "split must be train|test"
        assert subset in [
            "ARC-Easy",
            "ARC-Challenge",
        ], "subset must be ARC-Easy|ARC-Challenge"

        print0(f"Loading Arc {split} {subset}")
        self.ds = load_dataset("allenai/ai2_arc", subset, split=split)
        self.ds = self.ds.shuffle(seed=42)
        self.length = len(self.ds)
        print0(f"Loaded {self.length:,} questions")

    def num_examples(self):
        return self.length

    def get_example(self, idx):
        row = self.ds[idx]
        question = row["question"]
        choices = row["choices"]["text"]
        letters = row["choices"]["label"]
        answer = row["answerKey"]

        user_message = render_mcq(question, letters, choices)
        assistant_message = answer

        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]

        conversation = {"messages": messages, "letters": letters}

        return conversation


class GSM8K(Task):
    def __init__(self, split="train", subset="main", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"]

        print0(f"Loading GSM5k {split} {subset}")
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)
        self.length = len(self.ds)
        print0(f"Loaded {self.length:,} questions")

    def num_examples(self):
        return self.length

    def get_example(self, idx):
        row = self.ds[idx]
        question = row["question"]
        answer = row["answer"]

        # Remove the final #### answer marker if you want
        # or keep it — both are fine for midtraining
        answer = answer.strip()

        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

        return {"messages": messages}


class SpellingTask(Task):
    def __init__(self, split="train", size=1000, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "split must be train|test"
        print0(f"Loading SpellingTask {split}")

        with open("download/words_alpha.txt") as f:
            self.words = [line.strip() for line in f]
        self.length = min(len(self.words), size)
        self.words = self.words[: self.length]

        # deterministic shuffle
        rng = random.Random(42)
        rng.shuffle(self.words)

        print0(f"Loaded {self.length:,} words")

    def num_examples(self):
        return self.length

    def get_example(self, idx):
        rng = random.Random(idx)
        word = rng.choice(self.words)
        word_letters = ",".join(word)

        messages = [
            {"role": "user", "content": f"Spell the word: {word}"},
            {"role": "assistant", "content": f"{word}:{word_letters}"},
        ]

        return {"messages": messages}


if __name__ == "__main__":
    spelling = SpellingTask(split="train", size=1000)
    # print 10 examples
    for i in range(10):
        print0(spelling.get_example(i))
