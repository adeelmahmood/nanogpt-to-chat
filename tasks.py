import random
from datasets import load_dataset

from utils import render_mcq

class Task:
    def __init__(self, start=0, stop=None):
        self.start = start
        self.stop = stop

    def __len__(self):
        raise NotImplementedError()
    
    def get_example(self, idx):
        raise NotImplementedError()
    

class TaskMixture(Task):
    def __init__(self, tasks, seed=42):
        self.tasks = tasks
        self.indices = []

        for idx, task in enumerate(tasks):
            for i in range(len(task)):
                self.indices.append((idx, i))

        rng = random.Random(seed)
        rng.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)
    
    def get(self, idx):
        ti, i = self.indices[idx]
        return self.tasks[ti].get_example(i)
    


class SmolTalkTask(Task):
    def __init__(self, start=0, stop=None, split = "train"):
        super().__init__(start, stop)
        assert split in ["train", "test"], "split must be train|test"

        print(f"Loading SmolTalk {split}")
        # self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=f"{split}[:100]")
        self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=f"{split}")
        self.ds = self.ds.shuffle(seed=42)
        self.length = len(self.ds)
        print(f"Loaded {self.length:,} conversations")

    def __len__(self):
        return self.length
    
    def get_example(self, idx):
        # print(f"get_example {idx}")
        row = self.ds[idx]
        messages = row["messages"]

        return { "messages": messages }



class MMLU(Task):

    letters = ('A', 'B', 'C', 'D')

    def __init__(self, start=0, stop=None, split="train", subset="auxiliary_train"):
        super().__init__(start, stop)
        assert split in ["train", "test"], "split must be train|test"

        print(f"Loading MMLU {split} {subset}")
        self.ds = load_dataset("cais/mmlu", subset, split=split)
        if subset == "auxiliary_train":
            self.ds = self.ds.map(lambda row: row['train'], remove_columns=['train'])
        self.ds = self.ds.shuffle(seed=42)
        self.length = len(self.ds)
        print(f"Loaded {self.length:,} questions")

    def __len__(self):
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
            "letters": self.letters
        }

        return conversation


class Arc(Task):

    def __init__(self, start=0, stop=None, split="train", subset="ARC-Easy"):
        super().__init__(start, stop)
        assert split in ["train", "test"], "split must be train|test"
        assert subset in ["ARC-Easy", "ARC-Challenge"], "subset must be ARC-Easy|ARC-Challenge"

        print(f"Loading Arc {split} {subset}")
        self.ds = load_dataset("allenai/ai2_arc", subset, split=split)
        self.ds = self.ds.shuffle(seed=42)
        self.length = len(self.ds)
        print(f"Loaded {self.length:,} questions")

    def __len__(self):
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
        
        conversation = {
            "messages": messages,
            "letters": letters
        }

        return conversation



class GSM8K(Task):
    def __init__(self, split="train", subset="main"):
        super().__init__()
        assert split in ["train", "test"]

        print(f"Loading GSM5k {split} {subset}")
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)
        self.length = len(self.ds)
        print(f"Loaded {self.length:,} questions")

    def __len__(self):
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

    


if __name__ == "__main__":
    task = GSM8K()
    print(task.get_example(0))
