import random
from datasets import load_dataset



from chat import render_conversation

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
        self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split)
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
        
        print(row)


if __name__ == "__main__":
    mmlu = MMLU()
    # smoltalk = SmolTalkTask()
    # tasks = TaskMixture([smoltalk])
    # print(f"All tasks len: {len(tasks)}")
    # print(tasks.get(0))

    # import tiktoken
    # tokenizer = tiktoken.get_encoding('gpt2')
    # print(render_conversation(tasks.get(0), tokenizer))
    mmlu.get_example(0)
