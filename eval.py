import argparse
import re
from chat import decode_with_special_tokens, get_special_tokens, render_conversation
from engine import Engine, Sampler
from gpt import GPTConfig, GPTModel
from tasks import GSM8K, MMLU, Arc, TaskMixture
import torch

from utils import load_checkpoint


class EvalResult:
    def __init__(self, correct: bool, pred: str, gold: str, mixture: str):
        self.correct = correct
        self.pred = pred
        self.gold = gold
        self.mixture = mixture


class EvalTask:
    def build_prompt(self, example):
        """Return conversation WITHOUT assistant answer"""

    def score(self, generated_text: str, example) -> EvalResult:
        """Scores the generated text"""


class MMLUEval(MMLU):
    def build_prompt(self, example):
        return {"messages": [example["messages"][0]]}

    def score(self, generated_text: str, example) -> EvalResult:
        gold = example["messages"][1]["content"].strip()

        # extract A-D
        match = re.search(r"([A-D])+", generated_text)
        pred = match.group(1) if match else ""

        return EvalResult(
            correct=(pred == gold),
            pred=pred,
            gold=gold,
            mixture=f"{self.__class__.__name__}",
        )


class ARCEval(Arc):
    def build_prompt(self, example):
        return {"messages": [example["messages"][0]]}

    def score(self, generated_text: str, example):
        gold = example["messages"][1]["content"].strip()
        match = re.search(r"([A-Z])+", generated_text)
        pred = match.group(1) if match else ""

        return EvalResult(
            correct=(pred == gold),
            pred=pred,
            gold=gold,
            mixture=f"{self.__class__.__name__}",
        )


class GSM8KEval(GSM8K):
    def build_prompt(self, example):
        # only the user message
        return {"messages": [example["messages"][0]]}

    def score(self, generated_text: str, example):
        gold = example["messages"][1]["content"].strip()

        # extract last number from model output
        nums = re.findall(r"-?\d+\.?\d*", generated_text)
        pred = nums[-1] if nums else ""

        return EvalResult(
            correct=(pred == gold),
            pred=pred,
            gold=gold,
            mixture=f"{self.__class__.__name__}",
        )


class EvalRunner:
    def __init__(self, model, tokenizer, max_new_tokens=100):
        self.model = model
        self.tokenizer = tokenizer
        self.engine = Engine(
            model, sampler=Sampler(temperature=0.0, top_k=None), use_kv_cache=True
        )
        self.max_new_tokens = max_new_tokens

    def run_task(self, task, limit=None):
        total = 0
        correct = 0
        results = []

        special = get_special_tokens()

        n = len(task) if limit is None else limit
        for i in range(n):
            torch.manual_seed(1337 + i)
            if device == "cuda":
                torch.cuda.manual_seed(1337 + i)

            # handle Task or TaskMixture
            if hasattr(task, "tasks") and hasattr(task, "indices"):
                # TaskMixture: indices holds tuples (task_idx, example_idx)
                ti, example_idx = task.indices[i]
                eval_task = task.tasks[ti]
                example = eval_task[example_idx]
            else:
                # Regular Task
                eval_task = task
                example = task[i]

            # build prompt
            prompt = eval_task.build_prompt(example)
            ids, masks = render_conversation(prompt, tokenizer=self.tokenizer)
            # add assistant start token
            ids.append(special.assistant_start)
            masks.append(0)
            idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            prompt_len = idx.shape[1]

            # generate
            out_ids, _ = self.engine.generate(
                idx,
                max_new_tokens=self.max_new_tokens,
                stop_token_id=special.assistant_end,
            )

            # decode only the assistant text
            gen_ids = out_ids[0, prompt_len:].tolist()
            gen_text = decode_with_special_tokens(gen_ids, tokenizer=self.tokenizer)

            # score
            result = eval_task.score(gen_text, example)
            results.append(result)

            total += 1
            correct += int(result.correct)

            if i % 10 == 0:
                print(f"Step {i+1}/{n} - Acc so far: {correct/total:.4f}")

        return {
            "accuracy": correct / total,
            "total": total,
            "correct": correct,
            "results": results,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    import tiktoken

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    args = parse_args()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    tokenizer = tiktoken.get_encoding("gpt2")

    model = GPTModel(GPTConfig(vocab_size=50304))
    model.to(device)
    load_checkpoint(
        path=args.model_file,
        model=model,
        device=device,
        optimizer=None,
    )
    model.eval()

    task = TaskMixture(
        [
            ARCEval(subset="ARC-Easy", stop=25),
            ARCEval(subset="ARC-Challenge", stop=25),
            MMLUEval(stop=50),
            GSM8KEval(stop=50),
        ]
    )
    runner = EvalRunner(model=model, tokenizer=tokenizer)
    results = runner.run_task(task)
    print(f'Accuracy over {results["total"]} examples: {results["accuracy"]:.4f}')
    # compute accuracy per mixture
    per_mixture = {}
    for res in results["results"]:
        if res.mixture not in per_mixture:
            per_mixture[res.mixture] = {"total": 0, "correct": 0}
        per_mixture[res.mixture]["total"] += 1
        per_mixture[res.mixture]["correct"] += int(res.correct)

    for mix, stats in per_mixture.items():
        acc = stats["correct"] / stats["total"]
        print(f"  {mix}: {acc}. Correct {stats['correct']}/{stats['total']}")
