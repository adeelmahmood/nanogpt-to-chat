from tasks import SmolTalkTask, TaskMixture
import torch
import tiktoken

from chat import (
    decode_with_special_tokens,
    get_special_tokens,
    render_conversation,
    render_conversation,
)


def sft_loader(
    model, dataset, batch_size, tokenizer, device, ddp_rank=0, ddp_world_size=1
):
    pad_token_id = get_special_tokens().assistant_end

    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, _ in batch) - 1

        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)

        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, : n - 1] = ids_tensor[:-1]

            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1
            targets[i, : n - 1] = row_targets

        inputs = inputs.to(device)
        targets = targets.to(device)
        return inputs, targets

    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            example = dataset.get_example(i)
            ids, mask = render_conversation(example, tokenizer)

            # Limit sequence length to model's block size (until we can make Rope dynamic)
            if model is not None and len(ids) > model.config.block_size:
                ids = ids[: model.config.block_size]
                mask = mask[: model.config.block_size]

            batch.append((ids, mask))

            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    task = TaskMixture([SmolTalkTask(stop=10, split="train")])
    print(f"len(task) = {len(task)}")
    sft_loader_obj = sft_loader(
        model=None, dataset=task, batch_size=1, tokenizer=tokenizer, device="cpu"
    )
    x, y = next(sft_loader_obj)
    for i, t in zip(x, y):
        for i_c, t_c in zip(i.tolist(), t.tolist()):
            print(decode_with_special_tokens([i_c], tokenizer), t_c)
