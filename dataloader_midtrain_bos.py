from chat import decode_with_special_tokens, get_special_tokens, render_conversation
from tasks import TaskMixture
import torch


def midtraining_loader_bos(
    tokenizer,
    task,
    batch_size,
    seq_len,
    device,
    ddp_rank=0,
    ddp_world_size=1,
    buffer_size=128,
):
    row_capacity = seq_len + 1
    conv_buffer = []
    pointer = ddp_rank
    task_size = len(task)

    def add_more():
        nonlocal pointer
        while len(conv_buffer) < buffer_size:
            conv = task.get_example(pointer)
            ids, _ = render_conversation(conversation=conv, tokenizer=tokenizer)
            conv_buffer.append(ids)
            pointer = (pointer + ddp_world_size) % task_size

    while True:
        rows = []
        for _ in range(batch_size):
            row = []
            while len(row) < row_capacity:
                add_more()
                remaining = row_capacity - len(row)

                # pick largest conversation
                best_i, best_len = -1, 0
                for i, ids in enumerate(conv_buffer):
                    L = len(ids)
                    if L <= remaining and L > best_len:
                        best_i, best_len = i, L

                if best_i >= 0:
                    ids = conv_buffer.pop(best_i)
                    row.extend(ids)
                else:
                    # nothing fits, crop one conv
                    ids = conv_buffer.pop(0)
                    row.extend(ids[:remaining])

            rows.append(row[:row_capacity])

        batch = torch.tensor(rows, dtype=torch.long)
        x = batch[:, :-1].to(device, non_blocking=True)
        y = batch[:, 1:].to(device, non_blocking=True)
        yield x, y


if __name__ == "__main__":
    import tiktoken
    import torch
    from tasks import SmolTalkTask

    tokenizer = tiktoken.get_encoding("gpt2")
    task = TaskMixture([SmolTalkTask()])

    loader = midtraining_loader_bos(
        tokenizer=tokenizer,
        task=task,
        batch_size=10,
        seq_len=256,  # small for readability
        device="cpu",
        buffer_size=256,
    )

    # counter = 0
    # while True:
    #     x, y = next(loader)

    #     x = x[0]
    #     y = y[0]

    #     print("\n=== BATCH VERIFICATION (x -> y) ===")
    #     for i in range(len(x)):
    #         x_tok = decode_with_special_tokens([x[i].item()], tokenizer).replace("\n", "\\n")

    #         if y[i].item() == -1:
    #             y_tok = "<IGNORED>"
    #             tag = "-----"
    #         else:
    #             y_tok = decode_with_special_tokens([y[i].item()], tokenizer).replace("\n", "\\n")
    #             tag = "TRAIN"

    #         print(
    #             f"{i:3d} | x='{x_tok:>10}' "
    #             f"-> y='{y_tok:>10}' | {tag}"
    #         )

    #     print('-------------')
    #     counter +=1

    #     if counter > 2:
    #         break

    def test_bos_alignment(loader, num_batches=100):
        special = get_special_tokens()
        bos = special.bos

        total_rows = 0
        bos_at_start = 0

        for _ in range(num_batches):
            x, y = next(loader)  # x: (B, T)
            total_rows += x.size(0)
            bos_at_start += (x[:, 0] == bos).sum().item()

        ratio = bos_at_start / total_rows
        print(f"BOS@start ratio: {ratio:.4f} ({bos_at_start}/{total_rows})")

        return ratio

    def test_bos_frequency(loader, num_batches=50):
        special = get_special_tokens()
        bos = special.bos

        counts = []

        for _ in range(num_batches):
            x, _ = next(loader)
            counts.extend((x == bos).sum(dim=1).tolist())

        avg = sum(counts) / len(counts)
        mx = max(counts)

        print(f"Avg BOS per row: {avg:.2f}")
        print(f"Max BOS in a row: {mx}")

        return avg, mx

    test_bos_alignment(loader)
    test_bos_frequency(loader)
