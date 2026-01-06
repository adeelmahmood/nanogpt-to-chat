from dataclasses import dataclass


@dataclass
class SpecialTokens:
    bos: int
    user_start: int
    user_end: int
    assistant_start: int
    assistant_end: int
    python_start: int
    python_end: int
    output_start: int
    output_end: int


def get_special_tokens() -> SpecialTokens:
    return SpecialTokens(
        bos=50256,
        user_start=50257,
        user_end=50258,
        assistant_start=50259,
        assistant_end=50260,
        python_start=50261,
        python_end=50262,
        output_start=50263,
        output_end=50264,
    )


def decode_with_special_tokens(token_ids, tokenizer):
    """
    Decode token IDs, replacing special tokens with readable strings.
    """
    special = get_special_tokens()
    
    # Reverse mapping: ID -> string
    special_map = {
        special.bos: "<|bos|>",
        special.user_start: "<|user_start|>",
        special.user_end: "<|user_end|>",
        special.assistant_start: "<|assistant_start|>",
        special.assistant_end: "<|assistant_end|>",
        special.python_start: "<|python_start|>",
        special.python_end: "<|python_end|>",
        special.output_start: "<|output_start|>",
        special.output_end: "<|output_end|>",
    }
    
    result = []
    for token_id in token_ids:
        if token_id in special_map:
            # It's a special token
            result.append(special_map[token_id])
        else:
            # Regular token - decode it
            result.append(tokenizer.decode([token_id]))
    
    return "".join(result)


def render_conversation(conversation, tokenizer, max_tokens=None):
    special = get_special_tokens()
    ids, masks = [], []

    def add(token_ids, mask_val):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        ids.extend(token_ids)
        masks.extend([mask_val] * len(token_ids))

    # start with BOS (dont train on it)
    add(special.bos, 0)

    for i, msg in enumerate(conversation["messages"]):
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            # user messages, dont train on it
            add(special.user_start, 0)
            add(tokenizer.encode(content), 0)
            add(special.user_end, 0)

        elif role == "assistant":
            # assistant messages, TRAIN on them
            add(special.assistant_start, 0)
            add(tokenizer.encode(content), 1)
            add(special.assistant_end, 1)

    if max_tokens is not None:
        ids = ids[:max_tokens]
        masks = masks[:max_tokens]

    return ids, masks
            

if __name__ == "__main__":
    import tiktoken
    from tasks import SmolTalkTask

    tokenizer = tiktoken.get_encoding("gpt2")
    task = SmolTalkTask()

    # grab ONE conversation
    conv = task.get_example(0)

    ids, mask = render_conversation(conv, tokenizer)

    print("\n=== RAW CONVERSATION ===")
    for msg in conv["messages"]:
        print(f"{msg['role'].upper()}: {msg['content']}")

    print("\n=== TOKENS + MASK ===")
    for i, (tid, m) in enumerate(zip(ids, mask)):
        token_str = decode_with_special_tokens([tid], tokenizer)
        token_str = token_str.replace("\n", "\\n")

        tag = "TRAIN" if m == 1 else "-----"
        print(f"{i:4d} | id={tid:6d} | mask={m} | {tag} | '{token_str}'")
