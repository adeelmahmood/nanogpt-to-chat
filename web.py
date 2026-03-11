import os
import glob

import streamlit as st
import torch
import tiktoken

from chat import get_special_tokens
from engine import Engine, Sampler
from gpt import GPTModel, get_gpt_config
from utils import load_checkpoint

st.set_page_config(page_title="nano-chat", layout="wide")
st.title("nano-chat")

RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")


# --- Helpers ---

@st.cache_resource(show_spinner=False)
def _load_model(path, depth):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    model = GPTModel(get_gpt_config(depth))
    model.to(device)
    load_checkpoint(path=path, model=model, device=device, optimizer=None)
    model.eval()
    return model, device


def scan_checkpoints():
    """Return list of .pt files under runs/, excluding .rng.pt files."""
    all_pts = glob.glob(os.path.join(RUNS_DIR, "**", "*.pt"), recursive=True)
    return sorted(p for p in all_pts if not p.endswith(".rng.pt"))


def infer_depth(path):
    name = os.path.basename(path)
    if "d12" in name:
        return "d12"
    return "d20"


def short_label(path):
    """Show path relative to runs dir for readability."""
    try:
        return os.path.relpath(path, RUNS_DIR)
    except ValueError:
        return path


@st.dialog("Conversation history", width="large")
def show_history_dialog(slot, label):
    turns = st.session_state.turns
    if not turns:
        st.info("No turns yet.")
        return
    tokenizer = tiktoken.get_encoding("gpt2")
    for turn in turns:
        st.chat_message("user").write(turn["user"])
        st.chat_message("assistant").write(turn["responses"][slot])


def build_tokens(turns, slot, new_user_text, tokenizer, special):
    ids = [special.bos]
    for turn in turns:
        ids += [special.user_start, *tokenizer.encode(turn["user"]), special.user_end]
        ids += [special.assistant_start, *tokenizer.encode(turn["responses"][slot]), special.assistant_end]
    ids += [special.user_start, *tokenizer.encode(new_user_text), special.user_end]
    ids.append(special.assistant_start)
    return ids


def generate(model, device, token_ids, sampler, max_tokens, special, placeholder):
    engine = Engine(model, sampler, use_kv_cache=True)
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    text = ""
    for token_id, token_text in engine.stream(idx, max_new_tokens=max_tokens, stop_token_id=special.assistant_end):
        if token_id == special.assistant_end:
            break
        text += token_text
        placeholder.markdown(text)
    return text


# --- Session state ---
if "turns" not in st.session_state:
    st.session_state.turns = []
if "loaded_models" not in st.session_state:
    st.session_state.loaded_models = []  # list of (model, device, label)


# --- Sidebar ---
checkpoints = scan_checkpoints()
ckpt_labels = [short_label(p) for p in checkpoints]
ckpt_by_label = {short_label(p): p for p in checkpoints}

with st.sidebar:
    st.header("Models")
    num_models = st.selectbox("Number of models", [1, 2, 3], index=1)

    selections = []
    for i in range(num_models):
        st.subheader(f"Model {chr(65 + i)}")
        if checkpoints:
            label = st.selectbox("Checkpoint", ckpt_labels, key=f"ckpt_{i}")
            path = ckpt_by_label[label]
            auto_depth = infer_depth(path)
        else:
            st.warning("No checkpoints found in runs/")
            label, path, auto_depth = None, None, "d20"

        depth_idx = 0 if auto_depth == "d12" else 1
        depth = st.selectbox("Depth", ["d12", "d20"], index=depth_idx, key=f"depth_{i}")
        selections.append((path, depth, label))

    if st.button("Load models", type="primary", disabled=not checkpoints):
        loaded = []
        for i, (path, depth, label) in enumerate(selections):
            if path:
                try:
                    with st.spinner(f"Loading Model {chr(65 + i)}..."):
                        model, device = _load_model(path, depth)
                    loaded.append((model, device, short_label(path)))
                except Exception as e:
                    st.error(f"Model {chr(65 + i)} failed: {e}")
                    break
        if len(loaded) == num_models:
            st.session_state.loaded_models = loaded
            st.session_state.turns = []
            st.success(f"{num_models} model(s) loaded")

    st.divider()
    st.header("Sampling")
    max_new_tokens = st.slider("Max new tokens", 20, 200, 100)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7, step=0.05)
    top_k = st.slider("Top-k", 1, 100, 50)
    st.divider()

    if st.button("Clear chat"):
        st.session_state.turns = []
        st.rerun()


# --- Main area ---
loaded_models = st.session_state.loaded_models
n = len(loaded_models)
ready = n > 0

if ready:
    cols = st.columns(n)
    for i, (_, _, label) in enumerate(loaded_models):
        model_label = f"Model {chr(65 + i)}"
        hdr, btn = cols[i].columns([4, 1])
        hdr.markdown(f"**{model_label}** — `{label}`")
        if btn.button("History", key=f"hist_{i}", disabled=not st.session_state.turns):
            show_history_dialog(i, model_label)

for turn in st.session_state.turns:
    st.chat_message("user").write(turn["user"])
    cols = st.columns(n)
    for i in range(n):
        with cols[i]:
            st.chat_message("assistant").write(turn["responses"][i])

user_input = st.chat_input("Type a message...", disabled=not ready)

if user_input:
    tokenizer = tiktoken.get_encoding("gpt2")
    special = get_special_tokens()
    sampler = Sampler(temperature=temperature, top_k=top_k)

    st.chat_message("user").write(user_input)

    cols = st.columns(n)
    placeholders = []
    for i in range(n):
        with cols[i]:
            with st.chat_message("assistant"):
                placeholders.append(st.empty())

    responses = []
    for i, (model, device, _) in enumerate(loaded_models):
        token_ids = build_tokens(st.session_state.turns, i, user_input, tokenizer, special)
        resp = generate(model, device, token_ids, sampler, max_new_tokens, special, placeholders[i])
        responses.append(resp)

    st.session_state.turns.append({"user": user_input, "responses": responses})
