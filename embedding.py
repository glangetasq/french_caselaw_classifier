from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer



_DEFAULT_BATCH_SIZE: dict[str, int] = {
    "mps":  128,
    "cuda": 256,
    "cpu":  32,
}


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return tokenizer, model


def _truncate_head_tail(text: str, tokenizer, n_head: int = 63, n_tail: int = 63) -> str:
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= n_head + n_tail:
        return text
    tokens = tokens[:n_head] + tokens[-n_tail:]
    return tokenizer.convert_tokens_to_string(tokens)


def embed(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int | None = None,
    max_length: int = 128,
    head_tail: bool = False,
    progress: bool = True,
) -> np.ndarray:
    if head_tail:
        texts = [_truncate_head_tail(t, tokenizer) for t in texts]
    if batch_size is None:
        batch_size = _DEFAULT_BATCH_SIZE.get(device.type, 32)
    batches = range(0, len(texts), batch_size)
    if progress:
        batches = tqdm(batches, desc="embedding", unit="batch")
    out = []
    for i in batches:
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            cls = model(**enc).last_hidden_state[:, 0, :].cpu().numpy()
        out.append(cls)
    return np.vstack(out)


def save_embeddings(embeddings: np.ndarray, index: pd.Index, path: Path) -> None:
    """Save embeddings alongside the row index so they can be realigned on reload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, embeddings=embeddings, index=index.values)


def load_embeddings(path: Path | str, index: pd.Index) -> np.ndarray:
    """Load embeddings and reorder rows to match the given index."""
    data = np.load(Path(path), allow_pickle=True)
    saved_index = data["index"]
    embeddings  = data["embeddings"]
    pos   = {idx: i for i, idx in enumerate(saved_index)}
    order = [pos[i] for i in index]
    return embeddings[order]
