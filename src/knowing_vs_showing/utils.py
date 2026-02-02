from __future__ import annotations

import json
import os
import random
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch


def model_short_name(model_name: str) -> str:
    """
    Convert a Hugging Face model id to a short label used in result tables.

    This mapping is intentionally simple and can be extended if needed.
    """
    lower = model_name.lower()
    if "olmo" in lower:
        return "olmo-7b"
    if "llama" in lower:
        return "llama-8b"
    if "gemma" in lower:
        return "gemma-9b"
    return lower.replace("/", "_")


def split_half_indices(n: int, seed: int) -> tuple[list[int], list[int]]:
    """
    Deterministically split indices into two halves using a fixed shuffle seed.
    """
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    half = n // 2
    a = sorted(idx[:half])
    b = sorted(idx[half:])
    return a, b


def split_train_eval_indices(n: int, seed: int, eval_frac: float) -> tuple[list[int], list[int]]:
    """
    Deterministically split indices into train and eval sets using a shuffle seed.
    """
    if n <= 0:
        return [], []
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    eval_n = max(1, int(round(n * eval_frac)))
    eval_idx = sorted(idx[:eval_n])
    train_idx = sorted(idx[eval_n:])
    return train_idx, eval_idx


def chat_text(tokenizer: Any, user_prompt: str, *, add_generation_prompt: bool) -> str:
    """
    Build the model-specific chat-formatted input string.

    If the tokenizer does not support a chat template, the raw user prompt is returned.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            return user_prompt
    return user_prompt


def tokenize_text(tokenizer: Any, text: str) -> dict[str, torch.Tensor]:
    """
    Tokenize a pre-formatted text into tensors.
    """
    return tokenizer(text, return_tensors="pt")


def to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_dataclass_json(path: Path, obj: Any) -> None:
    write_json(path, asdict(obj))


def try_git_commit_push(paths: Iterable[Path], message: str) -> None:
    """
    Best-effort helper for auto-committing outputs in a git repo.

    This is optional. It will not raise if git is not configured.
    """
    root = Path.cwd()
    if not (root / ".git").exists():
        return

    str_paths = [str(p) for p in paths]
    try:
        subprocess.run(["git", "add", *str_paths], check=False)
        status = subprocess.run(
            ["git", "status", "--porcelain"], check=False, text=True, capture_output=True
        )
        if not status.stdout.strip():
            return
        subprocess.run(["git", "commit", "-m", message], check=False)
        subprocess.run(["git", "push"], check=False)
    except Exception:
        return


def getenv_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}
