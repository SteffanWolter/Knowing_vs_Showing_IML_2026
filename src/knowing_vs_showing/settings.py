from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Defaults:
    """Central defaults for CLI scripts."""

    seed: int = 20
    device: str = "cuda"
    max_new_tokens: int = 96


DEFAULTS = Defaults()
