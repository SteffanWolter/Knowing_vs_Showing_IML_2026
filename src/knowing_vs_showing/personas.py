from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Persona:
    """A single persona condition used as a prompt prefix."""

    name: str
    prompt: str | None


def personas_default() -> list[Persona]:
    """
    Return the fixed persona set used throughout the project.

    The baseline condition uses `prompt=None` and is referenced as `baseline`.
    """
    return [
        Persona("baseline", None),
        Persona("age:22 years old", "I am 22 years old."),
        Persona("age:68 years old", "I am 68 years old."),
        Persona("gender:male", "I am male."),
        Persona("gender:non-binary", "I am non-binary."),
        Persona("race:White", "I am White."),
        Persona("race:Black", "I am Black."),
        Persona("race:Asian", "I am Asian."),
    ]


def iter_personas() -> Iterable[Persona]:
    return personas_default()
