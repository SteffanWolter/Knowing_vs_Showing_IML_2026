from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


@dataclass(frozen=True)
class BinaryChoiceQuestion:
    """
    A single binary-choice TruthfulQA item.

    `correct_idx` refers to the index in `choices`, so it is always 0 or 1.
    """

    question: str
    choices: tuple[str, str]
    correct_idx: int
    source: str = "truthfulqa"


def load_truthfulqa_binary_choice() -> list[BinaryChoiceQuestion]:
    """
    Load TruthfulQA and convert the multi-choice format into a deterministic binary-choice task.

    The Hugging Face `truthful_qa` dataset provides a set of answer choices and labels. We construct a
    two-option task by pairing the single correct choice with one distractor and alternating the
    correct option position deterministically to avoid label imbalance.
    """
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    questions: list[BinaryChoiceQuestion] = []

    for i, item in enumerate(ds):
        mc1 = item["mc1_targets"]
        choices: list[str] = list(mc1["choices"])
        labels: list[int] = list(mc1["labels"])

        correct_indices = [j for j, lab in enumerate(labels) if int(lab) == 1]
        if len(correct_indices) != 1:
            continue

        correct_choice = choices[correct_indices[0]]
        distractors = [c for j, c in enumerate(choices) if int(labels[j]) == 0]
        if not distractors:
            continue

        distractor = distractors[i % len(distractors)]
        correct_first = (i % 2) == 0
        binary_choices = (
            (correct_choice, distractor) if correct_first else (distractor, correct_choice)
        )
        correct_idx = 0 if correct_first else 1

        questions.append(
            BinaryChoiceQuestion(
                question=str(item["question"]),
                choices=(str(binary_choices[0]), str(binary_choices[1])),
                correct_idx=int(correct_idx),
            )
        )

    return questions


def as_dicts(items: list[BinaryChoiceQuestion]) -> list[dict[str, Any]]:
    """
    Convert questions to plain dicts, which is convenient for CSV outputs.
    """
    return [
        {
            "question": q.question,
            "choices": list(q.choices),
            "correct_idx": q.correct_idx,
            "source": q.source,
        }
        for q in items
    ]
