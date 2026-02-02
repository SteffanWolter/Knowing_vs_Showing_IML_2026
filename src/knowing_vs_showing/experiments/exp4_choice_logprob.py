from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from knowing_vs_showing.datasets.truthfulqa import (
    BinaryChoiceQuestion,
    load_truthfulqa_binary_choice,
)
from knowing_vs_showing.models.hf_loader import load_causal_lm
from knowing_vs_showing.personas import iter_personas
from knowing_vs_showing.prompts.truthfulqa import add_probe_suffix, build_truthfulqa_binary_prompt
from knowing_vs_showing.settings import DEFAULTS
from knowing_vs_showing.utils import (
    chat_text,
    cleanup_cuda,
    model_short_name,
    split_train_eval_indices,
    to_device,
    tokenize_text,
    try_git_commit_push,
    write_dataclass_json,
)


@dataclass(frozen=True)
class RunConfig:
    timestamp: str
    dataset: str
    model: str
    tqa_n: int
    split_seed: int
    eval_frac: float
    load_in_4bit: bool
    device: str
    option_token_id_mode: str
    opt1_token_id: int
    opt2_token_id: int
    notes: str


def _prompt(q: BinaryChoiceQuestion, persona: str | None) -> str:
    return build_truthfulqa_binary_prompt(
        question=q.question,
        choices=q.choices,
        persona=persona,
        include_reason=False,
    )


def _single_token_id(tokenizer: Any, text: str) -> int | None:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return int(ids[0]) if len(ids) == 1 else None


def pick_option_token_ids(tokenizer: Any) -> tuple[int, int, str]:
    candidates = [(" 1", " 2"), ("1", "2")]
    for a, b in candidates:
        a_id = _single_token_id(tokenizer, a)
        b_id = _single_token_id(tokenizer, b)
        if a_id is not None and b_id is not None:
            return int(a_id), int(b_id), f"{a}/{b}"
    return (
        int(tokenizer.encode("1", add_special_tokens=False)[0]),
        int(tokenizer.encode("2", add_special_tokens=False)[0]),
        "fallback-first-token",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 04: final-layer choice-by-logprob evaluation (TruthfulQA), without free-form generation."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model id, e.g., allenai/OLMo-2-1124-7B-Instruct",
    )
    parser.add_argument(
        "--tqa-n", type=int, default=0, help="Limit dataset size before split (0 = full)."
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULTS.seed)
    parser.add_argument("--eval-frac", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=DEFAULTS.device)
    parser.add_argument("--load-in-4bit", action="store_true")

    parser.add_argument("--output", required=True, help="CSV output path.")
    parser.add_argument(
        "--save-preds", action="store_true", help="Also save per-question predictions."
    )
    parser.add_argument(
        "--include-question", action="store_true", help="Include question text in prediction CSV."
    )

    parser.add_argument(
        "--auto-push", action="store_true", help="Auto commit and push outputs (best-effort)."
    )
    parser.add_argument(
        "--git-commit-msg", type=str, default="exp4: choice-by-logprob (TruthfulQA)"
    )

    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    questions = load_truthfulqa_binary_choice()
    if args.tqa_n and int(args.tqa_n) > 0:
        questions = questions[: int(args.tqa_n)]

    _train_idx, eval_idx = split_train_eval_indices(
        len(questions), int(args.split_seed), float(args.eval_frac)
    )
    eval_set = [questions[i] for i in eval_idx]

    loaded = load_causal_lm(
        str(args.model), device=str(args.device), load_in_4bit=bool(args.load_in_4bit)
    )
    model = loaded.model
    tokenizer = loaded.tokenizer

    model_label = model_short_name(str(args.model))

    opt1_id, opt2_id, opt_id_mode = pick_option_token_ids(tokenizer)

    config = RunConfig(
        timestamp=timestamp,
        dataset="TruthfulQA",
        model=str(args.model),
        tqa_n=int(args.tqa_n),
        split_seed=int(args.split_seed),
        eval_frac=float(args.eval_frac),
        load_in_4bit=bool(args.load_in_4bit),
        device=str(args.device),
        option_token_id_mode=str(opt_id_mode),
        opt1_token_id=int(opt1_id),
        opt2_token_id=int(opt2_id),
        notes="Computes next-token probabilities for option 1 vs 2 at the JSON Answer position using final logits.",
    )
    config_path = out_path.with_name(out_path.stem + "_config.json")
    write_dataclass_json(config_path, config)

    rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    model.eval()
    with torch.no_grad():
        for persona in iter_personas():
            correct = 0
            margin_sum = 0.0
            logit_margin_sum = 0.0
            total = 0

            for q_idx, q in enumerate(tqdm(eval_set, desc=f"choice_logprob:{persona.name}")):
                prompt = _prompt(q, persona.prompt)
                text = add_probe_suffix(chat_text(tokenizer, prompt, add_generation_prompt=True))
                batch = to_device(tokenize_text(tokenizer, text), str(args.device))
                outputs = model(**batch, return_dict=True)
                logits = outputs.logits[:, -1, :].squeeze(0).to(torch.float32)

                opt_logits = torch.stack([logits[opt1_id], logits[opt2_id]], dim=0)
                probs = torch.softmax(opt_logits, dim=0).cpu().numpy()

                correct_idx = int(q.correct_idx)
                pred_idx = int(np.argmax(probs))
                is_correct = int(pred_idx == correct_idx)

                correct += is_correct
                margin = float(probs[correct_idx] - probs[1 - correct_idx])
                logit_margin = float(opt_logits[correct_idx] - opt_logits[1 - correct_idx])
                margin_sum += margin
                logit_margin_sum += logit_margin

                if args.save_preds:
                    pred_row: dict[str, Any] = {
                        "model": model_label,
                        "dataset": "TruthfulQA",
                        "persona": persona.name,
                        "question_idx": int(q_idx),
                        "correct_idx": int(correct_idx),
                        "pred_idx": int(pred_idx),
                        "p_opt1": float(probs[0]),
                        "p_opt2": float(probs[1]),
                        "margin_correct_minus_wrong": float(margin),
                        "logit_margin_correct_minus_wrong": float(logit_margin),
                        "is_correct": int(is_correct),
                        "split_seed": int(args.split_seed),
                        "eval_n": int(len(eval_set)),
                    }
                    if args.include_question:
                        pred_row["question"] = q.question
                    pred_rows.append(pred_row)

                total += 1

            acc = float(correct / total) if total else 0.0
            mean_margin = float(margin_sum / total) if total else 0.0
            mean_logit_margin = float(logit_margin_sum / total) if total else 0.0

            rows.append(
                {
                    "model": model_label,
                    "dataset": "TruthfulQA",
                    "persona": persona.name,
                    "choice_logprob_accuracy": float(acc),
                    "mean_margin_correct_minus_wrong": float(mean_margin),
                    "mean_logit_margin_correct_minus_wrong": float(mean_logit_margin),
                    "n_questions": int(total),
                    "split_seed": int(args.split_seed),
                    "eval_n": int(len(eval_set)),
                    "eval_frac": float(args.eval_frac),
                    "option_token_id_mode": str(opt_id_mode),
                    "opt1_token_id": int(opt1_id),
                    "opt2_token_id": int(opt2_id),
                }
            )

    pd.DataFrame(rows).to_csv(out_path, index=False)
    outputs: list[Path] = [out_path, config_path]

    if args.save_preds:
        preds_path = out_path.with_name(out_path.stem + "_preds.csv")
        pd.DataFrame(pred_rows).to_csv(preds_path, index=False)
        outputs.append(preds_path)

    cleanup_cuda()

    if args.auto_push:
        try_git_commit_push(outputs, args.git_commit_msg)


if __name__ == "__main__":
    main()
