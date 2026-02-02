from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from knowing_vs_showing.datasets.truthfulqa import (
    BinaryChoiceQuestion,
    load_truthfulqa_binary_choice,
)
from knowing_vs_showing.models.hf_loader import load_causal_lm
from knowing_vs_showing.parsing import parse_binary_choice
from knowing_vs_showing.personas import iter_personas
from knowing_vs_showing.probing.logreg import train_logreg_probe
from knowing_vs_showing.prompts.truthfulqa import add_probe_suffix, build_truthfulqa_binary_prompt
from knowing_vs_showing.settings import DEFAULTS
from knowing_vs_showing.utils import (
    chat_text,
    cleanup_cuda,
    model_short_name,
    split_half_indices,
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
    load_in_4bit: bool
    device: str
    split_a_idx: list[int]
    split_b_idx: list[int]
    eval_frac: float
    max_new_tokens: int
    notes: str


def _prompt_for_eval(q: BinaryChoiceQuestion, persona: str | None, *, include_reason: bool) -> str:
    return build_truthfulqa_binary_prompt(
        question=q.question,
        choices=q.choices,
        persona=persona,
        include_reason=include_reason,
    )


def _forward_hidden_state(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str,
    device: str,
) -> torch.Tensor:
    """
    Return the final-layer hidden state at the last token position of the decision prompt.
    """
    text = chat_text(tokenizer, prompt, add_generation_prompt=True)
    text = add_probe_suffix(text)
    batch = to_device(tokenize_text(tokenizer, text), device)
    with torch.no_grad():
        outputs = model(**batch, output_hidden_states=True, return_dict=True)
    h = outputs.hidden_states[-1][:, -1, :].to(torch.float32).cpu().squeeze(0)
    return h


def _train_probe(
    model: Any,
    tokenizer: Any,
    train_questions: list[BinaryChoiceQuestion],
    *,
    device: str,
    seed: int,
) -> tuple[Any, float, float]:
    states: list[torch.Tensor] = []
    labels: list[int] = []
    for q in train_questions:
        prompt = _prompt_for_eval(q, None, include_reason=False)
        states.append(_forward_hidden_state(model, tokenizer, prompt=prompt, device=device))
        labels.append(int(q.correct_idx))

    res = train_logreg_probe(states=states, labels=labels, seed=seed, cv_folds=5)
    return res.probe, res.cv_mean, res.cv_std


def _generate_choice(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> tuple[str, int]:
    text = chat_text(tokenizer, prompt, add_generation_prompt=True)
    batch = to_device(tokenize_text(tokenizer, text), device)

    with torch.no_grad():
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", None)
        if pad_id is None:
            pad_id = 0
        out = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=int(pad_id),
        )

    gen = out[0][batch["input_ids"].shape[1] :]
    answer_text = tokenizer.decode(gen, skip_special_tokens=True)
    pred = parse_binary_choice(answer_text)
    return answer_text, int(pred)


def _probe_predict(
    model: Any,
    tokenizer: Any,
    probe: Any,
    *,
    prompt: str,
    device: str,
) -> int:
    h = _forward_hidden_state(model, tokenizer, prompt=prompt, device=device)
    x = h.unsqueeze(0).numpy()
    return int(probe.predict(x)[0])


def _run_direction(
    model: Any,
    tokenizer: Any,
    *,
    model_label: str,
    train_set: list[BinaryChoiceQuestion],
    eval_set: list[BinaryChoiceQuestion],
    direction: str,
    device: str,
    seed: int,
    max_new_tokens: int,
    save_preds: bool,
    include_question: bool,
    save_answers: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None, float, float]:
    probe, cv_mean, cv_std = _train_probe(model, tokenizer, train_set, device=device, seed=seed)

    rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    for persona in iter_personas():
        gen_correct = 0
        probe_correct = 0
        unclear = 0

        for q_idx, q in enumerate(eval_set):
            gen_prompt = _prompt_for_eval(q, persona.prompt, include_reason=True)
            probe_prompt = _prompt_for_eval(q, persona.prompt, include_reason=False)

            answer_text, gen_pred = _generate_choice(
                model, tokenizer, prompt=gen_prompt, device=device, max_new_tokens=max_new_tokens
            )
            probe_pred = _probe_predict(model, tokenizer, probe, prompt=probe_prompt, device=device)

            correct_idx = int(q.correct_idx)
            if gen_pred == -1:
                unclear += 1
            if gen_pred == correct_idx:
                gen_correct += 1
            if probe_pred == correct_idx:
                probe_correct += 1

            if save_preds:
                pred_row: dict[str, Any] = {
                    "model": model_label,
                    "dataset": "TruthfulQA",
                    "direction": direction,
                    "persona": persona.name,
                    "question_idx": int(q_idx),
                    "correct_idx": correct_idx,
                    "gen_pred": int(gen_pred),
                    "probe_pred": int(probe_pred),
                    "gen_is_correct": int(gen_pred == correct_idx) if gen_pred != -1 else 0,
                    "probe_is_correct": int(probe_pred == correct_idx),
                    "gen_unclear": int(gen_pred == -1),
                }
                if include_question:
                    pred_row["question"] = q.question
                if save_answers:
                    pred_row["answer_text"] = answer_text
                pred_rows.append(pred_row)

        total = len(eval_set)
        gen_acc = float(gen_correct / total) if total else 0.0
        probe_acc = float(probe_correct / total) if total else 0.0
        gap = float(probe_acc - gen_acc)

        rows.append(
            {
                "model": model_label,
                "dataset": "TruthfulQA",
                "persona": persona.name,
                "direction": direction,
                "generation_accuracy": gen_acc,
                "probe_accuracy": probe_acc,
                "knowledge_gap": gap,
                "n_questions": int(total),
                "n_unclear": int(unclear),
                "train_n": int(len(train_set)),
                "eval_n": int(len(eval_set)),
                "probe_cv_mean": float(cv_mean),
                "probe_cv_std": float(cv_std),
            }
        )

    preds_df = pd.DataFrame(pred_rows) if save_preds else None
    return pd.DataFrame(rows), preds_df, float(cv_mean), float(cv_std)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 01: two-way 50/50 split knowing–showing gap on TruthfulQA."
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
    parser.add_argument(
        "--eval-frac", type=float, default=0.5, help="Must be 0.5 for the two-way half split."
    )
    parser.add_argument("--device", type=str, default=DEFAULTS.device)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULTS.max_new_tokens)
    parser.add_argument("--load-in-4bit", action="store_true")

    parser.add_argument("--output", required=True, help="CSV output path for aggregated results.")
    parser.add_argument(
        "--save-preds", action="store_true", help="Also save per-question predictions."
    )
    parser.add_argument(
        "--include-question", action="store_true", help="Include question text in prediction CSV."
    )
    parser.add_argument(
        "--save-answers",
        action="store_true",
        help="Include raw generated answers in prediction CSV.",
    )

    parser.add_argument(
        "--auto-push", action="store_true", help="Auto commit and push outputs (best-effort)."
    )
    parser.add_argument("--git-commit-msg", type=str, default="exp1: two-way gap (TruthfulQA)")

    args = parser.parse_args()

    if abs(float(args.eval_frac) - 0.5) > 1e-9:
        raise ValueError(
            "This experiment is defined as a strict 50/50 split. Please use --eval-frac 0.5."
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    questions = load_truthfulqa_binary_choice()
    if args.tqa_n and int(args.tqa_n) > 0:
        questions = questions[: int(args.tqa_n)]

    a_idx, b_idx = split_half_indices(len(questions), int(args.split_seed))
    split_a = [questions[i] for i in a_idx]
    split_b = [questions[i] for i in b_idx]

    model_label = model_short_name(args.model)

    config = RunConfig(
        timestamp=timestamp,
        dataset="TruthfulQA",
        model=str(args.model),
        tqa_n=int(args.tqa_n),
        split_seed=int(args.split_seed),
        load_in_4bit=bool(args.load_in_4bit),
        device=str(args.device),
        split_a_idx=a_idx,
        split_b_idx=b_idx,
        eval_frac=float(args.eval_frac),
        max_new_tokens=int(args.max_new_tokens),
        notes="Two-way protocol: train(A)->eval(B) and train(B)->eval(A); mean reported over both directions.",
    )

    config_path = out_path.with_name(out_path.stem + "_config.json")
    write_dataclass_json(config_path, config)

    loaded = load_causal_lm(
        str(args.model), device=str(args.device), load_in_4bit=bool(args.load_in_4bit)
    )
    model = loaded.model
    tokenizer = loaded.tokenizer

    df_ab, preds_ab, _, _ = _run_direction(
        model,
        tokenizer,
        model_label=model_label,
        train_set=split_a,
        eval_set=split_b,
        direction="A_to_B",
        device=str(args.device),
        seed=int(args.split_seed),
        max_new_tokens=int(args.max_new_tokens),
        save_preds=bool(args.save_preds),
        include_question=bool(args.include_question),
        save_answers=bool(args.save_answers),
    )

    df_ba, preds_ba, _, _ = _run_direction(
        model,
        tokenizer,
        model_label=model_label,
        train_set=split_b,
        eval_set=split_a,
        direction="B_to_A",
        device=str(args.device),
        seed=int(args.split_seed),
        max_new_tokens=int(args.max_new_tokens),
        save_preds=bool(args.save_preds),
        include_question=bool(args.include_question),
        save_answers=bool(args.save_answers),
    )

    mean_df = (
        pd.concat([df_ab, df_ba], ignore_index=True)
        .groupby(["model", "dataset", "persona"], as_index=False)
        .agg(
            generation_accuracy=("generation_accuracy", "mean"),
            probe_accuracy=("probe_accuracy", "mean"),
            knowledge_gap=("knowledge_gap", "mean"),
            n_questions=("n_questions", "mean"),
            n_unclear=("n_unclear", "mean"),
            train_n=("train_n", "mean"),
            eval_n=("eval_n", "mean"),
            probe_cv_mean=("probe_cv_mean", "mean"),
            probe_cv_std=("probe_cv_std", "mean"),
        )
    )
    mean_df["direction"] = "mean"

    full_df = pd.concat([df_ab, df_ba, mean_df], ignore_index=True)
    full_df.to_csv(out_path, index=False)

    outputs: list[Path] = [out_path, config_path]

    if args.save_preds and preds_ab is not None and preds_ba is not None:
        preds_path = out_path.with_name(out_path.stem + "_preds.csv")
        pd.concat([preds_ab, preds_ba], ignore_index=True).to_csv(preds_path, index=False)
        outputs.append(preds_path)

    cleanup_cuda()

    if args.auto_push:
        try_git_commit_push(outputs, args.git_commit_msg)


if __name__ == "__main__":
    main()
