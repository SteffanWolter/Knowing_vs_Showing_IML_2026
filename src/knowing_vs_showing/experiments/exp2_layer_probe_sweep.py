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
from knowing_vs_showing.probing.logreg import train_logreg_probe
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
    layers: list[int]
    layer_step: int
    notes: str


def _prompt(q: BinaryChoiceQuestion, persona: str | None) -> str:
    return build_truthfulqa_binary_prompt(
        question=q.question,
        choices=q.choices,
        persona=persona,
        include_reason=False,
    )


def _parse_layers(layers_arg: str | None, num_layers: int, step: int) -> list[int]:
    if layers_arg:
        raw = [int(x.strip()) for x in layers_arg.split(",") if x.strip()]
        return sorted({layer_index for layer_index in raw if 0 <= layer_index < num_layers})
    return list(range(0, num_layers, max(1, int(step))))


def _extract_states_by_layer(
    model: Any,
    tokenizer: Any,
    *,
    questions: list[BinaryChoiceQuestion],
    persona_prompt: str | None,
    layers: list[int],
    device: str,
) -> tuple[dict[int, list[torch.Tensor]], list[int]]:
    states: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    labels: list[int] = []

    model.eval()
    with torch.no_grad():
        for q in tqdm(questions, desc="hidden_states", leave=False):
            prompt = _prompt(q, persona_prompt)
            text = add_probe_suffix(chat_text(tokenizer, prompt, add_generation_prompt=True))
            batch = to_device(tokenize_text(tokenizer, text), device)
            outputs = model(**batch, output_hidden_states=True, return_dict=True)

            hs_all = outputs.hidden_states
            for layer in layers:
                hs = hs_all[layer + 1][:, -1, :].to(torch.float32).cpu().squeeze(0)
                states[layer].append(hs)
            labels.append(int(q.correct_idx))

    return states, labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 02: layer-wise linear probe sweep (TruthfulQA)."
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

    parser.add_argument(
        "--layers", type=str, default=None, help="Comma-separated layer indices (0-based)."
    )
    parser.add_argument(
        "--layer-step", type=int, default=1, help="Stride when sweeping all layers."
    )

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
        "--git-commit-msg", type=str, default="exp2: layer probe sweep (TruthfulQA)"
    )

    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    questions = load_truthfulqa_binary_choice()
    if args.tqa_n and int(args.tqa_n) > 0:
        questions = questions[: int(args.tqa_n)]

    train_idx, eval_idx = split_train_eval_indices(
        len(questions), int(args.split_seed), float(args.eval_frac)
    )
    train_set = [questions[i] for i in train_idx]
    eval_set = [questions[i] for i in eval_idx]

    loaded = load_causal_lm(
        str(args.model), device=str(args.device), load_in_4bit=bool(args.load_in_4bit)
    )
    model = loaded.model
    tokenizer = loaded.tokenizer

    num_layers = int(getattr(model.config, "num_hidden_layers"))
    layers = _parse_layers(args.layers, num_layers, int(args.layer_step))

    model_label = model_short_name(str(args.model))

    config = RunConfig(
        timestamp=timestamp,
        dataset="TruthfulQA",
        model=str(args.model),
        tqa_n=int(args.tqa_n),
        split_seed=int(args.split_seed),
        eval_frac=float(args.eval_frac),
        load_in_4bit=bool(args.load_in_4bit),
        device=str(args.device),
        layers=list(layers),
        layer_step=int(args.layer_step),
        notes="Trains a separate logistic regression probe for each layer on baseline prompts, then evaluates per persona.",
    )
    config_path = out_path.with_name(out_path.stem + "_config.json")
    write_dataclass_json(config_path, config)

    train_states, train_labels = _extract_states_by_layer(
        model,
        tokenizer,
        questions=train_set,
        persona_prompt=None,
        layers=layers,
        device=str(args.device),
    )

    probes: dict[int, Any] = {}
    cv_mean: dict[int, float] = {}
    cv_std: dict[int, float] = {}

    for layer in tqdm(layers, desc="train_probes"):
        res = train_logreg_probe(
            states=train_states[layer], labels=train_labels, seed=int(args.split_seed), cv_folds=5
        )
        probes[layer] = res.probe
        cv_mean[layer] = res.cv_mean
        cv_std[layer] = res.cv_std

    rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    for persona in iter_personas():
        eval_states, eval_labels = _extract_states_by_layer(
            model,
            tokenizer,
            questions=eval_set,
            persona_prompt=persona.prompt,
            layers=layers,
            device=str(args.device),
        )
        y_true = np.array(eval_labels, dtype=np.int64)

        for layer in layers:
            x = torch.stack(eval_states[layer]).to(torch.float32).cpu().numpy()
            preds = probes[layer].predict(x)
            acc = float((preds == y_true).mean()) if len(y_true) else 0.0

            rows.append(
                {
                    "model": model_label,
                    "dataset": "TruthfulQA",
                    "persona": persona.name,
                    "layer": int(layer),
                    "probe_accuracy": float(acc),
                    "probe_cv_mean": float(cv_mean[layer]),
                    "probe_cv_std": float(cv_std[layer]),
                    "n_questions": int(len(eval_set)),
                    "train_n": int(len(train_set)),
                    "split_seed": int(args.split_seed),
                    "eval_n": int(len(eval_set)),
                    "eval_frac": float(args.eval_frac),
                }
            )

            if args.save_preds:
                for i, q in enumerate(eval_set):
                    pred_row: dict[str, Any] = {
                        "model": model_label,
                        "dataset": "TruthfulQA",
                        "persona": persona.name,
                        "layer": int(layer),
                        "question_idx": int(i),
                        "correct_idx": int(q.correct_idx),
                        "pred_idx": int(preds[i]),
                        "is_correct": int(preds[i] == y_true[i]),
                        "split_seed": int(args.split_seed),
                        "eval_n": int(len(eval_set)),
                        "train_n": int(len(train_set)),
                    }
                    if args.include_question:
                        pred_row["question"] = q.question
                    pred_rows.append(pred_row)

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
