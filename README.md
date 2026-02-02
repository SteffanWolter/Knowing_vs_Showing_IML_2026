# Knowing vs. Showing

This folder is the cleaned and self-contained code submission for an interpretable machine learning project that studies a knowing–showing gap in large language models. The core idea is to compare (i) what a model answers under persona cues (showing) with (ii) what is linearly decodable from its internal representations (knowing). Concretely, we measure generation accuracy on TruthfulQA under different persona cues and compare it to probe-based accuracy computed from hidden states extracted at a fixed decision point in the prompt.

The project is structured as a small set of experiments that can be run independently and that all write machine-readable outputs (CSV plus a JSON config) to a user-chosen output directory. Each experiment is implemented as a CLI entry point and can be executed via Poetry. The same code paths are used for both quick smoke tests and full runs by changing a small number of flags (for example, limiting the number of TruthfulQA questions for faster iteration).

This code intentionally avoids any reliance on private datasets. TruthfulQA is loaded via the Hugging Face `datasets` library. Model checkpoints are loaded via `transformers`, and probes are trained using scikit-learn logistic regression.

## Installation

This project uses Poetry for dependency management. Install Poetry on your system using the official installation instructions, then run `poetry install` from inside this folder. If you run on a GPU machine, install a CUDA-enabled PyTorch build that matches your system. The code does not assume a specific CUDA version, but it does assume that `torch.cuda.is_available()` works when you request `--device cuda`.

If you want 4-bit loading, install optional dependencies with `poetry install -E quant`. If you want to regenerate plotting figures, install `poetry install -E plot`.

## Environment variables

Set the environment variables shown in `.env.example` before running the experiments. The Hugging Face token is recommended for reliable model downloads, especially for gated checkpoints. A GitHub token is only required if you enable the optional auto-push feature in the experiment scripts. If you prefer to store variables in a local `.env` file, you can copy `.env.example` to `.env`, but you still need to ensure the variables are exported into your shell environment before executing the commands.

## Running the experiments

All experiments are exposed as Poetry scripts and can be executed using `poetry run`. The entry points are `kvs-exp1-gap`, `kvs-exp2-layer-probe`, `kvs-exp3-logit-lens`, and `kvs-exp4-choice-logprob`. Each command supports `--help` and writes a CSV plus a JSON config to the output path you provide.

The recommended workflow is to start with a small TruthfulQA subset to confirm that the environment and tokenizer templates work, then switch to the full dataset (by using `--tqa-n 0`) for final numbers. For reproducibility, all splits use a fixed seed that is saved to the config JSON, and the two-way protocol in Experiment 1 trains on split A and evaluates on split B and then swaps the roles to reduce split idiosyncrasies.

The following examples assume you are inside this folder and that you have already run `poetry install`. They also assume that you have a GPU available when you set `--device cuda`.

```bash
poetry run kvs-exp1-gap \
  --model "allenai/OLMo-2-1124-7B-Instruct" \
  --tqa-n 50 \
  --split-seed 20 \
  --device cuda \
  --output "outputs/exp01_gap_smoke.csv"
```

```bash
poetry run kvs-exp2-layer-probe \
  --model "allenai/OLMo-2-1124-7B-Instruct" \
  --tqa-n 50 \
  --split-seed 20 \
  --device cuda \
  --layer-step 2 \
  --output "outputs/exp02_layer_probe_smoke.csv"
```

```bash
poetry run kvs-exp3-logit-lens \
  --model "allenai/OLMo-2-1124-7B-Instruct" \
  --tqa-n 50 \
  --split-seed 20 \
  --device cuda \
  --layer-step 2 \
  --output "outputs/exp03_logit_lens_smoke.csv"
```

```bash
poetry run kvs-exp4-choice-logprob \
  --model "allenai/OLMo-2-1124-7B-Instruct" \
  --tqa-n 50 \
  --split-seed 20 \
  --device cuda \
  --output "outputs/exp04_choice_logprob_smoke.csv"
```

If you want full runs on the complete TruthfulQA validation split, set `--tqa-n 0`. If you want a compact per-question trace for qualitative analysis, you can enable `--save-preds`. If you also want the raw question text and raw generated answers in the trace file, you can additionally enable `--include-question` and `--save-answers`, but you should expect the resulting CSV to be large.

## References and attribution

- **TruthfulQA** – Lin, S., Hilton, J., & Evans, O. (2022). *TruthfulQA: Measuring How Models Mimic Human Falsehoods*. ACL 2022. [Paper](https://aclanthology.org/2022.acl-long.229/) | [Dataset](https://huggingface.co/datasets/truthful_qa)

- **Probing Methodology** – Belinkov, Y. (2022). *Probing Classifiers: Promises, Shortcomings, and Advances*. Computational Linguistics. [Paper](https://aclanthology.org/2022.cl-1.7/)

- **Probe Selectivity** – Hewitt, J., & Liang, P. (2019). *Designing and Interpreting Probes with Control Tasks*. EMNLP 2019. [Paper](https://aclanthology.org/D19-1275/)

- **Tuned Lens** – Belrose, N., et al. (2023). *Eliciting Latent Predictions from Transformers with the Tuned Lens*. arXiv. [Paper](https://arxiv.org/abs/2303.08112)

- **Future Lens** – Pal, K., et al. (2023). *Future Lens: Anticipating Subsequent Tokens from a Single Hidden State*. CoNLL 2023. [Paper](https://aclanthology.org/2023.conll-1.37/)

- **Persona Prompting** – Cheng, M., Durmus, E., & Jurafsky, D. (2023). *Marked Personas: Using Natural Language Prompts to Measure Stereotypes in Language Models*. ACL 2023. [Paper](https://aclanthology.org/2023.acl-long.84/)

- **Implicit Personalization** – Jin, Z., et al. (2024). *Implicit Personalization in Language Models: A Systematic Study*. EMNLP 2024 Findings. [Paper](https://aclanthology.org/2024.findings-emnlp.717/)

- **Inference-Time Intervention** – Li, K., et al. (2023). *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model*. NeurIPS 2023. [Paper](https://papers.nips.cc/paper_files/paper/2023/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html)
