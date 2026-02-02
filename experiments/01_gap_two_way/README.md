# Experiment 01: Two-way knowing–showing gap on TruthfulQA

This experiment measures the knowing–showing gap on TruthfulQA using a 50/50 shuffle split with two-way averaging. The probe is trained on split A and evaluated on split B under all persona prompts, and then the direction is swapped (train on B, evaluate on A). The reported per-persona result is the mean over both directions, which reduces sensitivity to any single random split.

The output is a CSV file with per-direction rows and an additional mean row, plus a JSON configuration file that records the split indices and settings. If you enable the optional prediction dump, the experiment also writes a per-question CSV that can include the raw question text and the raw generated answer string for qualitative inspection.

You can run the experiment from the repository root via Poetry. The following command performs a small smoke test on 50 questions and writes results to an `outputs/` directory inside the submission folder.

```bash
poetry run kvs-exp1-gap \
  --model "allenai/OLMo-2-1124-7B-Instruct" \
  --tqa-n 50 \
  --split-seed 20 \
  --device cuda \
  --output "outputs/exp01_gap_smoke.csv"
```

For the final numbers, set `--tqa-n 0` to use the full TruthfulQA validation split. If you want a per-question trace, add `--save-preds`. If you also want to store the raw question and raw generation output, add `--include-question` and `--save-answers`, but keep in mind that these settings create large files.
