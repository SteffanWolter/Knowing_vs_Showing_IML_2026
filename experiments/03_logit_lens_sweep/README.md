# Experiment 03: Logit-lens style layer sweep (binary decision margin)

This experiment performs a logit-lens style analysis for the binary-choice TruthfulQA prompt. For each transformer layer, it projects the hidden state at the answer position through the language model head and compares the relative probability of choosing option 1 versus option 2. The main output is the layer-wise decision accuracy and the average probability margin between the correct and incorrect options.

This analysis does not train a probe. It is intended as a diagnostic that is closer to the model’s own output interface than a learned probe, while still allowing a layer-by-layer view of the decision signal.

You can run the logit-lens sweep via Poetry. The following command performs a small smoke test on 50 questions and sweeps every second layer.

```bash
poetry run kvs-exp3-logit-lens \
  --model "allenai/OLMo-2-1124-7B-Instruct" \
  --tqa-n 50 \
  --split-seed 20 \
  --device cuda \
  --layer-step 2 \
  --output "outputs/exp03_logit_lens_smoke.csv"
```

For the final run, set `--tqa-n 0`. If you enable `--save-preds`, the experiment writes a per-question CSV that contains a row for every layer and every question, so the file size grows quickly.
