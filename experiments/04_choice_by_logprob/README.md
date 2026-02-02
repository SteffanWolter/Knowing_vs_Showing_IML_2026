# Experiment 04: Choice-by-logprob (final layer, no free-form generation)

This experiment evaluates the binary-choice task without free-form generation. It computes the next-token probabilities for the option tokens at the JSON answer position using the final model logits, and it selects the option with higher probability. This isolates the decision signal from any downstream formatting effects caused by decoding a longer JSON object.

You can run this evaluation via Poetry. The following command performs a small smoke test on 50 questions.

```bash
poetry run kvs-exp4-choice-logprob \
  --model "allenai/OLMo-2-1124-7B-Instruct" \
  --tqa-n 50 \
  --split-seed 20 \
  --device cuda \
  --output "outputs/exp04_choice_logprob_smoke.csv"
```

For the final run, set `--tqa-n 0`. If you enable `--save-preds`, the experiment writes a per-question trace file with the predicted option, probabilities, and margins.
