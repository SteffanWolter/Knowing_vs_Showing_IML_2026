# Experiment 02: Layer-wise linear probes (TruthfulQA correctness)

This experiment trains a logistic regression probe for multiple transformer layers and evaluates how linearly decodable the TruthfulQA correctness signal is as a function of depth. The probe is trained on the baseline (no persona) prompts on the training split, and then evaluated on the evaluation split for each persona prompt. The main purpose is to test whether the internal correctness signal is stable across personas and to visualize at which depth it becomes easily recoverable by a linear classifier.

You can run the layer sweep via Poetry. The following command performs a small smoke test on 50 questions and sweeps every second layer.

```bash
poetry run kvs-exp2-layer-probe \
  --model "allenai/OLMo-2-1124-7B-Instruct" \
  --tqa-n 50 \
  --split-seed 20 \
  --device cuda \
  --layer-step 2 \
  --output "outputs/exp02_layer_probe_smoke.csv"
```

For the final run on the full dataset, set `--tqa-n 0`. If you want to inspect individual questions, you can add `--save-preds`. If you also add `--include-question`, the predictions file will contain the raw question strings.
