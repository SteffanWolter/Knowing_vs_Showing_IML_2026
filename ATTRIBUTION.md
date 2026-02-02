# Attribution and reuse

This submission contains original experiment code written for an interpretable machine learning course project. The goal is to provide a clean and reproducible implementation of a knowing–showing analysis on a public benchmark (TruthfulQA) with persona cues, together with a small set of complementary diagnostics (layer-wise probes, logit-lens style projections, and final-layer choice-by-logprob).

The project uses external libraries for model access and basic machine learning functionality. All model inference is performed via the Hugging Face Transformers library, and TruthfulQA is loaded via the Hugging Face Datasets library. Linear probes are implemented using scikit-learn logistic regression. These libraries are used as intended and are not modified.

The experiment design is inspired by prior work on probing and on analyzing intermediate representations, including general discussions of probing methodology and diagnostics such as logit-lens style projections. These sources are cited in the accompanying report and in the bibliography used for the written submission. The code in this folder is not copied verbatim from those papers, but implements the relevant ideas in a task-specific way for TruthfulQA’s binary-choice format and for the fixed persona prompts used in the project.

