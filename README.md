# Sanskrit-English Hybrid Machine Translation

Research repository supporting the paper:

**A Hybrid Rule-Based and Neural Approach for Sanskrit–English Machine Translation Using Karaka Semantic Role Labeling**

## Project overview

This project investigates a hybrid Sanskrit–English machine translation approach that combines:

- Sanskrit morphological feature extraction
- Paninian/Karaka semantic role information
- Transformer-based neural translation
- Explainability and semantic-role consistency validation

The repository currently contains research documentation, supporting pipeline materials, qualitative demonstration examples, the paper figure, and the results reported in the manuscript.

## Repository structure

- `src/` — supporting pipeline components
- `data/` — dataset schema and data notes
- `examples/` — challenging qualitative demonstration cases
- `experiments/` — experiment and reproducibility notes
- `results/` — results reported in the paper
- `figures/` — paper figures
- `docs/` — reproducibility documentation

## Reported results

`results/reported_results.csv` contains the values reported in the manuscript's Table 1. These values are labeled `reported_in_paper` and should **not** be interpreted as an independent reproduction of the experiments.

The repository does not currently claim that the reported benchmark results have been independently reproduced from the original dataset and training checkpoints.

## Qualitative examples

The examples in `examples/` are illustrative stress-test cases for Sanskrit morphology and Karaka-role interpretation. They are demonstration examples, not additional benchmark results.

## Reproducibility

A complete independent reproduction requires the exact research dataset, preprocessing pipeline, model configuration, training setup, random seeds, checkpoints, and evaluation procedure used for the reported experiments. These should be added when they are available and legally shareable.

## Figure

The architecture figure in `figures/` shows the six-stage pipeline described in the paper:

Sanskrit Input → Linguistic Feature Extraction → Karaka Role Assignment → Neural Translation → Explainability Validation → English Output.
