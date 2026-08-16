# Sanskrit-English Hybrid Machine Translation

Research repository supporting the paper:

**A Hybrid Rule-Based and Neural Approach for Sanskrit–English Machine Translation Using Karaka Semantic Role Labeling**

## Project overview

This project investigates a hybrid Sanskrit–English machine translation approach combining Sanskrit morphological features, Paninian/Karaka semantic roles, Transformer-based neural translation, and explainability/semantic-role consistency validation.

## Repository contents

- `pipeline.py` — supporting pipeline components
- `challenging_cases.tsv` — illustrative Sanskrit/Karaka stress-test examples
- `reported_results.csv` — results reported in the paper
- `fig1_hybrid_architecture.png` — paper architecture figure
- `reproducibility_checklist.md` — reproducibility checklist
- `requirements.txt` — Python dependencies
- `README.md` — project documentation

## Reported results

`reported_results.csv` contains the values reported in the manuscript's Table 1. They are labeled `reported_in_paper` and are **not presented as independently reproduced results**.

## Qualitative examples

`challenging_cases.tsv` contains illustrative examples for morphology and Karaka-role interpretation. These are demonstration cases, not additional benchmark results.

## Reproducibility

A complete independent reproduction requires the exact research dataset, preprocessing pipeline, model configuration, training setup, random seeds, checkpoints, and evaluation procedure used for the reported experiments. These should be added when available and legally shareable.

## Architecture

`fig1_hybrid_architecture.png` shows the six-stage pipeline described in the paper:

Sanskrit Input → Linguistic Feature Extraction → Karaka Role Assignment → Neural Translation → Explainability Validation → English Output.
