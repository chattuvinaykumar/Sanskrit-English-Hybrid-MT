# Sanskrit-English Hybrid Machine Translation with Paninian Karaka SRL

Official repository for the ICON 2026 paper:
**"A Hybrid Rule-Based and Neural Approach for Sanskrit-English Machine Translation Using Karaka Semantic Role Labeling"**  
Accepted for Poster Session & Poster Booster (Short Oral Presentation), ACL Anthology Publication.

**Authors:** Chattu Vinaykumar, Veeraiah Thalagondapati  
**Affiliation:** Department of Computer Science & Engineering, Ecole Centrale School of Engineering, Mahindra University, Hyderabad, India  

---

## Repository Structure

```text
├── paper/
│   ├── main.tex                 # 6-page Camera-Ready LaTeX source (ACL format)
│   └── custom.bib               # 17 complete bibliographic citations with DOIs & ACL links
├── src/
│   ├── paninian_morphology.py   # Greedy longest-suffix Paninian morphological engine
│   ├── karaka_analyzer.py       # Paninian syntactic-semantic role assigner (Kartri, Karma, Karana)
│   ├── hybrid_transformer_model.py # PyTorch Transformer with Feature Embeddings & Positional Encoding
│   ├── explainability_validator.py # Post-hoc validation layer with bilingual lemma alignment
│   ├── train.py                 # Training script with label smoothing (0.1) and gradient clipping
│   └── evaluate_and_translate.py# Interactive demo pipeline and metric reporter
├── dataset/
│   ├── corpus_breakdown.csv     # 800/700/500 category split & 80/10/10 data partitions
│   └── dataset_samples.csv      # Annotated Sanskrit-English parallel sentence pairs
├── experiments/
│   └── results_ablation.csv     # Quantitative ablation benchmarks (BLEU, METEOR, Human)
└── README.md                    # Project setup and reproduction guide
```

---

## Quickstart

### 1. Requirements
```bash
pip install torch sacrebleu nltk
```

### 2. Run Inference & Explainability Validation Demo
```bash
python src/evaluate_and_translate.py
```

### 3. Initialize & Train Model
```bash
python src/train.py
```

### 4. Compile Camera-Ready Paper
```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Benchmark Results (Test Set, n=200)

| System Configuration | BLEU | METEOR | Human Rating (1–5) |
| :--- | :---: | :---: | :---: |
| Baseline (Transformer only) | 22.4 | 0.28 | 3.2 |
| + Morphological features | 25.8 | 0.31 | 3.7 |
| + Karaka semantic roles | 25.9 | 0.32 | 3.9 |
| **Full Hybrid (+ Validation)** | **25.8** | **0.33** | **4.1** |

*(Human evaluation inter-rater agreement: Fleiss' $\kappa = 0.73$)*
