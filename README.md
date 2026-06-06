# French Caselaw — Legal Domain Classification

Multi-class classification of French court decisions into 4 legal domains: Civil (`civ`), Commercial (`com`), Criminal (`crim`), Social (`soc`). Production use case: routing decisions in a caselaw search engine.

## Data

[`antoinejeannot/jurisprudence`](https://huggingface.co/datasets/antoinejeannot/jurisprudence) — French court decisions from HuggingFace.

- **Train/test**: `cour_de_cassation` (~535k decisions, stratified 80/20)
- **OOD**: `tribunal_judiciaire` + `cour_d_appel` (~140k decisions)

Labels are inferred from chamber names (`chambre sociale` → `soc`, etc.). OOD evaluation on different court hierarchies is the real generalization test.

## Key Challenges

- **Template overfitting**: cassation decisions share a standardized format — models can classify by structure rather than legal vocabulary, and fail on diverse inputs
- **Class imbalance**: up to 3× between `civ` (majority) and `com`/`crim` (minority) → `class_weight='balanced'`, Macro F2 as primary metric
- **French morphology**: `licencier`/`licenciement` are the same signal — char n-grams and CamemBERT handle this better than word-level tokenisation

## Results

Primary metric: **Macro F2** (search context — FN more costly than FP).

| Model | OOS F2 | OOD F2 | Inference |
|---|---|---|---|
| Dummy | 24.9% | — | — |
| TF-IDF + Naive Bayes | 94.2% | 49.0% | 0.4 ms |
| TF-IDF + Logit (word) | 98.5% | 55.1% | 0.4 ms |
| TF-IDF + Logit (char 3-5) | **98.5%** | **59.5%** | 3.3 ms |
| Frozen CamemBERT + Logit | 95.8% | 35.3% | 38 ms |

* **`char_logit` is the strongest baseline**: best OOD generalisation (59.5%), near-perfect OOS (98.5%), acceptable latency
* **Frozen CamemBERT has the worst OOD drop (−60pp)**: CLS embedding encodes cassation structure rather than legal semantics — template overfitting in practice

**Next**: fine-tune CamemBERT with a classification head on cassation data.

## Structure

```
transfer.ipynb      # main notebook
preprocessing.py    # HTML stripping, label assignment, train/test split
embedding.py        # CamemBERT embed(), save/load .npz
evaluation.py       # eval_multiclass(), EvalMulticlassResult
visualization.py    # plots, summary_table()
cache.py            # load_or_run() — cache by file extension (.pkl/.parquet/.npz)
```

## Setup

```bash
pip install -r requirements.txt
jupyter notebook transfer.ipynb
```

Embeddings and fitted models are cached under `cache/` on first run.
