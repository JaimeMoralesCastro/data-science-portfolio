# Abusive language detection and domain shift

A RoBERTa toxicity classifier reaches **F1 0.835** on the data it was trained
on. Point it at a different social network and it loses **14 points**. This
project is about that gap.

Natural Language Processing, Politehnica University of Bucharest (Erasmus+).

📄 [Project journal](docs/project-journal.pdf) · 🎤 [Slides](docs/slides/)

---

## The question

Toxicity detection is usually benchmarked in-domain: train on a corpus, test on
a held-out slice of the same corpus. But a moderation model is deployed on text
it has never seen the style of. So:

> Does a model trained on formal text (Wikipedia comments) hold up when it
> receives Twitter?

The answer, from a **3 × 3 design** — three training sets, each evaluated on
three test sets — is *not entirely*.

## Data

| Corpus | Source | Size | Register |
|---|---|---|---|
| **Jigsaw** | Wikipedia talk-page comments | 159,571 | long, formal, edited |
| **OLID** | Twitter | 11,916 | short, informal, slang and hashtags |

Only **10 %** of Jigsaw is toxic, and roughly **1 in 4** Twitter tokens never
appears in Wikipedia at all — the vocabulary itself shifts, not just the tone.

## Results

F1 on the positive (abusive) class. Rows are training sets, columns test sets:

| Trained on | test Jigsaw | test Twitter | combined |
|---|---|---|---|
| Jigsaw only | **0.835** | 0.693 | 0.817 |
| OLID only | 0.672 | **0.709** | 0.676 |
| Mixed | 0.809 | 0.691 | 0.793 |

Three readings:

- **Every model degrades off home turf.** The Wikipedia model drops 14 points on
  Twitter; the Twitter model drops 4 points on Wikipedia, from a much lower peak.
- **Mixing trades peak for stability.** The mixed model is never the best on any
  single test set, but it is never the worst either — the usual shape of a
  robustness/performance trade-off.
- **More data is not the same as more coverage.** OLID is 13 times smaller than
  Jigsaw and still beats it on Twitter, because the register matches.

Confusion matrix in [`results/`](results/).

## Method notes

- **Model:** `roberta-base`, fine-tuned for binary classification, 3 epochs.
- **Two cleaning pipelines**, one per register: mentions are replaced with
  `USER`, and hashtags are segmented back into words so `#notmyproblem` becomes
  recoverable signal rather than one unknown token.
- **Class weights instead of undersampling**, because discarding 90 % of Jigsaw
  to balance it would throw away most of the corpus.
- Trained on Google Colab with a T4 GPU, roughly 90 minutes per model.

## Requirements

```bash
pip install -r requirements.txt
```

The Jigsaw dataset must be downloaded from Kaggle (*Toxic Comment
Classification Challenge*) and uploaded when prompted; OLID is pulled
automatically from Hugging Face. Neither corpus nor the trained checkpoints are
committed here.

## Running it

The three scripts run in order:

```bash
python src/preprocess.py   # download, clean and split both corpora
python src/train.py        # fine-tune the 3 RoBERTa models -> models/
python src/evaluate.py     # evaluate 3 models x 3 test sets, print the table
```

## Layout

```
src/preprocess.py        data download, cleaning pipelines, splits
src/train.py             fine-tuning of the three models
src/evaluate.py          3x3 evaluation and results table
results/                 confusion matrix
docs/                    project journal and milestone slide decks
milestone1/              initial literature review and reference papers
```
