# Flight price drop prediction

**Should I buy this flight now, or wait?** Given a flight search, a binary
classifier predicts whether the price will drop by at least 5 % within the next
seven days (`WAIT`) or not (`BUY NOW`).

Large-Scale Machine Learning (AAGE), Universidade da Coruña — Practical 1.
Jaime Morales Castro and Nicolás Pernas Gómez.

🇪🇸 [Versión en español](README.es.md) · 📄 [Full report](docs/report.pdf) · 🎤 [Defense slides](docs/defense-slides.pdf)

---

## The problem

The dataset is Expedia's *Flight Prices* (31 GB of real searches). The subset
used here is the coast-to-coast corridor — flights between {LAX, SFO} and
{JFK, EWR, BOS} in both directions, so 12 directed routes and **6,147,167
observations**, which reduce to **3,145,954 rows** after labelling and feature
engineering.

Everything runs on **Spark MLlib**, locally, in a single session.

## Three decisions that shaped the project

**1. Defining what "wait" means.** The first label depended on how many times a
given itinerary had already been observed — information that does not exist at
search time. Replacing it with a fixed seven-day window cut the survivorship
bias from **23 points to 9** (quantified in `src/analisis_supervivencia.py`).

**2. Splitting by time, never at random.** The partition is chronological by
flight date. A random split lets the model see future prices for the same
itinerary and inflates every metric.

**3. Choosing the threshold with money, not with F1.** The two are different
operating points. The economic threshold (≈ 0.15) maximises dollars saved; 0.5
maximises F1. Since almost every search is better off waiting, all the value is
concentrated in the minority of searches where buying immediately is right —
so the threshold is picked from the cost matrix in `src/evaluation.py`.

## Results

Four algorithms, ranked by AUC-PR on the held-out test set. The trees win
because the classes do not separate linearly, and GBT and Random Forest end up
statistically tied — the tie is broken economically.

| Model | AUC-PR | AUC-ROC | Savings per search |
|---|---|---|---|
| Random Forest | 0.593 | 0.730 | $19.95 |
| Gradient-Boosted Trees | 0.586 | 0.722 | $19.79 |
| Logistic regression | 0.544 | 0.677 | $19.49 |
| Decision tree | 0.507 | 0.662 | $19.32 |

Baselines: always wait, $19.32 per search; always buy now, $0.00. A random
classifier scores 0.377 AUC-PR, so the trees are roughly 60 % above chance.

The threshold sweep in [`results/barrido_umbral.csv`](results/barrido_umbral.csv)
peaks at **0.15 with $26.04 saved per search** on the final run, where F1 peaks
elsewhere. The most informative features are fare class, route, days to
departure and airline — see
[`results/importancia_variables.csv`](results/importancia_variables.csv).

## Requirements

- Python 3.11 and `pip install -r requirements.txt`
- JDK 17
- Windows only: `winutils.exe` and `hadoop.dll` for Hadoop 3.3.6 in
  `C:\hadoop\bin`, from [cdarlint/winutils](https://github.com/cdarlint/winutils)

`src/config.py` uses the `JAVA_HOME` and `HADOOP_HOME` environment variables
when they are valid and falls back to hardcoded paths otherwise. To check that
Spark starts:

```bash
python src/config.py
```

## Data

**Not included** — the original file is 31 GB. Download *Expedia Flight Prices*
from Kaggle and place it at `data/itineraries.csv`. The corridor subset is
produced by running [`notebooks/data_selection.ipynb`](notebooks/data_selection.ipynb).

## Running it

```bash
python src/data_processing.py   # label, features and partitions
python src/main.py              # train, evaluate and write results/
```

The first step builds `data/corredor.parquet` from the subset CSV if missing,
then `data/dataset_modelado.parquet`. The second takes 35–60 minutes;
`python src/main.py --muestra 0.05` does a quick smoke run.

Optional, reproduces the figures in the limitations section of the report:

```bash
python src/analisis_supervivencia.py
```

## Layout

```
src/config.py                   paths, Spark session and parameters
src/data_processing.py          selection, labelling, features and partitions
src/models.py                   pipelines, hyperparameter grid and training
src/evaluation.py               metrics, decision threshold and economic simulation
src/main.py                     full run
src/analisis_supervivencia.py   survivorship-bias quantification
notebooks/data_selection.ipynb  subset selection
notebooks/02_eda.ipynb          exploratory analysis
results/                        metrics from the final run
docs/                           report, assignment brief, defense slides, figures
```
