# Data Science Portfolio

Four self-contained projects in machine learning, signal processing and network
simulation, built between 2024 and 2026 during my BSc in **Data Science and
Engineering** (Universidade da Coruña) and my Erasmus+ exchange at
**Politehnica University of Bucharest**.

Each folder is an independent project with its own code, results and written
report. Raw datasets are not committed — every project documents where to get
them and how to reproduce the run.

🇪🇸 [Versión en español](README.es.md)

---

## Projects

| Project | What it answers | Stack |
|---|---|---|
| [**Buy Now or Wait**](buy-now-or-wait/) | Given a flight search, will the price drop at least 5 % within a week? Four models compared on 6.1 M real Expedia searches, with the decision threshold chosen by dollars saved instead of F1. | PySpark · MLlib · GBT · Parquet |
| [**Sonar mine classification**](sonar-mine-classification/) | Mine or rock? Five model families compared under an asymmetric risk criterion, where a missed mine costs far more than a false alarm. | Julia · SVM · ANN · k-NN · DoME |
| [**Abusive language detection**](abusive-language-detection/) | Does a toxicity model trained on Wikipedia survive Twitter? A 3 × 3 train/test design says: not entirely. | PyTorch · RoBERTa · Hugging Face |
| [**Rumor spread simulation**](rumor-spread-sir-simulation/) | How does network shape change the speed of a rumor? Agent-based SIR on scale-free vs small-world graphs. | Python · NetworkX · SIR |

---

## What these projects have in common

- **The decision comes before the metric.** In the flight model the operating
  threshold is chosen by expected savings; in the sonar model by sensitivity,
  because a false negative is a mine on the seabed. Accuracy is rarely the
  quantity that matters.
- **Validation that matches reality.** Temporal splits by flight date instead of
  random splits, cross-validation on small samples, and out-of-domain test sets
  when the question is generalisation.
- **Leakage hunted deliberately.** The flight project documents how the first
  label definition leaked information unavailable at search time, and how a
  fixed window cut that bias from 23 points to 9.

---

## Repository layout

```
buy-now-or-wait/              Spark pipeline, notebooks, metrics, report
sonar-mine-classification/    Julia scripts, Sonar dataset, confusion matrices
abusive-language-detection/   RoBERTa training and evaluation, 3x3 results
rumor-spread-sir-simulation/  Agent-based model, plots, report
```

Each project has its own `README.md` with requirements, how to obtain the data
and how to run it.

## Tech used across the portfolio

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Julia](https://img.shields.io/badge/Julia-9558B2?style=flat-square&logo=julia&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C?style=flat-square&logo=apachespark&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![NetworkX](https://img.shields.io/badge/NetworkX-2C5BB4?style=flat-square)

---

## Author

**Jaime Morales Castro** — Data Science and Engineering, Universidade da Coruña ·
Erasmus+ at Politehnica University of Bucharest.
[LinkedIn](https://linkedin.com/in/jaime-morales-castro)

Academic work is credited to its co-authors inside each project's README.

## License

Code is released under the [MIT License](LICENSE). Written reports, slides and
third-party datasets keep their own terms; the referenced datasets (Expedia
Flight Prices, UCI Sonar, Jigsaw, OLID) are **not** redistributed here.
