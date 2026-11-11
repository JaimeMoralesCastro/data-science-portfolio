# Sonar mine classification

**Mine or rock?** 208 sonar returns, 60 frequency bands each, and five families
of models compared under a risk criterion that is deliberately asymmetric:
mistaking a mine for a rock costs far more than the reverse.

Machine Learning I (AAI), Universidade da Coruña. Written in **Julia**.

📄 [Full report](docs/report.pdf) · 🧪 [Cross-validation output](results/cv_results.txt)

---

## Why accuracy is the wrong metric here

A classifier that calls everything a rock still scores respectably on accuracy
on this dataset, and would also clear a minefield by declaring it empty. So the
models are selected by **sensitivity (recall on the mine class) and F1**, and
every configuration is reported with its full confusion matrix.

Model selection uses k-fold cross-validation over the complete
[UCI Sonar](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks)
dataset, with a fixed fold assignment (`data/cv_indices.dat`) shared across model
families so the comparison is paired rather than coincidental.

## Results

Best configuration per family, averaged across folds:

| Model | Accuracy | Sensitivity (mines) | F1 |
|---|---|---|---|
| **SVM, polynomial kernel** (C=5, degree 3) | **0.831** | **0.784** | **0.813** |
| DoME (30 nodes) | 0.807 | 0.764 | 0.791 |
| k-NN (k=5) | 0.813 | 0.743 | 0.788 |
| Neural network (32 hidden units) | 0.789 | 0.714 | 0.761 |
| Decision tree (depth 8) | 0.726 | 0.711 | 0.707 |

The polynomial SVM wins on both criteria at once, which is the uncommon case:
it is the most accurate *and* the one that misses the fewest mines. The linear
and RBF kernels collapse to ~0.55 sensitivity, which is the clearest signal in
the whole study that the classes are not linearly separable in the raw band
space.

Per-configuration metrics, standard deviations and confusion matrices are in
[`results/cv_results.txt`](results/cv_results.txt); the SVM matrices are broken
out in [`results/svm_confusion_matrices.txt`](results/svm_confusion_matrices.txt).

## Requirements

Julia with `CSV`, `DataFrames`, `Flux`, `Statistics`, `Plots`, `StatsPlots`,
`PrettyTables`, `Serialization` and `ScikitLearn` (which bridges to Python's
scikit-learn for SVM, tree and k-NN).

```julia
using Pkg
Pkg.add(["CSV","DataFrames","Flux","Plots","StatsPlots","PrettyTables","ScikitLearn"])
```

## Running it

```bash
julia src/main.jl
```

The dataset is small enough to be included, under `data/sonar/`.

## Layout

```
src/main.jl                  full project: load, train, cross-validate, report
src/model_comparison.jl      the five model families and their hyperparameter grids
src/revised_submission.jl    reworked version submitted for the retake
src/course_utils.jl          course-provided helpers (encoding, holdout, CV, metrics)
src/lineplot.R               R script for the model comparison line plot
data/sonar/                  UCI Sonar dataset (mines vs rocks)
results/                     cross-validation metrics, confusion matrices, figures
docs/                        report (PDF + LaTeX source), figures, reference paper
```

`src/course_utils.jl` and `src/model_comparison.jl` come from the course's
paired practical sessions; the project write-up and the revised submission are
the deliverables of this work.

## Reference

Gorman, R. P. and Sejnowski, T. J. (1988). *Analysis of hidden units in a
layered network trained to classify sonar targets.* Included as
[`docs/reference-paper-1988.pdf`](docs/reference-paper-1988.pdf).
