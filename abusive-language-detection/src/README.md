# Abusive Language Detection in Social Networks
Jaime Morales — Milestone 3

## Requirements

```
pip install torch transformers pandas scikit-learn matplotlib seaborn tqdm datasets
```

## How to run

The project has 3 scripts that must be run in order on Google Colab.

**1. preprocess.py**
Downloads and prepares the data. You need to upload the Jigsaw dataset from Kaggle when prompted. The OLID dataset is downloaded automatically from HuggingFace.

**2. train.py**
Trains 3 RoBERTa-base models: one on Jigsaw data, one on OLID data, and one on both combined. Models are saved to the models/ folder.

**3. evaluate.py**
Loads the 3 trained models and evaluates them on the test sets. Prints the results table.

## Notes

- Designed to run on Google Colab with T4 GPU
- Training takes around 90 minutes per model with 3 epochs
