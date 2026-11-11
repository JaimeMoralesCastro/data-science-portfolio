import pandas as pd
import numpy as np
import re
import os
import json
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

os.makedirs("data", exist_ok=True)

# ── OLID download ─────────────────────────────────────────────────────────────
print("Downloading OLID...")
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/idontflow/OLID/master/olid-training-v1.0.tsv",
    "data/olid_train_raw.tsv"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/idontflow/OLID/master/testset-levela.tsv",
    "data/olid_testA_raw.tsv"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/idontflow/OLID/master/labels-levela.csv",
    "data/olid_labelsA.csv"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/idontflow/OLID/master/testset-levelb.tsv",
    "data/olid_testB_raw.tsv"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/idontflow/OLID/master/labels-levelb.csv",
    "data/olid_labelsB.csv"
)
print("OLID downloaded.")

# ── Jigsaw pipeline ───────────────────────────────────────────────────────────
print("\nProcessing Jigsaw...")
jigsaw_raw = pd.read_csv("data/jigsaw_train.csv")

toxic_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
jigsaw_raw['label'] = (jigsaw_raw[toxic_cols].sum(axis=1) > 0).astype(int)
print(f"Jigsaw raw: {len(jigsaw_raw):,} rows | toxic: {jigsaw_raw['label'].sum():,} ({jigsaw_raw['label'].mean()*100:.1f}%)")

def clean_jigsaw(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z0-9\s\.\!\?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

jigsaw_raw['clean_text'] = jigsaw_raw['comment_text'].apply(clean_jigsaw)
jigsaw = jigsaw_raw[jigsaw_raw['clean_text'].str.len() > 10][['clean_text', 'label']].reset_index(drop=True)

train_val, jigsaw_test = train_test_split(jigsaw, test_size=0.1, random_state=42, stratify=jigsaw['label'])
jigsaw_train, jigsaw_val = train_test_split(train_val, test_size=0.111, random_state=42, stratify=train_val['label'])

jigsaw_train.to_csv("data/jigsaw_train.csv", index=False)
jigsaw_val.to_csv("data/jigsaw_val.csv", index=False)
jigsaw_test.to_csv("data/jigsaw_test.csv", index=False)
print(f"Jigsaw → train: {len(jigsaw_train):,} | val: {len(jigsaw_val):,} | test: {len(jigsaw_test):,}")

# learning curve subsets (10 / 25 / 50 %)
for pct in [10, 25, 50]:
    subset = jigsaw_train.sample(frac=pct/100, random_state=42).reset_index(drop=True)
    subset.to_csv(f"data/jigsaw_train_{pct}.csv", index=False)
    print(f"  Subset {pct}%: {len(subset):,} rows")

# class weights (calculados sobre el train completo)
weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=jigsaw_train['label'].values)
with open("data/class_weights.json", "w") as f:
    json.dump({"0": float(weights[0]), "1": float(weights[1])}, f)
print(f"Class weights → 0: {weights[0]:.4f} | 1: {weights[1]:.4f}")

# ── OLID pipeline ─────────────────────────────────────────────────────────────
print("\nProcessing OLID...")

def split_hashtag(match):
    tag = match.group(0).lstrip('#')
    return re.sub(r'([A-Z])', r' \1', tag).strip().lower()

def clean_olid(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\S+', 'USER', text)
    text = re.sub(r'#\S+', split_hashtag, text)
    text = text.lower()
    text = re.sub(r'[^\w\s\.\!\?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

olid_raw = pd.read_csv("data/olid_train_raw.tsv", sep='\t')
olid_raw['clean_text'] = olid_raw['tweet'].apply(clean_olid)
olid_raw['label'] = (olid_raw['subtask_a'] == 'OFF').astype(int)
olid_train = olid_raw[olid_raw['clean_text'].str.len() > 5][['clean_text', 'label', 'subtask_b', 'subtask_c']].reset_index(drop=True)
olid_train.to_csv("data/olid_train.csv", index=False)
print(f"OLID train: {len(olid_train):,} | offensive: {olid_train['label'].sum():,} ({olid_train['label'].mean()*100:.1f}%)")

olid_testA = pd.read_csv("data/olid_testA_raw.tsv", sep='\t')
olid_labelsA = pd.read_csv("data/olid_labelsA.csv", header=None, names=['id', 'subtask_a'])
olid_testA = olid_testA.merge(olid_labelsA, on='id')
olid_testA['clean_text'] = olid_testA['tweet'].apply(clean_olid)
olid_testA['label'] = (olid_testA['subtask_a'] == 'OFF').astype(int)

olid_testB = pd.read_csv("data/olid_testB_raw.tsv", sep='\t')
olid_labelsB = pd.read_csv("data/olid_labelsB.csv", header=None, names=['id', 'subtask_b'])
olid_testB = olid_testB.merge(olid_labelsB, on='id')[['id', 'subtask_b']]

olid_test = olid_testA.merge(olid_testB, on='id', how='left')
olid_test = olid_test[olid_test['clean_text'].str.len() > 5][['clean_text', 'label', 'subtask_b']].reset_index(drop=True)
olid_test.to_csv("data/olid_test.csv", index=False)
print(f"OLID test:  {len(olid_test):,} | offensive: {olid_test['label'].sum():,} ({olid_test['label'].mean()*100:.1f}%)")

# ── Mixed dataset ─────────────────────────────────────────────────────────────
print("\nBuilding mixed dataset...")
mixed = pd.concat([
    jigsaw_train[['clean_text', 'label']],
    olid_train[['clean_text', 'label']]
]).sample(frac=1, random_state=42).reset_index(drop=True)
mixed.to_csv("data/mixed_train.csv", index=False)
print(f"Mixed train: {len(mixed):,} rows")

# ── Vocabulary overlap ────────────────────────────────────────────────────────
print("\nComputing vocabulary overlap...")
jigsaw_vocab = set(' '.join(jigsaw_train['clean_text']).split())
olid_vocab   = set(' '.join(olid_train['clean_text']).split())
overlap_pct  = len(jigsaw_vocab & olid_vocab) / len(olid_vocab) * 100
print(f"Jigsaw vocab: {len(jigsaw_vocab):,} tokens")
print(f"OLID vocab:   {len(olid_vocab):,} tokens")
print(f"OLID tokens present in Jigsaw vocab: {overlap_pct:.1f}%")

print("\nAll files saved to data/. Done!")
