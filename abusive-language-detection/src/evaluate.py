import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

from train import TextDataset

tokenizer = RobertaTokenizer.from_pretrained('models/roberta-jigsaw')
model     = RobertaForSequenceClassification.from_pretrained('models/roberta-jigsaw')
model.eval()

test_df     = pd.read_csv("data/processed/test.csv")
test_loader = DataLoader(TextDataset(test_df, tokenizer), batch_size=8)

preds, labels = [], []
with torch.no_grad():
    for batch in test_loader:
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        preds.extend(torch.argmax(outputs.logits, dim=1).numpy())
        labels.extend(batch['label'].numpy())

print(f"Test Acc: {accuracy_score(labels, preds):.4f} | Test F1: {f1_score(labels, preds):.4f}")
print(classification_report(labels, preds, target_names=['Clean', 'Toxic']))

os.makedirs("results", exist_ok=True)
cm = confusion_matrix(labels, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Clean', 'Toxic'], yticklabels=['Clean', 'Toxic'])
plt.ylabel('Real')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig("results/confusion_matrix.png")
plt.close()