import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score
import os

class TextDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df['clean_text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=128, padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

from tqdm import tqdm

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['label'])
        outputs.loss.backward()
        optimizer.step()
        total_loss += outputs.loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            preds.extend(torch.argmax(outputs.logits, dim=1).numpy())
            labels.extend(batch['label'].numpy())
    return accuracy_score(labels, preds), f1_score(labels, preds)

train_df = pd.read_csv("data/processed/train.csv")
val_df   = pd.read_csv("data/processed/val.csv")

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

train_loader = DataLoader(TextDataset(train_df, tokenizer), batch_size=8, shuffle=True)
val_loader   = DataLoader(TextDataset(val_df, tokenizer),   batch_size=8)

model     = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-5)

EPOCHS = 1
for epoch in range(EPOCHS):
    loss = train_epoch(model, train_loader, optimizer)
    acc, f1 = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

os.makedirs("models", exist_ok=True)
model.save_pretrained("models/roberta-jigsaw")
tokenizer.save_pretrained("models/roberta-jigsaw")
print("Model saved.")