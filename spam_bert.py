import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
    get_scheduler,
)

from util.open_data import open_data

# %% ===============================
# CARGAR DATOS
# ===============================

# reproducibilidad
seed = 123
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

df = open_data()
df = df[df["message_length"] < 5000]
df["label"] = df["Spam/Ham"].map({"ham": 0, "spam": 1})
df = df[["text", "label"]].dropna()
# df.head()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Train/test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=seed,
    stratify=df["label"],
)
train_texts.head()
df.info()


# %% ===============================
# TOKENIZACION
# ===============================

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def tokenize_data(texts, labels, tokenizer, max_len=128):
    encodings = tokenizer(
        texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt"
    )
    return encodings, torch.tensor(labels)


train_encodings, train_labels_tensor = tokenize_data(
    train_texts, train_labels, tokenizer
)
test_encodings, test_labels_tensor = tokenize_data(test_texts, test_labels, tokenizer)


# %% ===============================
# OBJETO DATASET
# ===============================
class SpamDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()} | {
            "labels": self.labels[idx]
        }


train_dataset = SpamDataset(train_encodings, train_labels_tensor)
test_dataset = SpamDataset(test_encodings, test_labels_tensor)

# %%===============================
# CARGAR MODELO
# ===============================
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
# model.to(device)

# Modelo DistilBERT, es más pequeño y rápido de entrenar
model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased")
model.to(device)

# %%===============================
# TRAINING SETUP
# ===============================
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * 4,  # epochs=4
)


# %%===============================
# TRAINING LOOP
# ===============================
def train(model, train_loader, optimizer, scheduler, epochs=4):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} loss: {avg_loss:.4f}")


train(model, train_loader, optimizer, lr_scheduler)

# Alternativa más optimizada que el training loop anterior
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    fp16=True,  # mixed precision
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()


# %%===============================
# EVALUACION
# ===============================
def evaluate(model, test_loader):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluando"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, axis=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())

    return np.array(predictions), np.array(true_labels)


y_pred, y_true = evaluate(model, test_loader)

# ===============================
# METRICAS
# ===============================
print(classification_report(y_true, y_pred, target_names=["Ham", "Spam"]))

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"],
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("bert_matriz_confusion.png")
plt.show()

# ===============================
# Guardar modelo
# ===============================
output_dir = "./bert_spam_model"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")


# ===============================
# Función de predicción
# ===============================
def predict_spam(texts, model, tokenizer):
    model.eval()
    inputs = tokenizer(
        texts, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, axis=1)
    return ["Ham" if p == 0 else "Spam" for p in preds]


# Ejemplo
print(predict_spam(["Win a free iPhone!", "Meeting at 10am."]))
