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
df_sample = df.sample(frac=0.2, random_state=42)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Sample Train/test split
train_texts_sample, test_texts_sample, train_labels_sample, test_labels_sample = (
    train_test_split(
        df_sample["text"].tolist(),
        df_sample["label"].tolist(),
        test_size=0.2,
        random_state=seed,
        stratify=df_sample["label"],
    )
)
df_sample.info()


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


# Tokenizacion muestra
train_encodings_sample, train_labels_tensor_sample = tokenize_data(
    train_texts_sample, train_labels_sample, tokenizer
)
test_encodings_sample, test_labels_tensor_sample = tokenize_data(
    test_texts_sample, test_labels_sample, tokenizer
)


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


# Muestra
train_dataset_sample = SpamDataset(train_encodings_sample, train_labels_tensor_sample)
test_dataset_sample = SpamDataset(test_encodings_sample, test_labels_tensor_sample)

# %%===============================
# CARGAR MODELO
# ===============================
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Modelo DistilBERT, es más pequeño y rápido de entrenar
model = BertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
model.to(device)

# ===============================
# TRAINING
# ===============================

# %%
# 1. Batch_size = 16, epochs = 2, learning rate = 5e-5
##
train_loader_sample_16 = DataLoader(train_dataset_sample, batch_size=16, shuffle=True)
test_loader_sample_16 = DataLoader(test_dataset_sample, batch_size=16)

optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

lr_scheduler_1 = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader_sample_16) * 2,  # epochs=2
)


def train(model, train_loader, optimizer, scheduler, epochs=2):
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


# train(model, train_loader_sample_16, optimizer, lr_scheduler_1, epochs=2)

y_pred_1, y_true_1 = evaluate(model, test_loader_sample_16)

print(classification_report(y_true_1, y_pred_1, target_names=["Ham", "Spam"]))


# Matriz de confusión
cm = confusion_matrix(y_true_1, y_pred_1)
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


output_dir = "./bert_spam_model"
os.makedirs(output_dir, exist_ok=True)
# model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# %%
# 2. Batch_size = 32, epochs = 3, learning rate = 5e-5
##

model = BertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
model.to(device)

train_loader_sample_32 = DataLoader(train_dataset_sample, batch_size=32, shuffle=True)
test_loader_sample_32 = DataLoader(test_dataset_sample, batch_size=32)

optimizer_2 = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

lr_scheduler_2 = get_scheduler(
    name="linear",
    optimizer=optimizer_2,
    num_warmup_steps=0,
    num_training_steps=len(train_loader_sample_32) * 3,  # epochs=2
)

train(model, train_loader_sample_32, optimizer_2, lr_scheduler_2, epochs=3)

y_pred_2, y_true_2 = evaluate(model, test_loader_sample_32)

print(classification_report(y_true_2, y_pred_2, target_names=["Ham", "Spam"]))

# os.makedirs(output_dir, exist_ok=True)
# model.save_pretrained(output_dir)


# Matriz de confusión
cm_2 = confusion_matrix(y_true_2, y_pred_2)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm_2,
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

# model.save_pretrained(output_dir)


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
print(predict_spam(["Win a free iPhone!", "Meeting at 10am."], model, tokenizer))

# %%
# 3. Batch_size = 32, epochs = 2, learning rate = 1e-4
##
model = BertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
model.to(device)

train_loader_sample_32 = DataLoader(train_dataset_sample, batch_size=32, shuffle=True)
test_loader_sample_32 = DataLoader(test_dataset_sample, batch_size=32)

optimizer_3 = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

lr_scheduler_3 = get_scheduler(
    name="linear",
    optimizer=optimizer_3,
    num_warmup_steps=0,
    num_training_steps=len(train_loader_sample_32) * 2,  # epochs=2
)

# train(model, train_loader_sample_32, optimizer_3, lr_scheduler_3, epochs=2)

y_pred_3, y_true_3 = evaluate(model, test_loader_sample_32)

print(classification_report(y_true_3, y_pred_3, target_names=["Ham", "Spam"]))


# Matriz de confusión
cm_3 = confusion_matrix(y_true_3, y_pred_3)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm_3,
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


####
## Congelando capas intermedias
####

# %%
# 2. Batch_size = 32, epochs = 3, learning rate = 5e-5
##

model = BertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
model.to(device)

train_loader_sample_32 = DataLoader(train_dataset_sample, batch_size=32, shuffle=True)
test_loader_sample_32 = DataLoader(test_dataset_sample, batch_size=32)

optimizer_2 = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

lr_scheduler_2 = get_scheduler(
    name="linear",
    optimizer=optimizer_2,
    num_warmup_steps=0,
    num_training_steps=len(train_loader_sample_32) * 3,  # epochs=2
)

# Se congelan las capas intermedias
for param in model.bert.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

# train(model, train_loader_sample_32, optimizer_2, lr_scheduler_2, epochs=3)

y_pred_2, y_true_2 = evaluate(model, test_loader_sample_32)

print(classification_report(y_true_2, y_pred_2, target_names=["Ham", "Spam"]))

# os.makedirs(output_dir, exist_ok=True)
# model.save_pretrained(output_dir)


# Matriz de confusión
cm_2 = confusion_matrix(y_true_2, y_pred_2)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm_2,
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

# model.save_pretrained(output_dir)


###########################
## Datos completos
###########################

model = BertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
model.to(device)

# Train/test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=seed,
    stratify=df["label"],
)
df.info()


# Tokenizacion datos
train_encodings, train_labels_tensor = tokenize_data(
    train_texts, train_labels, tokenizer
)
test_encodings, test_labels_tensor = tokenize_data(test_texts, test_labels, tokenizer)

# Datos completos
train_dataset = SpamDataset(train_encodings, train_labels_tensor)
test_dataset = SpamDataset(test_encodings, test_labels_tensor)

train_loader_32 = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader_32 = DataLoader(test_dataset, batch_size=32)

optimizer_4 = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

lr_scheduler_4 = get_scheduler(
    name="linear",
    optimizer=optimizer_4,
    num_warmup_steps=0,
    num_training_steps=len(train_loader_32) * 3,  # epochs=2
)

## MUY LENTO
# train(model, train_loader_32, optimizer_4, lr_scheduler_4, epochs=3)
