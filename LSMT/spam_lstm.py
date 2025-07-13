import os
import re
import torch
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# Cargar dataset 
scrip_dir = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(scrip_dir,"enron_spam_data.csv")

df = pd.read_csv(csv_path)

df['text'] = df['Subject'].fillna('') + ' ' + df['Message'].fillna('')

df = df[['text','Spam/Ham']]
df['message_length'] = df['text'].apply(len)

df['Label'] = df['Spam/Ham'].map(
    {
        'ham':0,
        'spam':1
    }
)

# Descargar recursos y preprocesamiento
nltk.download('stopwords')
nltk.download('punkt_tab', force=True)
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    filtered = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return filtered

df['tokens'] = df['text'].apply(preprocess)

#Crear vocavulario
counter = Counter()
for tokens in df['tokens']:
    counter.update(tokens)

vocab = {word: i+2 for i, (word, _) in enumerate(counter.most_common(10000))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

def encode(tokens):
    return [vocab.get(w, 1) for w in tokens]  # 1 = <UNK>

df['input_ids'] = df['tokens'].apply(encode)

# Dataset y DataLoader
class SpamDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = [torch.tensor(x, dtype=torch.long) for x in inputs]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def collate_fn(batch):
    inputs, labels = zip(*batch)
    padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return padded, labels

X_train, X_test, y_train, y_test = train_test_split(df['input_ids'], df['Label'], test_size=0.2, random_state=42)

train_dataset = SpamDataset(X_train.tolist(), y_train.tolist())
test_dataset = SpamDataset(X_test.tolist(), y_test.tolist())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

# Definir el modelo
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out.squeeze()

# Entrenar  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(vocab_size=len(vocab), embed_dim=64, hidden_dim=128).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Evaluar
from sklearn.metrics import accuracy_score, confusion_matrix

model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        outputs = model(x_batch).cpu().numpy()
        preds = (outputs > 0.5).astype(int)
        y_true.extend(y_batch.numpy())
        y_pred.extend(preds)

acc = accuracy_score(y_true, y_pred)
print("Test Score:{:.2f}%".format(acc*100))
print(f"Test Accuracy: {acc:.4f}")

cm = confusion_matrix(y_test, y_pred)
labels = ["Ham", "Spam"]  

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
