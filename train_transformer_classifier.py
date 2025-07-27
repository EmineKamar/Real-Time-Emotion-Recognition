import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Cihaz: {device}")

# Veri yükle
X = np.load("data/whisper_embeddings.npy")
y = pd.read_csv("data/labels.csv")["label"].values

# Etiket encode
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Dataset tanımı
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = EmotionDataset(X, y_encoded)

# Train/val ayrımı
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)

# Transformer modeli
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.embedding_proj = nn.Linear(input_dim, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.embedding_proj(x).unsqueeze(1)  # [B, 1, 128]
        x = self.encoder(x)                      # [B, 1, 128]
        out = self.classifier(x)
        return out.squeeze(1)

# Model başlat
input_dim = X.shape[1]
num_classes = len(np.unique(y_encoded))
model = TransformerClassifier(input_dim=input_dim, num_classes=num_classes).to(device)

# Kayıp ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Eğitim
EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    train_loss, train_preds, train_labels = 0, [], []
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_preds.extend(torch.argmax(output, dim=1).cpu().numpy())
        train_labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(train_labels, train_preds)
    print(f"📚 Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f} - Acc: {acc*100:.2f}%")

    # Validation
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            val_output = model(X_val)
            val_preds.extend(torch.argmax(val_output, dim=1).cpu().numpy())
            val_labels.extend(y_val.cpu().numpy())
    val_acc = accuracy_score(val_labels, val_preds)
    print(f"✅ Validation Accuracy: {val_acc*100:.2f}%\n")

# Model kaydet
torch.save(model.state_dict(), "transformer_emotion.pth")
print("💾 Model kaydedildi: transformer_emotion.pth")
