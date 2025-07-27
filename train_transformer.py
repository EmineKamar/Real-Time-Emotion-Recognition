import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from transformer_model import TransformerClassifier  # Kendi model dosyan

# Ayarlar
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Veriyi yükle
df = pd.read_csv("whisper_embeddings.csv")

# Embedding kolonlarını bul
embedding_cols = [col for col in df.columns if col.isdigit()]
EMBEDDING_DIM = len(embedding_cols)
print(f"EMBEDDING_DIM = {EMBEDDING_DIM}")

# Etiket sütununu seç
label_col = "emotion" if "emotion" in df.columns else "label"
y = LabelEncoder().fit_transform(df[label_col])
X = df[embedding_cols].values.astype("float32")

# Tensor'lara çevir
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# Dataset ve eğitim/validasyon split
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# Model oluştur
model = TransformerClassifier(input_dim=EMBEDDING_DIM, num_classes=len(set(y))).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# Eğitim döngüsü
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device).long()  # <--- burada long() önemli!
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    correct = total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device).long()  # <--- burada da long()
            out = model(xb)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f} - Val Acc: {acc*100:.2f}%")

# Modeli kaydet
torch.save(model.state_dict(), "transformer_model.pth")

# Sınıflandırma raporu
print("\n🔍 Sınıflandırma Raporu:")
print(classification_report(all_labels, all_preds))
