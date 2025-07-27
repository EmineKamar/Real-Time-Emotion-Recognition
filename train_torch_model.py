import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import joblib

# GPU kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Cihaz: {device}")

# Etiketleri sayısal olarak kodlayalım
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("crema_mfcc_features.csv")
X = df.drop("label", axis=1).values.astype(np.float32)
y = df["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, "label_encoder.pkl")  # modelden sonra kullanılacak

# Veri kümesi
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = EmotionDataset(X, y_encoded)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
class EmotionCNN(nn.Module):
    def __init__(self, input_dim=13, num_classes=6):
        super(EmotionCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 13)
        return self.net(x)

model = EmotionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Eğitim
EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"🔁 Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss/len(train_loader):.4f}")

# Modeli kaydet
torch.save(model.state_dict(), "emotion_cnn.pth")
print("✅ Model kaydedildi: emotion_cnn.pth")
