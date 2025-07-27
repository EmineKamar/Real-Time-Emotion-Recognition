# train_model.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import librosa
import pandas as pd
import numpy as np
from model import EmotionCNN

LABEL_MAP = {
    0: "NEU", 1: "HAP", 2: "SAD", 3: "ANG", 4: "FEA", 5: "DIS"
}

MAX_LEN = 100  # max frame count

class MFCCDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = []
        self.y = []
        for _, row in df.iterrows():
            mfcc = np.array(eval(row["mfcc"]))  # string'den array'e
            if mfcc.shape[0] < MAX_LEN:
                pad_width = MAX_LEN - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad_width), (0,0)), mode='constant')
            else:
                mfcc = mfcc[:MAX_LEN]
            self.X.append(mfcc)
            self.y.append(row["label"])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), self.y[idx]

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MFCCDataset("mfcc_series.csv")
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = EmotionCNN(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(15):
        model.train()
        total_loss = 0
        correct = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = correct / len(dataset)
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Acc={acc:.2%}")

    torch.save(model.state_dict(), "cnn1d_emotion.pth")
    print("✅ Model kaydedildi: cnn1d_emotion.pth")

if __name__ == "__main__":
    train()
