# prepare_data.py
from datasets import load_dataset
import librosa
import numpy as np
import pandas as pd

dataset = load_dataset("myleslinder/crema-d", trust_remote_code=True)["train"]

data = []
for sample in dataset:
    y = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]
    label = sample["label"]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
    data.append({"mfcc": mfcc.tolist(), "label": label})

df = pd.DataFrame(data)
df.to_csv("mfcc_series.csv", index=False)
print("📁 Kayıt tamamlandı: mfcc_series.csv")
