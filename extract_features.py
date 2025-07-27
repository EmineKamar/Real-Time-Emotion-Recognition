# extract_features.py
from datasets import load_dataset
import librosa
import numpy as np
import pandas as pd

dataset = load_dataset("myleslinder/crema-d", trust_remote_code=True)["train"]

def extract_mfcc(audio_array, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

features = []
for sample in dataset:
    audio_array = sample['audio']['array']
    sr = sample['audio']['sampling_rate']
    mfcc_feat = extract_mfcc(audio_array, sr)
    label = sample['label']  # ✔️ düzeltildi
    features.append([*mfcc_feat, label])

columns = [f"mfcc_{i}" for i in range(13)] + ["label"]
df = pd.DataFrame(features, columns=columns)
df.to_csv("crema_mfcc_features.csv", index=False)
print("Özellikler kaydedildi ✅")
