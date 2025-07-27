import os
import numpy as np
import torch
import torchaudio
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from transformers import AutoProcessor, AutoModel

# GPU kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Cihaz: {device}")

# Whisper modeli ve işlemcisi
processor = AutoProcessor.from_pretrained("openai/whisper-base")
model = AutoModel.from_pretrained("openai/whisper-base", trust_remote_code=True).to(device)
model.eval()

# Dataset yükle
dataset = load_dataset("myleslinder/crema-d", trust_remote_code=True)["train"]

# Duygu etiket eşlemesi
label_to_emotion = {
    0: "NEU",  # Neutral
    1: "HAP",  # Happy
    2: "SAD",  # Sad
    3: "ANG",  # Angry
    4: "FEA",  # Fear
    5: "DIS",  # Disgust
}

# Veri hazırlama
features = []
labels = []

print("🎧 Embedding çıkarımı başlıyor...")
for sample in tqdm(dataset):
    audio = sample["audio"]
    waveform = torch.tensor(audio["array"]).float().unsqueeze(0)
    sampling_rate = audio["sampling_rate"]

    # Whisper için yeniden örnekleme
    if sampling_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sampling_rate, 16000)

    # Whisper input
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        out = model(input_values)
        embedding = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # [embedding_dim]

    features.append(embedding)
    labels.append(label_to_emotion[sample["label"]])

# Kayıt
os.makedirs("data", exist_ok=True)
np.save("data/whisper_embeddings.npy", np.array(features))

df = pd.DataFrame({
    "label": labels
})
df.to_csv("data/labels.csv", index=False)

print("✅ Embedding çıkarımı tamamlandı. Kaydedilenler:")
print("- data/whisper_embeddings.npy")
print("- data/labels.csv")
