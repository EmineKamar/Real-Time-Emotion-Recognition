import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperModel

# Whisper modeli ve cihaz ayarı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperModel.from_pretrained("openai/whisper-tiny").to(device)
model.eval()

# CREMA-D veri setini yükle
dataset = load_dataset("myleslinder/crema-d", trust_remote_code=True)["train"]

embeddings = []
labels = []

print("🔍 Ses örneklerinden embedding çıkarılıyor...")

for sample in tqdm(dataset):
    try:
        audio_array = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]
        label = sample["label"]

        # Whisper 16kHz bekliyor
        if sampling_rate != 16000:
            audio_array = np.array(audio_array)
            audio_array = torch.tensor(audio_array).unsqueeze(0)
            audio_array = torch.nn.functional.interpolate(audio_array.unsqueeze(0), size=int(16000 * len(audio_array[0]) / sampling_rate), mode="linear").squeeze().numpy()

        # Whisper için işlem
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            outputs = model.encoder(inputs).last_hidden_state
            embed = torch.mean(outputs, dim=1).squeeze().cpu().numpy()

        embeddings.append(embed)
        labels.append(label)

    except Exception as e:
        print("⛔ Hata:", e)
        continue

# CSV dosyasına kaydet
df = pd.DataFrame(embeddings)
df["label"] = labels
df.to_csv("whisper_embeddings.csv", index=False)
print("✅ whisper_embeddings.csv başarıyla oluşturuldu.")
