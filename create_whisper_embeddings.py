import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import whisper
from tqdm import tqdm

# Ayarlar
audio_dir = r"C:\Users\emina\OneDrive\Masaüstü\vscode\gemma\data\AudioWAV"
label_file = r"C:\Users\emina\OneDrive\Masaüstü\vscode\gemma\data\labels.csv"
output_csv = "whisper_embeddings.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Modeli yükle
model = whisper.load_model("base", device=device)

# Label dosyasını oku
df = pd.read_csv(label_file)

# Embedding sonuçlarını tutacak liste
all_embeddings = []

# Sabit ses uzunluğu: 30 saniye, 16 kHz örnekleme
expected_len = 16000 * 30

for i, row in tqdm(df.iterrows(), total=len(df)):
    audio_path = row["path"]
    try:
        # Ses dosyasını yükle
        audio, sr = torchaudio.load(audio_path)  # (kanal, örnek_sayısı)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        # Stereo ise mono'ya indir
        audio = audio.mean(dim=0).numpy()

        # Ses uzunluğunu sabitle (kırp veya pad et)
        if len(audio) > expected_len:
            audio = audio[:expected_len]
        elif len(audio) < expected_len:
            padding = expected_len - len(audio)
            audio = np.pad(audio, (0, padding))

        # Whisper için mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(device)

        with torch.no_grad():
            embed = model.encoder(mel.unsqueeze(0))  # [1, frames, 768]
            mean_embed = embed.mean(dim=1).squeeze().cpu().numpy()

        # Embedding'i dict olarak hazırla
        embed_dict = {str(idx): mean_embed[idx] for idx in range(len(mean_embed))}
        embed_dict["emotion"] = row["emotion"]  # etiketi ekle

        all_embeddings.append(embed_dict)

    except Exception as e:
        print(f"❌ {audio_path}: {e}")

# Sonuçları dataframe yap ve csv olarak kaydet
out_df = pd.DataFrame(all_embeddings)
out_df.to_csv(output_csv, index=False)

print(f"✅ İşlem tamamlandı. Embeddingler {output_csv} dosyasına kaydedildi.")
