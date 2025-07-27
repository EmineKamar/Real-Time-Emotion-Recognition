import numpy as np
import torch
import sounddevice as sd
import time
from whisper_utils import transcribe, extract_embedding
from transformer_model import TransformerClassifier

SAMPLE_RATE = 16000
CHUNK_DURATION = 3  # saniye
EMBED_DIM = 512
THRESHOLD_CONFIDENCE = 0.4
EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🔄 Whisper modeli yükleniyor...")
import whisper
whisper_model = whisper.load_model("base", device=device)

print("🔄 Transformer modeli yükleniyor...")
model = TransformerClassifier(input_dim=EMBED_DIM, num_classes=len(EMOTIONS)).to(device)
model.load_state_dict(torch.load("transformer_model.pth", map_location=device))
model.eval()

def record_audio(duration_sec=CHUNK_DURATION, sample_rate=SAMPLE_RATE):
    print(f"🎙️ {duration_sec} saniyelik ses kaydediliyor, konuş lütfen...")
    audio = sd.rec(int(sample_rate * duration_sec), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def predict_emotion(embedding):
    with torch.no_grad():
        inp = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
        logits = model(inp)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        confidence = np.max(probs)
        predicted = EMOTIONS[np.argmax(probs)]
    return predicted, confidence, probs

try:
    while True:
        audio = record_audio()

        # Metin çıkar
        text = transcribe(audio, SAMPLE_RATE, model=whisper_model)

        if text == "":
            text = "[Anlaşılmadı]"

        # Embedding çıkar
        embedding = extract_embedding(audio, SAMPLE_RATE, model=whisper_model)

        # Duygu tahmini
        predicted, confidence, probs = predict_emotion(embedding)

        print(f"🗣️ Algılanan Konuşma: {text}")
        print(f"🧠 Duygu Tahmini: {predicted} (Güven: {confidence:.2f})")
        print(f"📊 Olasılık Dağılımı: {np.round(probs, 2)}\n")

except KeyboardInterrupt:
    print("\n🛑 Program sonlandırıldı.")
