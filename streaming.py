#streaming.py
import sys
sys.path.append("C:/Users/emina/OneDrive/Masaüstü/vscode/gemma")

from matrix import weighted_ensemble, emotion_memory, transition_matrix

if __name__ == "__main__":
    # Örnek ses analizi sonuçları (duygu, güven)
    streaming_predictions = [
        ("NEU", 0.9),
        ("SAD", 0.4),
        ("ANG", 0.7),
        ("HAP", 0.6),
        ("DIS", 0.3),
        ("FEA", 0.9),
        ("NEU", 0.8),
    ]

    print("Başlangıç belleği boş.")
    for idx, (emo, conf) in enumerate(streaming_predictions):
        final_emo, probs = weighted_ensemble(emo, conf, transition_matrix, emotion_memory, alpha=0.7)
        print(f"Adım {idx+1} — Anlık: {emo} (güven {conf:.2f}), Nihai: {final_emo}, Olasılıklar: {probs.round(3)}")
