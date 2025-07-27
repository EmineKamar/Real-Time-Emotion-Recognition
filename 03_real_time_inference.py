import sounddevice as sd
import numpy as np
import librosa
import time
from collections import deque

import speech_recognition as sr

# Yükle
transition_matrix = np.load("transition_matrix.npy")
states = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
state_to_idx = {s: i for i, s in enumerate(states)}
emotion_memory = deque(maxlen=5)

# Ensemble hesaplama
def weighted_ensemble(current_emotion, confidence, transition_matrix, memory, alpha=0.7):
    cur_idx = state_to_idx[current_emotion]
    markov_probs = transition_matrix[cur_idx]

    past_counts = np.zeros(len(states))
    for emo in memory:
        past_counts[state_to_idx[emo]] += 1
    past_distribution = past_counts / len(memory) if memory else np.ones(len(states)) / len(states)
    markov_past_avg = (markov_probs + past_distribution) / 2

    instant_vector = np.zeros(len(states))
    instant_vector[cur_idx] = 1

    combined_probs = alpha * instant_vector + (1 - alpha) * markov_past_avg
    final_idx = np.argmax(combined_probs)
    final_emotion = states[final_idx]
    memory.append(final_emotion)
    return final_emotion, combined_probs

# Dummy model (gerçek modelle değiştirilebilir)
def dummy_model_predict(mfcc):
    idx = np.random.choice(len(states))
    conf = np.random.uniform(0.5, 1.0)
    return states[idx], conf

def extract_mfcc(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

def main():
    print("🎙️ Gerçek zamanlı duygu tanıma başlatıldı. Ctrl+C ile çık.")
    while True:
        print("\n🔴 Ses kaydediliyor...")
        audio = sd.rec(int(2 * 22050), samplerate=22050, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()
        mfcc = extract_mfcc(audio, 22050)
        emo, conf = dummy_model_predict(mfcc)
        final, probs = weighted_ensemble(emo, conf, transition_matrix, emotion_memory)
        print(f"🧠 Tahmin: {emo} (güven: {conf:.2f}) → Nihai: {final}")
        print(f"📊 Dağılım: {probs.round(2)}")
        time.sleep(1)

if __name__ == "__main__":
    main()
