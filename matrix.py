#matrix.py
import numpy as np
from collections import deque

states = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
state_to_idx = {s: i for i, s in enumerate(states)}

# Markov modeli geçiş matrisi (örnek, kendi matrisinle değiştir)
transition_matrix = np.array([
    [0.14, 0.85, 0.001, 0.001, 0.001, 0.001],
    [0.001, 0.14, 0.85, 0.001, 0.001, 0.001],
    [0.001, 0.001, 0.14, 0.85, 0.001, 0.001],
    [0.001, 0.001, 0.001, 0.14, 0.85, 0.002],
    [0.0001,0.0001,0.0001,0.0001,0.9995,0.0001],
    [0.85, 0.0001,0.0001,0.0001,0.0001,0.15],
])

# Bellek kapasitesi: geçmiş kaç tahmin saklansın
MEMORY_SIZE = 5

# Belleği tutan yapı (FIFO queue)
emotion_memory = deque(maxlen=MEMORY_SIZE)

def weighted_ensemble(current_emotion, confidence, transition_matrix, memory, alpha=0.7):
    """
    current_emotion: anlık tahmin
    confidence: anlık tahmin güveni
    transition_matrix: Markov geçiş matrisi
    memory: deque tipi geçmiş tahminler
    alpha: anlık tahmine ağırlık
    """
    # Geçerli durum
    cur_idx = state_to_idx[current_emotion]

    # Markov tabanlı tahmin olasılıklarını al
    markov_probs = transition_matrix[cur_idx]

    # Geçmiş durumların frekansını hesapla
    past_counts = np.zeros(len(states))
    for emo in memory:
        past_counts[state_to_idx[emo]] += 1
    if len(memory) > 0:
        past_distribution = past_counts / len(memory)
    else:
        past_distribution = np.ones(len(states)) / len(states)  # eşit olasılık

    # Ensemble hesaplama: 
    # anlık tahmine alpha ağırlık ver, Markov + geçmişe (1-alpha) ağırlık ver
    combined_probs = np.zeros(len(states))

    # Anlık tahmin için one-hot vektör
    instant_vector = np.zeros(len(states))
    instant_vector[cur_idx] = 1

    # Markov + geçmiş dağılımı ortalaması
    markov_past_avg = (markov_probs + past_distribution) / 2

    # Toplam ensemble
    combined_probs = alpha * instant_vector + (1 - alpha) * markov_past_avg

    # Sonuç: en yüksek olasılığa sahip duygu
    final_idx = np.argmax(combined_probs)
    final_emotion = states[final_idx]

    # Belleğe ekle
    memory.append(final_emotion)

    return final_emotion, combined_probs
