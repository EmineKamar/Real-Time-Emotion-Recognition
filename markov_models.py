import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Duygu Durumları ve isim eşlemesi
states = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
state_names = {
    "ANG": "Angry",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad"
}
state_to_idx = {s: i for i, s in enumerate(states)}

label_to_emotion = {
    0: "NEU",  # Neutral
    1: "HAP",  # Happy
    2: "SAD",  # Sad
    3: "ANG",  # Angry
    4: "FEA",  # Fear
    5: "DIS",  # Disgust
}

def load_all_emotions_sequential():
    dataset = load_dataset("myleslinder/crema-d", trust_remote_code=True)["train"]
    emotions = [label_to_emotion[sample['label']] for sample in dataset]
    return emotions

def compute_transition_matrix_from_sequence(emotions, smoothing=0.1):
    n_states = len(states)
    counts = np.zeros((n_states, n_states))
    for i in range(len(emotions) - 1):
        cur, nxt = emotions[i], emotions[i + 1]
        if cur in state_to_idx and nxt in state_to_idx:
            counts[state_to_idx[cur], state_to_idx[nxt]] += 1
    counts += smoothing  # Laplace smoothing
    row_sums = counts.sum(axis=1, keepdims=True)
    transition_matrix = counts / row_sums
    return transition_matrix

def plot_transition_matrix(matrix):
    plt.figure(figsize=(8,6))
    sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=[state_names[s] for s in states],
                yticklabels=[state_names[s] for s in states], cmap="Blues")
    plt.title("Duygu Geçiş Matrisi (Markov Modeli)")
    plt.xlabel("Sonraki Duygu")
    plt.ylabel("Mevcut Duygu")
    plt.show()

def simulate_markov_chain(transition_matrix, start_state, steps=10):
    current_state = start_state
    sequence = [current_state]
    for _ in range(steps):
        probs = transition_matrix[state_to_idx[current_state]]
        next_state = np.random.choice(states, p=probs)
        sequence.append(next_state)
        current_state = next_state
    return sequence

if __name__=="__main__":
    emotions = load_all_emotions_sequential()
    transition_matrix = compute_transition_matrix_from_sequence(emotions, smoothing=0.1)
    print("Geçiş Matrisi:")
    for i, state in enumerate(states):
        print(f"{state_names[state]}: {transition_matrix[i]}")
    plot_transition_matrix(transition_matrix)

    start = "NEU"
    chain = simulate_markov_chain(transition_matrix, start_state=start, steps=15)
    chain_names = [state_names[s] for s in chain]
    print(f"\nBaşlangıç durumu '{state_names[start]}' olan tahmini duygu zinciri:")
    print(chain_names)
