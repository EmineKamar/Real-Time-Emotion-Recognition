import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns

states = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
state_names = {"ANG":"Angry","DIS":"Disgust","FEA":"Fear","HAP":"Happy","NEU":"Neutral","SAD":"Sad"}
state_to_idx = {s: i for i, s in enumerate(states)}
label_to_emotion = {0: "NEU", 1: "HAP", 2: "SAD", 3: "ANG", 4: "FEA", 5: "DIS"}

def compute_transition_matrix(emotions, smoothing=0.1):
    n_states = len(states)
    counts = np.zeros((n_states, n_states))
    for i in range(len(emotions) - 1):
        cur, nxt = emotions[i], emotions[i + 1]
        counts[state_to_idx[cur], state_to_idx[nxt]] += 1
    counts += smoothing
    return counts / counts.sum(axis=1, keepdims=True)

def main():
    print("🔄 Geçiş matrisi hesaplanıyor...")
    dataset = load_dataset("myleslinder/crema-d", trust_remote_code=True)["train"]
    emotions = [label_to_emotion[sample["label"]] for sample in dataset]
    matrix = compute_transition_matrix(emotions)

    np.save("transition_matrix.npy", matrix)
    print("✅ Geçiş matrisi transition_matrix.npy dosyasına kaydedildi.")

    # İsteğe bağlı görselleştirme
    plt.figure(figsize=(8,6))
    sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=[state_names[s] for s in states],
                yticklabels=[state_names[s] for s in states], cmap="Blues")
    plt.title("Duygu Geçiş Matrisi")
    plt.savefig("transition_matrix.png")
    print("📊 transition_matrix.png olarak kaydedildi.")

if __name__ == "__main__":
    main()
