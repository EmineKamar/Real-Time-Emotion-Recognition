import os
import pandas as pd

audio_dir = "C:\\Users\\emina\\OneDrive\\Masaüstü\\vscode\\gemma\\data\\AudioWAV"
label_file = "C:\\Users\\emina\\OneDrive\\Masaüstü\\vscode\\gemma\\data\\labels.csv"

data = []

for fname in os.listdir(audio_dir):
    if fname.endswith(".wav"):
        parts = fname.split("_")
        if len(parts) >= 3:
            emotion = parts[2]  # Örneğin: ANG, HAP, SAD
            path = os.path.join(audio_dir, fname)
            data.append({"path": path, "emotion": emotion})

df = pd.DataFrame(data)
df.to_csv(label_file, index=False)
print(f"{label_file} dosyası {len(df)} kayıtla oluşturuldu.")
