# model_utils.py
import torch

def predict_emotion(model, features, device):
    model.eval()
    with torch.no_grad():
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(features_tensor)
        predicted_label = torch.argmax(output, dim=1).item()
        return predicted_label
