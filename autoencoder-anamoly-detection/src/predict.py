import torch
from src.model import Autoencoder
import numpy as np

def load_model(input_dim):
    model = Autoencoder(input_dim)
    model.load_state_dict(torch.load("models/autoencoder.pth"))
    model.eval()
    return model

def predict_anomaly(data):
    model = load_model(data.shape[1])
    X = torch.tensor(data, dtype=torch.float32)

    with torch.no_grad():
        recon = model(X)
        loss = torch.mean((X - recon) ** 2, dim=1).numpy()

    threshold = np.mean(loss) + 2 * np.std(loss)
    return (loss > threshold).tolist()