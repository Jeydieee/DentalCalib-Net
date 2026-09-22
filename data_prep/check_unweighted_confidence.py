import json
import numpy as np
import torch
import torch.nn as nn

class DentalCalibNet(nn.Module):
    """Must match the architecture in train_dentalcalib_net.py exactly."""
    def __init__(self, input_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_vectors(path):
    with open(path) as f:
        data = json.load(f)
    X = np.array([d['input_vector'] for d in data], dtype=np.float32)
    y = np.array([d['label'] for d in data], dtype=np.float32)
    return X, y


def check_confidence_vs_true_rate(model_name, vectors_path, checkpoint_path):
    X, y = load_vectors(vectors_path)
    true_positive_rate = y.mean()

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    feat_mean = checkpoint['feature_mean']
    feat_std = checkpoint['feature_std']

    X_norm = X.copy()
    X_norm[:, 5:8] = (X_norm[:, 5:8] - feat_mean) / feat_std

    model = DentalCalibNet(input_dim=X.shape[1])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_norm))
        probs = torch.sigmoid(logits).numpy().flatten()

    mean_predicted_confidence = probs.mean()

    print(f"\n=== {model_name} (unweighted BCE) ===")
    print(f"  True positive rate:         {true_positive_rate*100:.2f}%")
    print(f"  Mean predicted confidence:  {mean_predicted_confidence*100:.2f}%")
    print(f"  Overconfidence factor:      {mean_predicted_confidence/true_positive_rate:.2f}x")


if __name__ == "__main__":
    VECTORS_DIR = r"D:\DentalCalib-Net\stage4_outputs\training_vectors"
    MODELS_DIR = r"D:\DentalCalib-Net\stage4_outputs\dentalcalib_net_models"

    check_confidence_vs_true_rate(
        "yolov8",
        f"{VECTORS_DIR}/yolov8_val_vectors.json",
        f"{MODELS_DIR}/dentalcalib_net_yolov8_unweighted.pt"
    )

    check_confidence_vs_true_rate(
        "rtdetr",
        f"{VECTORS_DIR}/rtdetr_val_vectors.json",
        f"{MODELS_DIR}/dentalcalib_net_rtdetr_unweighted.pt"
    )