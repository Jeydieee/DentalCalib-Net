import json
import numpy as np
import torch
import torch.nn as nn

class DentalCalibNet(nn.Module):
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


def normalize_features(X, feature_mean, feature_std):
    X = X.copy()
    X[:, 5:8] = (X[:, 5:8] - feature_mean) / feature_std
    return X


def evaluate_on_training_data(model_key):
    print(f"\n=== {model_key.upper()} ===")

    checkpoint_path = f"D:/DentalCalib-Net/stage4_outputs/dentalcalib_net_models/dentalcalib_net_{model_key}.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    model = DentalCalibNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    feat_mean = checkpoint['feature_mean']
    feat_std = checkpoint['feature_std']

    vectors_path = f"D:/DentalCalib-Net/stage4_outputs/training_vectors/{model_key}_val_vectors.json"
    with open(vectors_path) as f:
        data = json.load(f)

    X = np.array([d['input_vector'] for d in data], dtype=np.float32)
    y = np.array([d['label'] for d in data], dtype=np.float32)

    X_norm = normalize_features(X, feat_mean, feat_std)

    with torch.no_grad():
        logits = model(torch.tensor(X_norm))
        predicted_conf = torch.sigmoid(logits).squeeze().numpy()

    # simple calibration check ON THE TRAINING DATA ITSELF
    n_bins = 15
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (predicted_conf >= bin_edges[i]) & (predicted_conf < bin_edges[i+1])
        if mask.sum() == 0:
            continue
        bin_acc = y[mask].mean()
        bin_conf = predicted_conf[mask].mean()
        ece += (mask.sum() / len(y)) * abs(bin_acc - bin_conf)

    print(f"Training-data ECE (i.e. how well DCN fits the data it was trained on): {ece:.4f}")
    print(f"Mean predicted confidence: {predicted_conf.mean():.4f}")
    print(f"Actual positive rate (true label mean): {y.mean():.4f}")
    print(f"Prediction std dev: {predicted_conf.std():.4f}")


if __name__ == "__main__":
    evaluate_on_training_data('yolov8')
    evaluate_on_training_data('rtdetr')