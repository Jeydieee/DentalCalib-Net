import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

class DentalCalibNet(nn.Module):
    """3-layer MLP: 8-dim input -> 64 (ReLU, dropout) -> 32 (ReLU) -> 1 (sigmoid)."""
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


def normalize_features(X, feature_mean=None, feature_std=None):
    X = X.copy()
    if feature_mean is None:
        feature_mean = X[:, 5:8].mean(axis=0)
        feature_std = X[:, 5:8].std(axis=0)
        feature_std[feature_std == 0] = 1.0
    X[:, 5:8] = (X[:, 5:8] - feature_mean) / feature_std
    return X, feature_mean, feature_std


def train_dentalcalib_net(X, y, model_name, epochs=100, patience=10, batch_size=512, lr=1e-3):
    X_norm, feat_mean, feat_std = normalize_features(X)

    X_tensor = torch.tensor(X_norm)
    y_tensor = torch.tensor(y).unsqueeze(1)

    n = len(X_tensor)
    indices = np.random.RandomState(42).permutation(n)
    split = int(n * 0.8)
    train_idx, val_idx = indices[:split], indices[split:]

    X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
    X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

    model = DentalCalibNet(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_pos = y_train.sum().item()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos]) if n_pos > 0 else torch.tensor([1.0])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_idx = perm[i:i+batch_size]
            xb, yb = X_train[batch_idx], y_train[batch_idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter >= patience:
            print(f"  [{model_name}] epoch {epoch}: val_loss={val_loss:.4f} (best={best_val_loss:.4f})")

        if patience_counter >= patience:
            print(f"  [{model_name}] early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, feat_mean, feat_std, best_val_loss


if __name__ == "__main__":
    VECTORS_DIR = r"D:\DentalCalib-Net\stage4_outputs\training_vectors"
    OUTPUT_DIR = r"D:\DentalCalib-Net\stage4_outputs\dentalcalib_net_models"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # YOLOv8 -- unchanged, already converged with early stopping at epoch 71
    print("\n=== Training DentalCalib-Net for yolov8 ===")
    X, y = load_vectors(f"{VECTORS_DIR}/yolov8_val_vectors.json")
    print(f"  Loaded {len(X)} vectors, {y.sum():.0f} positive ({100*y.mean():.1f}%)")
    model, feat_mean, feat_std, best_val_loss = train_dentalcalib_net(X, y, 'yolov8', epochs=100)
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_mean': feat_mean,
        'feature_std': feat_std,
        'best_val_loss': best_val_loss,
    }, f"{OUTPUT_DIR}/dentalcalib_net_yolov8.pt")
    print(f"  Saved -> {OUTPUT_DIR}/dentalcalib_net_yolov8.pt")

    # RT-DETR -- increased epoch cap to 300, since it hadn't converged at 100
    print("\n=== Training DentalCalib-Net for rtdetr ===")
    X, y = load_vectors(f"{VECTORS_DIR}/rtdetr_val_vectors.json")
    print(f"  Loaded {len(X)} vectors, {y.sum():.0f} positive ({100*y.mean():.1f}%)")
    model, feat_mean, feat_std, best_val_loss = train_dentalcalib_net(X, y, 'rtdetr', epochs=300)
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_mean': feat_mean,
        'feature_std': feat_std,
        'best_val_loss': best_val_loss,
    }, f"{OUTPUT_DIR}/dentalcalib_net_rtdetr.pt")
    print(f"  Saved -> {OUTPUT_DIR}/dentalcalib_net_rtdetr.pt")

    print("\nDone.")