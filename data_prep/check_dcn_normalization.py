import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

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


def inspect_normalization(model_key):
    print(f"\n=== {model_key.upper()} ===")

    # 1. Load the saved checkpoint's normalization stats (used at inference)
    checkpoint_path = f"D:/DentalCalib-Net/stage4_outputs/dentalcalib_net_models/dentalcalib_net_{model_key}.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    saved_mean = checkpoint['feature_mean']
    saved_std = checkpoint['feature_std']
    print(f"Saved (training-time) feature_mean: {saved_mean}")
    print(f"Saved (training-time) feature_std:  {saved_std}")

    # 2. Load the ORIGINAL training vectors and recompute mean/std from scratch
    vectors_path = f"D:/DentalCalib-Net/stage4_outputs/training_vectors/{model_key}_val_vectors.json"
    with open(vectors_path) as f:
        data = json.load(f)
    X = np.array([d['input_vector'] for d in data], dtype=np.float32)

    recomputed_mean = X[:, 5:8].mean(axis=0)
    recomputed_std = X[:, 5:8].std(axis=0)
    print(f"Recomputed from training_vectors quality features: mean={recomputed_mean}, std={recomputed_std}")

    mean_match = np.allclose(saved_mean, recomputed_mean, rtol=1e-5)
    std_match = np.allclose(saved_std, recomputed_std, rtol=1e-5)
    print(f"Mean matches: {mean_match} | Std matches: {std_match}")

    # 3. Check a real recalibrated prediction: manually reproduce the DCN confidence
    #    using saved stats, compare against what apply_recalibration.py actually output
    recal_path = f"D:/DentalCalib-Net/stage4_outputs/recalibrated_predictions/{model_key}_clean_recalibrated.json"
    with open(recal_path) as f:
        recal_data = json.load(f)

    sample_entry = recal_data[0]
    sample_pred = sample_entry['predictions'][0]

    print(f"\nSample prediction from recalibrated file:")
    print(f"  confidence_raw: {sample_pred['confidence_raw']:.4f}")
    print(f"  confidence_dcn (saved by apply_recalibration.py): {sample_pred['confidence_dcn']:.4f}")

    if 'quality_features_used' in sample_pred:
        print(f"  quality_features_used (if logged): {sample_pred['quality_features_used']}")
    else:
        print("  (raw quality feature values not stored in recalibrated file -- cannot directly re-verify this specific prediction's normalization without re-running apply_recalibration.py's lookup logic)")


if __name__ == "__main__":
    inspect_normalization('yolov8')
    inspect_normalization('rtdetr')