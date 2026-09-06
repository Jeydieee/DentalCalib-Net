import json
import numpy as np
from scipy.optimize import minimize_scalar
from pathlib import Path


def confidence_to_logit(conf, eps=1e-7):
    """Invert sigmoid to recover the logit from a confidence score."""
    conf = np.clip(conf, eps, 1 - eps)
    return np.log(conf / (1 - conf))


def apply_temperature(logit, T):
    """Apply temperature scaling to a logit, return calibrated confidence."""
    scaled_logit = logit / T
    return 1 / (1 + np.exp(-scaled_logit))


def negative_log_likelihood(T, logits, labels, eps=1e-7):
    """NLL of the temperature-scaled predictions against true binary labels."""
    if T <= 0:
        return np.inf
    calibrated = apply_temperature(logits, T)
    calibrated = np.clip(calibrated, eps, 1 - eps)
    nll = -np.mean(labels * np.log(calibrated) + (1 - labels) * np.log(1 - calibrated))
    return nll


def fit_temperature(vectors_path):
    """Fits optimal T on a set of labeled predictions (validation data)."""
    with open(vectors_path) as f:
        data = json.load(f)

    confidences = np.array([d['input_vector'][0] for d in data])  # index 0 = raw confidence
    labels = np.array([d['label'] for d in data], dtype=np.float64)

    logits = confidence_to_logit(confidences)

    result = minimize_scalar(
        negative_log_likelihood,
        args=(logits, labels),
        bounds=(0.01, 10.0),
        method='bounded',
    )

    optimal_T = result.x
    final_nll = result.fun

    return optimal_T, final_nll


if __name__ == "__main__":
    VECTORS_DIR = r"D:\DentalCalib-Net\stage4_outputs\training_vectors"
    OUTPUT_PATH = r"D:\DentalCalib-Net\stage4_outputs\temperature_scaling_params.json"

    results = {}

    for model_key, vectors_file in [
        ('yolov8', 'yolov8_val_vectors.json'),
        ('rtdetr', 'rtdetr_val_vectors.json'),
    ]:
        T, nll = fit_temperature(f"{VECTORS_DIR}/{vectors_file}")
        results[model_key] = {'temperature': T, 'nll': nll}
        print(f"{model_key}: optimal T = {T:.4f}, NLL = {nll:.4f}")

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved -> {OUTPUT_PATH}")