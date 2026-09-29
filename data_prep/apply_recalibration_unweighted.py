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


def normalize_bbox(bbox_xyxy, img_width=640, img_height=640):
    x1, y1, x2, y2 = bbox_xyxy
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    cx = ((x1 + x2) / 2) / img_width
    cy = ((y1 + y2) / 2) / img_height
    return [w, h, cx, cy]


def confidence_to_logit(conf, eps=1e-7):
    conf = np.clip(conf, eps, 1 - eps)
    return np.log(conf / (1 - conf))


def apply_temperature(logit, T):
    scaled_logit = logit / T
    return 1 / (1 + np.exp(-scaled_logit))


def load_dentalcalib_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = DentalCalibNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['feature_mean'], checkpoint['feature_std']


def apply_recalibration_to_condition(
    predictions_path, quality_features, quality_key,
    dcn_model, feat_mean, feat_std, temperature,
    out_path,
    is_ood=False,
):
    with open(predictions_path) as f:
        predictions_data = json.load(f)

    output = []
    missing_quality = 0

    for entry in predictions_data:
        filename = Path(entry['image_path']).name

        if is_ood:
            img_quality = quality_features.get('test_ood', {}).get(quality_key, {}).get(filename)
        else:
            img_quality = quality_features.get(quality_key, {}).get(filename)

        if img_quality is None:
            missing_quality += 1
            continue

        new_predictions = []
        for pred in entry['predictions']:
            confidence = pred['confidence']
            bbox_features = normalize_bbox(pred['bbox_xyxy'])

            logit = confidence_to_logit(np.array([confidence]))[0]
            ts_confidence = float(apply_temperature(logit, temperature))

            raw_vector = np.array([
                confidence,
                bbox_features[0], bbox_features[1], bbox_features[2], bbox_features[3],
                img_quality['laplacian_variance'],
                img_quality['pixel_std_dev'],
                img_quality['jpeg_block_artifact_score'],
            ], dtype=np.float32)

            normalized_vector = raw_vector.copy()
            normalized_vector[5:8] = (normalized_vector[5:8] - feat_mean) / feat_std

            with torch.no_grad():
                dcn_logit = dcn_model(torch.tensor(normalized_vector).unsqueeze(0))
                dcn_confidence = float(torch.sigmoid(dcn_logit).item())

            new_pred = dict(pred)
            new_pred['confidence_raw'] = confidence
            new_pred['confidence_ts'] = ts_confidence
            new_pred['confidence_dcn'] = dcn_confidence
            new_predictions.append(new_pred)

        output.append({'image_path': entry['image_path'], 'predictions': new_predictions})

    with open(out_path, 'w') as f:
        json.dump(output, f)

    total_preds = sum(len(e['predictions']) for e in output)
    print(f"{Path(predictions_path).name}: {total_preds} predictions recalibrated, "
          f"{missing_quality} images skipped (no quality features) -> {out_path}")


if __name__ == "__main__":
    STAGE2_DIR = r"D:\DentalCalib-Net\stage2_outputs"
    FILTERED_DIR = r"D:\DentalCalib-Net\stage4_outputs\rtdetr_filtered"
    QUALITY_FEATURES_PATH = r"D:\DentalCalib-Net\data_prep\quality_features.json"
    TS_PARAMS_PATH = r"D:\DentalCalib-Net\stage4_outputs\temperature_scaling_params.json"
    DCN_MODELS_DIR = r"D:\DentalCalib-Net\stage4_outputs\dentalcalib_net_models"
    OUTPUT_DIR = r"D:\DentalCalib-Net\stage4_outputs\recalibrated_predictions_unweighted"

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    with open(QUALITY_FEATURES_PATH) as f:
        quality_features = json.load(f)
    with open(TS_PARAMS_PATH) as f:
        ts_params = json.load(f)

    CORRUPTION_TYPES = ['gaussian_noise', 'motion_blur', 'brightness_variation', 'jpeg_compression']
    SEVERITIES = ['S1', 'S2', 'S3', 'S4', 'S5']

    for model_key in ['yolov8', 'rtdetr']:
        print(f"\n=== Recalibrating {model_key} (unweighted DCN) ===")

        dcn_model, feat_mean, feat_std = load_dentalcalib_model(f"{DCN_MODELS_DIR}/dentalcalib_net_{model_key}_unweighted.pt")
        temperature = ts_params[model_key]['temperature']

        if model_key == 'yolov8':
            clean_path = f"{STAGE2_DIR}/yolov8_predictions_clean.json"
        else:
            clean_path = f"{FILTERED_DIR}/rtdetr_predictions_clean_filtered.json"

        apply_recalibration_to_condition(
            clean_path, quality_features, 'test_clean',
            dcn_model, feat_mean, feat_std, temperature,
            f"{OUTPUT_DIR}/{model_key}_clean_recalibrated.json",
        )

        for corruption in CORRUPTION_TYPES:
            for severity in SEVERITIES:
                quality_key = f"{corruption}_{severity}"

                if model_key == 'yolov8':
                    pred_path = f"{STAGE2_DIR}/yolov8_ood_predictions/yolov8_{corruption}_{severity}.json"
                else:
                    pred_path = f"{FILTERED_DIR}/rtdetr_{corruption}_{severity}_filtered.json"

                out_path = f"{OUTPUT_DIR}/{model_key}_{corruption}_{severity}_recalibrated.json"

                apply_recalibration_to_condition(
                    pred_path, quality_features, quality_key,
                    dcn_model, feat_mean, feat_std, temperature,
                    out_path,
                    is_ood=True,
                )

    print("\nDone. Unweighted DCN recalibration complete.")