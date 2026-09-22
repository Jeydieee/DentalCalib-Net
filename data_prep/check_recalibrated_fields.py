import json

with open(r"D:\DentalCalib-Net\stage4_outputs\recalibrated_predictions\yolov8_clean_recalibrated.json") as f:
    data = json.load(f)

print("Available keys in a sample prediction:")
print(list(data[0]['predictions'][0].keys()))