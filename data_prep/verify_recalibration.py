import json

with open(r"D:\DentalCalib-Net\stage4_outputs\recalibrated_predictions\yolov8_clean_recalibrated.json") as f:
    data = json.load(f)

sample_pred = data[0]['predictions'][0]
print("Sample prediction (YOLOv8, clean):")
print(f"  raw:  {sample_pred['confidence_raw']:.4f}")
print(f"  TS:   {sample_pred['confidence_ts']:.4f}")
print(f"  DCN:  {sample_pred['confidence_dcn']:.4f}")

with open(r"D:\DentalCalib-Net\stage4_outputs\recalibrated_predictions\rtdetr_clean_recalibrated.json") as f:
    data_rtdetr = json.load(f)

sample_pred_rtdetr = data_rtdetr[0]['predictions'][0]
print("\nSample prediction (RT-DETR, clean):")
print(f"  raw:  {sample_pred_rtdetr['confidence_raw']:.4f}")
print(f"  TS:   {sample_pred_rtdetr['confidence_ts']:.4f}")
print(f"  DCN:  {sample_pred_rtdetr['confidence_dcn']:.4f}")