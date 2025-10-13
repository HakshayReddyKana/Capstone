
from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential

# Allowlist the required classes for safe loading in PyTorch 2.6+
with torch.serialization.safe_globals([DetectionModel, Sequential]):
    # Load the YOLOv8 segmentation model
    model = YOLO("weights/best.pt")

# Export the model to ONNX format
model.export(format="onnx")