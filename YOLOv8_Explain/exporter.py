from ultralytics import YOLO

# Load the YOLOv8 segmentation model
model = YOLO("path_to_file.pt")

# Export the model to ONNX format
model.export(format="onnx")