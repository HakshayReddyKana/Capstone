import numpy as np
from PIL import Image
import shap
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the YOLOv8 detection model
model = YOLO("weights/best.pt")

def yolo_predict_function(images):
    """
    SHAP-compatible prediction function for YOLOv8 object detection.
    Expects flattened images and returns scalar predictions.
    """
    # Reshape flattened images back to 3D
    if len(images.shape) == 2:  # Single flattened image
        img_3d = images.reshape(640, 640, 3).astype(np.uint8)
        return np.array([predict_single_image(img_3d)])
    elif len(images.shape) == 3:  # Batch of flattened images
        batch_predictions = []
        for img_flat in images:
            img_3d = img_flat.reshape(640, 640, 3).astype(np.uint8)
            pred = predict_single_image(img_3d)
            batch_predictions.append(pred)
        return np.array(batch_predictions)
    else:
        # Handle 3D single image or 4D batch
        if len(images.shape) == 3:  # Single 3D image
            return np.array([predict_single_image(images)])
        elif len(images.shape) == 4:  # Batch of 3D images
            batch_predictions = []
            for img in images:
                pred = predict_single_image(img)
                batch_predictions.append(pred)
            return np.array(batch_predictions)

def predict_single_image(image):
    """
    Predict on a single image and return a scalar value.
    """
    # Ensure image is in the right format for YOLOv8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Run inference
    results = model(image, verbose=False)

    # Return the maximum confidence score as the prediction
    if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
        max_conf = float(results[0].boxes.conf.max().cpu().numpy())
        return max_conf
    else:
        return 0.0  # No detections

# Load and prepare the image
image_path = r"TRAFFIC.jpeg"
img = Image.open(image_path)

# Convert to RGB if needed
if img.mode != 'RGB':
    img = img.convert('RGB')

img = img.resize((640, 640))
image = np.array(img)

print(f"Image shape: {image.shape}")
print("Testing YOLOv8 model...")

# Test the model
test_result = model(image, verbose=False)
if len(test_result) > 0 and test_result[0].boxes is not None:
    num_detections = len(test_result[0].boxes)
    print(f"✅ Model working: {num_detections} detections found")
else:
    print("❌ Model not detecting objects")

print("\n🔍 Setting up SHAP explainer...")

# Create background dataset for SHAP
# Flatten the image for SHAP (SHAP expects 2D input: samples × features)
background_images = np.tile(image.flatten(), (5, 1))  # 5 copies of flattened image

# Create SHAP KernelExplainer
explainer = shap.KernelExplainer(yolo_predict_function, background_images)

print("Generating SHAP values...")

# Generate SHAP values
# Use nsamples to control computational cost
# Pass the flattened image to SHAP
shap_values = explainer.shap_values(image.flatten(), nsamples=50)

print("✅ SHAP values computed successfully!")

# Visualize the results
print("Creating visualization...")

# SHAP image plot
shap.image_plot(shap_values, -image)

print("🎉 SHAP explanation completed!")
print("📊 The plot shows which pixels contribute most to the model's predictions")