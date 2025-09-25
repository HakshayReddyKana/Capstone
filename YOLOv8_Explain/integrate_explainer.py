from YOLOv8_Explainer import yolov8_heatmap, display_images
import os

# Path to your finetuned model (adjust if needed)
model_path = "weights/best.pt"

# Path to an image for testing (using TRAFFIC.jpeg from workspace)
image_path = "TRAFFIC.jpeg"

print(f"Model path: {model_path}")
print(f"Image path: {image_path}")
print(f"Model exists: {os.path.exists(model_path)}")
print(f"Image exists: {os.path.exists(image_path)}")

# Initialize the explainer with your model
explainer = yolov8_heatmap(
    weight=model_path,
    method="EigenCAM",  # You can change this to other methods like "GradCAM", "HiResCAM", etc.
    layer=[10, 12, 14, 16, 18, -3],  # Default layers, adjust if needed
    conf_threshold=0.4,
    ratio=0.02,
    show_box=True,
    renormalize=False
)

print("Explainer initialized.")

# Generate explanations for the image
images = explainer(img_path=image_path)

print(f"Number of images generated: {len(images)}")

# Save the results instead of displaying
for i, img in enumerate(images):
    img.save(f"explanation_{i}.png")
    print(f"Saved explanation_{i}.png")

print("Explainability analysis completed. Images saved as explanation_*.png")