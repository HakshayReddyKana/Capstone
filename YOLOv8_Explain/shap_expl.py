import onnxruntime as ort
import numpy as np
from PIL import Image

# Load the ONNX model
ort_session = ort.InferenceSession("yolov8n.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def yolo_predict(images):
    # Preprocess the images
    images = [np.array(image) for image in images]
    images = np.stack(images, axis=0)

    # Run the ONNX model
    ort_inputs = {ort_session.get_inputs()[0].name: images}
    ort_outs = ort_session.run(None, ort_inputs)

    # Extract the segmentation masks or other relevant outputs
    masks = ort_outs[0]  # Adjust based on your model's output structure
    return masks

# Create a masker for the images
masker = shap.maskers.Image("inpaint_telea", (640, 640, 3))

# Create an explainer using the custom prediction function
explainer = shap.Explainer(yolo_predict, masker)

# Select an image to explain
image_path = r"path_to_image.png"
img = Image.open(image_path)
img = img.resize((640, 640))
image = np.array(img)
image_with_batch = np.expand_dims(image, axis=0)
print(image_with_batch.shape)

# Generate SHAP values
shap_values = explainer(image_with_batch, max_evals=20, outputs=shap.Explanation.argsort.flip[:1])

# Visualize the explanation
shap.image_plot(shap_values, image_with_batch)