# Using streamlit or gradio

import streamlit as st
from PIL import Image
import tempfile
import os
import time
from YOLOv8_Explainer import yolov8_heatmap, display_images

st.title("YOLOv8 Explainer - XAI Web Interface")
st.markdown("Upload an image to see how YOLOv8 makes its predictions!")

# Sidebar for configuration
st.sidebar.header("Configuration")
method = st.sidebar.selectbox(
    "Explanation Method",
    ["GradCAM", "EigenCAM", "EigenGradCAM", "HiResCAM", "LayerCAM"],
    index=0
)

show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=False)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.1)

# Main interface
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    # Save uploaded image to temporary file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        image.save(tmp_file.name)
        tmp_path = tmp_file.name

    try:
        # Initialize explainer with user settings
        with st.spinner("Generating explanation..."):
            model = yolov8_heatmap(
                weight="weights/best.pt",
                method=method,
                show_box=show_boxes,
                conf_threshold=conf_threshold
            )

            # Generate explanation
            imagelist = model(img_path=tmp_path)

        # Display results
        if imagelist:
            st.image(imagelist[0], caption=f"Explanation ({method})", use_column_width=True)

            st.success(" Explanation generated successfully!")

            # Additional info
            st.markdown("---")
            st.markdown("### Analysis Details")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Method", method)
            with col2:
                st.metric("Confidence Threshold", f"{conf_threshold:.1f}")
            with col3:
                st.metric("Bounding Boxes", "Yes" if show_boxes else "No")

        else:
            st.error("❌ No explanation generated. Try adjusting the confidence threshold.")

    except Exception as e:
        st.error(f"❌ Error generating explanation: {str(e)}")

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

else:
    st.info("💡 Upload an image to get started!")
    st.markdown("""
    ### 🎯 How it works:
    1. **Upload** an image (JPG, PNG)
    2. **Choose** your preferred explanation method
    3. **Adjust** confidence threshold if needed
    4. **View** the explanation heatmap overlay

    ### 📖 Explanation Methods:
    - **GradCAM**: Gradient-based localization
    - **EigenCAM**: Principal component analysis
    - **EigenGradCAM**: Eigen + Gradient combination
    - **HiResCAM**: High-resolution explanations
    - **LayerCAM**: Layer-wise attention
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using YOLOv8 and Streamlit")