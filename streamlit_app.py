import streamlit as st
from PIL import Image
import easyocr
import numpy as np
import cv2

# Title of the web app
st.title("Image Upload and OCR Extraction App")

# Ask the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image using PIL
    image = Image.open(uploaded_file)
    
    # Convert PIL image to NumPy array
    image_np = np.array(image)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert image to BGR format as EasyOCR works with OpenCV format
    if len(image_np.shape) == 2:  # If the image is already grayscale
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    else:
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Initialize EasyOCR Reader
    reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have a GPU
    
    # Perform OCR on the image
    st.write("Extracting text using OCR...")
    result = reader.readtext(image_cv)

    # Display the OCR result
    st.write("Extracted Text:")
    for (bbox, text, prob) in result:
        st.text(f"{text} (Confidence: {prob:.2f})")