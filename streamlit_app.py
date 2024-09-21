import streamlit as st
from PIL import Image
import easyocr
import numpy as np
import cv2

# Load CSS for custom styling
with open('wave.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Sidebar with app info and settings
st.sidebar.title("App Details")
st.sidebar.info("This application allows users to upload an image and extract text using Optical Character Recognition (OCR) with EasyOCR.")

# Title of the web app
st.title("Detailed Image Upload and OCR Extraction App")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Preprocessing options
preprocess = st.sidebar.radio("Preprocessing Options:", ("None", "Grayscale", "Blurring"))

# OCR Language selection
languages = ['en', 'fr', 'de', 'es']  # Add more languages if needed
ocr_language = st.sidebar.selectbox("Select OCR Language", languages, index=0)

# Confidence threshold slider (optional, but not used in this case)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Option to enable GPU
use_gpu = st.sidebar.checkbox("Use GPU (if available)", value=False)

if uploaded_file is not None:
    # Open the uploaded image using PIL
    image = Image.open(uploaded_file)
    
    # Convert image to NumPy array
    image_np = np.array(image)

    # Preprocess the image based on user selection
    if preprocess == "Grayscale":
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    elif preprocess == "Blurring":
        image_np = cv2.GaussianBlur(image_np, (5, 5), 0)

    # Convert image to BGR format if it's not grayscale
    if len(image_np.shape) == 2:
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    else:
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Display the uploaded and preprocessed image
    st.image(image, caption="Uploaded and Preprocessed Image", use_column_width=True)

    # Initialize EasyOCR Reader
    reader = easyocr.Reader([ocr_language], gpu=use_gpu)

    # Perform OCR on the image
    st.write("Extracting text using OCR...")
    result = reader.readtext(image_cv)

    # Display the OCR result without showing confidence, line by line
    st.write("Extracted Text:")
    for (bbox, text, prob) in result:
        st.write(text)

    # Convert the image back to RGB for displaying in Streamlit
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

else:
    st.warning("Please upload an image file.")

# Footer with app credits
st.sidebar.markdown("---")
st.sidebar.write("**Created by:** Apurva Pandey")
st.sidebar.write("**Powered by:** EasyOCR, Streamlit, OpenCV")
