import streamlit as st
from PIL import Image
import easyocr

# Title of the web app
st.title("Image Upload and OCR Extraction App")

# Ask the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert image to bytes for EasyOCR
    image_bytes = uploaded_file.read()

    # Initialize EasyOCR Reader
    reader = easyocr.Reader(['en'], gpu=False)  # You can add more languages if needed
    
    # Perform OCR on the uploaded image
    st.write("Extracting text using OCR...")
    result = reader.readtext(image_bytes)

    # Display the OCR result
    st.write("Extracted Text:")
    for (bbox, text, prob) in result:
        st.text(f"{text} (Confidence: {prob:.2f})")