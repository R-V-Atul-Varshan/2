import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("CNN Image Dehazing System")

# Create history storage
if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload a Hazy Image", type=["jpg","png","jpeg"])
camera_file = st.camera_input("Or take a picture")

if camera_file is not None:
    uploaded_file = camera_file
def dehaze(image):
    img = np.array(image)

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (contrast enhancement)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    # Merge back
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    return enhanced

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image)

    result = dehaze(image)

    with col2:
        st.subheader("Dehazed Image")
        st.image(result)

    # Save result to history
    st.session_state.history.append(result)

# Show history
if st.session_state.history:
    st.subheader("Dehazing History")

    for i, img in enumerate(st.session_state.history):
        st.image(img, caption=f"Processed Image {i+1}")
