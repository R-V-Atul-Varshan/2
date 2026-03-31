import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Physics-Based Image Dehazing System")
st.write("Using the Dark Channel Prior (DCP) Algorithm")

# Create history storage
if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload a Hazy Image", type=["jpg", "png", "jpeg"])
camera_file = st.camera_input("Or take a picture")

if camera_file is not None:
    uploaded_file = camera_file

# --- Core Dehazing Functions (Dark Channel Prior) ---
def get_dark_channel(img, size=15):
    """Finds the darkest pixels across RGB channels in local patches."""
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_channel, kernel)
    return dark

def get_atmospheric_light(img, dark, top_percent=0.001):
    """Estimates the color of the haze/fog itself."""
    h, w = img.shape[:2]
    num_pixels = h * w
    num_brightest = max(int(num_pixels * top_percent), 1)
    
    dark_vec = dark.flatten()
    img_vec = img.reshape(num_pixels, 3)
    
    # Get indices of the brightest pixels in the dark channel
    indices = np.argpartition(dark_vec, -num_brightest)[-num_brightest:]
    
    # Average the atmospheric light in the original image
    atms_light = np.mean(img_vec[indices], axis=0)
    return atms_light

def get_transmission(img, atms_light, omega=0.95, size=15):
    """Estimates the depth/thickness of the haze."""
    norm_img = img / atms_light
    transmission = 1 - omega * get_dark_channel(norm_img, size)
    return transmission

def recover(img, trans, atms_light, t0=0.1):
    """Reverses the atmospheric scattering to recover the clear image."""
    trans = np.maximum(trans, t0)
    trans = trans[:, :, np.newaxis] # Expand dims for broadcasting
    res = (img - atms_light) / trans + atms_light
    return np.clip(res, 0, 255).astype(np.uint8)

def dehaze(image):
    # Convert PIL Image to float64 numpy array for math operations
    img = np.array(image).astype(np.float64)

    # 1. Estimate Dark Channel
    dark = get_dark_channel(img)

    # 2. Estimate Atmospheric Light
    atms_light = get_atmospheric_light(img, dark)

    # 3. Estimate Transmission Map
    trans = get_transmission(img, atms_light)

    # Smooth transmission map to reduce blocky artifacts 
    # (A Guided Filter is strictly better here, but Gaussian is a fast fallback)
    trans_smooth = cv2.GaussianBlur(trans, (31, 31), 0)

    # 4. Recover the final scene
    result = recover(img, trans_smooth, atms_light)

    return result

# --- Streamlit UI ---
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image)

    # Run the heavy computation with a spinner
    with st.spinner("Calculating physical haze layers..."):
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
