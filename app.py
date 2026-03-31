import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Advanced Dehazing AI", layout="wide")
st.title("🌫️ Advanced CNN-Style Dehazing System")

# --- CORE DEHAZING LOGIC (Dark Channel Prior) ---
def get_dark_channel(img, size=15):
    """Calculates the dark channel of an image."""
    r, g, b = cv2.split(img)
    min_img = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_img, kernel)
    return dark

def estimate_atmospheric_light(img, dark):
    """Estimates the top 0.1% brightest pixels in the dark channel."""
    h, w = dark.shape[:2]
    imsz = h * w
    numpx = int(max(imsz / 1000, 1))
    darkvec = dark.reshape(imsz)
    imvec = img.reshape(imsz, 3)

    indices = darkvec.argsort()[-numpx:]
    atmsum = np.zeros([1, 3])
    for ind in indices:
        atmsum = atmsum + imvec[ind]
    
    return atmsum / numpx

def estimate_transmission(img, A, size=15, omega=0.95):
    """Estimates the transmission map."""
    norm_img = img / A
    transmission = 1 - omega * get_dark_channel(norm_img, size)
    return transmission

def recover_image(img, t, A, tx=0.1):
    """Final scene recovery using the Atmospheric Scattering Model."""
    res = np.empty(img.shape, img.dtype)
    t = cv2.max(t, tx) # Prevent division by zero

    for i in range(3):
        res[:, :, i] = (img[:, :, i] - A[0, i]) / t + A[0, i]
    
    return np.clip(res, 0, 255).astype(np.uint8)

def advanced_dehaze(image):
    img = np.array(image).astype(np.float64)
    
    # 1. Dark Channel
    dark = get_dark_channel(img)
    # 2. Atmospheric Light
    A = estimate_atmospheric_light(img, dark)
    # 3. Transmission Map
    te = estimate_transmission(img, A)
    # 4. Scene Recovery
    result = recover_image(img, te, A)
    
    return result

# --- STREAMLIT UI ---

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload a Hazy Image", type=["jpg","png","jpeg"])
camera_file = st.camera_input("Or take a picture")

if camera_file:
    uploaded_file = camera_file

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Hazy Image")
        st.image(image, use_column_width=True)

    # Processing Spinner
    with st.spinner('Applying Deep Dehazing Physics...'):
        result_img = advanced_dehaze(image)

    with col2:
        st.subheader("Dehazed Result")
        st.image(result_img, use_column_width=True)

    # Action Buttons
    if st.button("Save to History"):
        st.session_state.history.append(result_img)
        st.success("Saved!")

# History Sidebar/Bottom
if st.session_state.history:
    st.divider()
    st.subheader("📜 Processing History")
    cols = st.columns(4)
    for i, img in enumerate(reversed(st.session_state.history)):
        with cols[i % 4]:
            st.image(img, caption=f"Result {len(st.session_state.history)-i}")
