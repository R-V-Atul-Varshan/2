import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Pro Dehaze Studio", layout="wide")
st.title("Physics-Based Image Dehazing")
st.write("Refined Dark Channel Prior (DCP) with Guided Filtering")

# --- Advanced Dehazing Classes & Functions ---

class GuidedFilter:
    """Refines the transmission map by aligning it with the guidance image edges."""
    def __init__(self, I, radius, eps):
        self.I = I / 255.0
        self.radius = radius
        self.eps = eps

    def filter(self, p):
        p = p / 255.0 if p.max() > 1 else p
        mean_I = cv2.boxFilter(self.I, -1, (self.radius, self.radius))
        mean_p = cv2.boxFilter(p, -1, (self.radius, self.radius))
        mean_Ip = cv2.boxFilter(self.I * p, -1, (self.radius, self.radius))
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(self.I * self.I, -1, (self.radius, self.radius))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, -1, (self.radius, self.radius))
        mean_b = cv2.boxFilter(b, -1, (self.radius, self.radius))

        return mean_a * self.I + mean_b

def get_dark_channel(img, size=15):
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.erode(min_channel, kernel)

def get_atmospheric_light(img, dark, top_percent=0.001):
    h, w = img.shape[:2]
    num_pixels = h * w
    num_brightest = max(int(num_pixels * top_percent), 1)
    dark_vec = dark.flatten()
    img_vec = img.reshape(num_pixels, 3)
    indices = np.argpartition(dark_vec, -num_brightest)[-num_brightest:]
    return np.mean(img_vec[indices], axis=0)

def get_transmission(img, atms_light, omega=0.95, size=15):
    norm_img = img / atms_light
    return 1.0 - omega * get_dark_channel(norm_img, size)

def recover(img, trans, atms_light, t0=0.1):
    trans = np.maximum(trans, t0)
    # Reshape for broadcasting
    trans_stack = np.repeat(trans[:, :, np.newaxis], 3, axis=2)
    res = (img - atms_light) / trans_stack + atms_light
    return np.clip(res, 0, 255).astype(np.uint8)

def dehaze_pro(image):
    img = np.array(image).astype(np.float64)
    
    # 1. Dark Channel
    dark = get_dark_channel(img)
    
    # 2. Atmospheric Light
    atms_light = get_atmospheric_light(img, dark)
    
    # 3. Raw Transmission
    trans_raw = get_transmission(img, atms_light)
    
    # 4. Guided Filter Refinement (The Secret Sauce)
    # Using the grayscale original image as guidance
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gf = GuidedFilter(gray, radius=60, eps=0.0001)
    trans_refined = gf.filter(trans_raw)
    
    # 5. Recovery
    result = recover(img, trans_refined, atms_light)
    
    # Optional: Light post-processing for vibrancy
    # Convert to LAB to bump contrast slightly
    lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l) # Simple brightness balance
    # Blend original L with equalized L for natural look
    l = cv2.addWeighted(l, 0.2, cv2.split(cv2.cvtColor(result, cv2.COLOR_RGB2LAB))[0], 0.8, 0)
    result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    
    return result

# --- UI Layout ---

uploaded_file = st.file_uploader("Upload Hazy Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Hazy Input", use_container_width=True)
    
    with st.spinner("Removing atmospheric particles..."):
        cleared_img = dehaze_pro(image)
        
    with col2:
        st.image(cleared_img, caption="Dehazed Result", use_container_width=True)
        
    # Download Button
    result_pil = Image.fromarray(cleared_img)
    st.download_button("Download Clear Image", data=uploaded_file, file_name="dehazed.png")
