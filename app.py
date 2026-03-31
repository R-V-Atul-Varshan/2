import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# --- 1. Define the CNN Architecture (Matches your .pth file) ---
class AODnet(nn.Module):
    def __init__(self):
        super(AODnet, self).__init__()
        # These layers match the e_conv1 through e_conv5 in your file 
        self.e_conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True) 
        self.e_conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True) 
        self.e_conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True) 
        self.e_conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias=True) 
        self.e_conv5 = nn.Conv2d(12, 3, 3, 1, 1, bias=True) 
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        concat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.e_conv3(concat1))
        concat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.e_conv4(concat2))
        concat3 = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.relu(self.e_conv5(concat3))
        # AOD-Net Formula: K(x) * I(x) - K(x) + 1
        clean_image = self.relu((x5 * x) - x5 + 1)
        return clean_image

# --- 2. Load the Pre-trained Model ---
@st.cache_resource
def load_model():
    model = AODnet()
    # Loading the weights you provided [cite: 5, 10]
    model.load_state_dict(torch.load('aodnet-pretrained.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# --- 3. Streamlit UI Setup ---
st.set_page_config(page_title="AI Dehazing System", layout="wide")
st.title("CNN Image Dehazing System")
st.write("Powered by AOD-Net Deep Learning Model")

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload a Hazy Image", type=["jpg","png","jpeg"])
camera_file = st.camera_input("Or take a picture")

if camera_file is not None:
    uploaded_file = camera_file

# --- 4. Processing Logic ---
def dehaze_ai(image, model):
    # Prepare image for the CNN
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # Standard scaling
    ])
    
    img_tensor = transform(image).unsqueeze(0) # Add batch dimension
    
    with torch.no_grad():
        # Pass image through the 5 conv layers [cite: 1, 3]
        result_tensor = model(img_tensor)
    
    # Convert back to viewable image
    result = result_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result

# --- 5. Main Execution ---
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    model = load_model()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Hazy Image")
        st.image(image, use_container_width=True)

    with st.spinner("AI is analyzing atmospheric light..."):
        result = dehaze_ai(image, model)

    with col2:
        st.subheader("AI Dehazed Result")
        st.image(result, use_container_width=True)

    # Save to history
    if not any(np.array_equal(result, h) for h in st.session_state.history):
        st.session_state.history.append(result)

# History Section
if st.session_state.history:
    st.divider()
    st.subheader("Recent Cleared Images")
    cols = st.columns(4)
    for i, img in enumerate(reversed(st.session_state.history[-4:])):
        cols[i].image(img, caption=f"Result {len(st.session_state.history)-i}")
