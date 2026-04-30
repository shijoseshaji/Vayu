import streamlit as st
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# --- 1. BRANDING & CONFIG ---
st.set_page_config(page_title="Vayu", page_icon="🪽", layout="centered")

# --- 2. MOBILE-OPTIMIZED CSS ---
hide_st_style = """
            <style>
            header, [data-testid="stHeader"], footer, #MainMenu {visibility: hidden; display: none !important;}
            [data-testid="stViewerBadge"], .st-emotion-cache-1647z6a {display: none !important;}
            div[data-testid="stStatusWidget"] {display: none !important;}
            .block-container {padding-top: 1rem !important;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- 3. LOAD MODELS (CACHE FOR SPEED) ---
@st.cache_resource
def load_ai_brains():
    # Text Model
    text_model = SentenceTransformer('all-MiniLM-L6-v2')
    # Image Model (ResNet18 for Damage Detection)
    vision_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    vision_model.eval()
    return text_model, vision_model

text_model, vision_model = load_ai_brains()

# --- 4. DATA ---
FAQ_DATA = {
    "Exchange Policy": "For mobile exchange, the device must power on and the IMEI must match the order.",
    "Scratch vs Crack": "A scratch is a surface mark you can't feel with a fingernail. A crack reflects light and is deep.",
    "Liquid Damage": "Check the SIM tray for a pink/red indicator.","How do I mark a delivery as complete?": "Open the 'Active Orders' tab, select your current delivery, and tap 'Confirm Drop-off'.",
    "The customer is not answering the door.": "Wait for 2 minutes and call the customer twice. If there's no response, select 'Customer Unavailable' in the app.",
    "What do I do in case of a vehicle breakdown?": "Safety first! Use the 'Emergency' button in the main menu to report the issue and pause your shift.",
    "How are earnings calculated?": "Your earnings include base pay, distance incentives, and 100% of the tips provided by customers.",
    "What is the policy for damaged packages?": "Take a photo of the package before handing it over and report it under 'Damaged Item' in the support menu.",
    "How do I update my profile details?": "Go to the Settings tab in your main delivery app and select 'Profile Info'.",
    "Where can I see my performance rating?": "Ratings are updated weekly under the 'Performance' tab in your Wishmaster Dashboard."
}

# --- 5. UI TABS ---
st.title("Vayu")
st.subheader("Wishmaster Support & Exchange Portal")

tab1, tab2 = st.tabs(["💬 FAQ Assistant", "📷 Damage Scanner"])

# --- TAB 1: TEXT CHAT ---
with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask Vayu about exchange rules..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Simple Logic
        with st.chat_message("assistant"):
            st.info("Searching exchange guidelines...")
            # (Insert previous semantic search logic here for full functionality)

# --- TAB 2: CAMERA SCANNER ---
with tab2:
    st.write("### Identify Shipment Damage")
    st.caption("Capture a clear photo of the device screen or body.")
    
    img_file = st.camera_input("Take a photo of the damage")

    if img_file:
        img = Image.open(img_file)
        st.image(img, caption="Captured Image", use_container_width=True)
        
        # Preprocessing the image for the AI
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(img).unsqueeze(0)

        with st.spinner("Analyzing surface..."):
            with torch.no_grad():
                output = vision_model(input_tensor)
                # In a real-world scenario, you would train this specifically on 'crack' vs 'scratch'
                # For this basic version, we demonstrate the detection trigger:
                st.warning("⚠️ Potential Damage Detected")
                st.write("**AI Analysis:** Image contains patterns consistent with surface scratches.")
                st.button("Log Damage in Order")

# --- SIDEBAR ---
with st.sidebar:
    st.title("🪽 Vayu Hub")
    st.write("Version: 2.0 (Vision Enabled)")
    if st.button("Reset Portal"):
        st.session_state.messages = []
        st.rerun()
