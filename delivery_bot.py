import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import requests
import base64
import io
import cv2
import numpy as np
from PIL import Image

# --- 1. BRANDING & CONFIG ---
st.set_page_config(page_title="Vayu", page_icon="🪽", layout="centered")

# --- 2. MOBILE-OPTIMIZED UI CLEANUP ---
hide_st_style = """
            <style>
            header, [data-testid="stHeader"], footer, #MainMenu {visibility: hidden; display: none !important;}
            [data-testid="stViewerBadge"], .st-emotion-cache-1647z6a {display: none !important;}
            div[data-testid="stStatusWidget"] {display: none !important;}
            [data-testid="stDecoration"] {display: none !important;}
            .block-container {padding-top: 1.5rem !important;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- 3. CONFIG & API KEYS ---
try:
    GCLOUD_API_KEY = st.secrets["GCLOUD_API_KEY"]
except:
    GCLOUD_API_KEY = None # App will still run FAQ, but Vision will show warning

# --- 4. LOAD AI BRAINS ---
@st.cache_resource
def load_models():
    text_model = SentenceTransformer('all-MiniLM-L6-v2')
    return text_model

text_model = load_models()

# --- 5. UNIFIED FAQ DATA (Delivery + Exchange) ---
FAQ_DATA = {
    # Delivery Related
    "How do I mark a delivery as complete?": "Open the 'Active Orders' tab, select your current delivery, and tap 'Confirm Drop-off'.",
    "The customer is not answering the door.": "Wait for 2 minutes and call the customer twice. If no response, select 'Customer Unavailable'.",
    "What to do in case of vehicle breakdown?": "Safety first! Use the 'Emergency' button in the menu to report the issue and pause your shift.",
    
    # Exchange Related
    "What is the mobile exchange policy?": "Device must power on, IMEI must match, and the screen must be free of major cracks.",
    "Difference between scratch and crack?": "A scratch is surface-level and doesn't reflect light. A crack is a deep fracture that feels sharp to the touch.",
    "What if the device has liquid damage?": "Check the SIM tray indicator. If red/pink, the exchange must be rejected as 'Liquid Damaged'.",
    "Customer doesn't have the original charger?": "Exchanges are allowed without chargers, but a value deduction may apply per app guidelines."
}
faq_questions = list(FAQ_DATA.keys())
faq_embeddings = text_model.encode(faq_questions, convert_to_tensor=True)

# --- 6. VAYU HEADER ---
st.title("Vayu")
st.subheader("Vayu: Giving wings to every Wishmaster.")
st.caption("AI-powered assistant for Delivery & Mobile Exchange")
st.divider()

# --- 7. APP TABS ---
tab1, tab2 = st.tabs(["💬 FAQ Assistant", "📷 Damage Scanner"])

with tab1:
    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask Vayu about delivery or exchange..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Semantic Search Logic
        query_embedding = text_model.encode(prompt, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, faq_embeddings)[0]
        best_match_idx = torch.argmax(cos_scores).item()
        
        with st.chat_message("assistant"):
            if cos_scores[best_match_idx].item() > 0.45:
                response = FAQ_DATA[faq_questions[best_match_idx]]
                st.info(response)
            else:
                response = "I'm not sure about that. Would you like to connect with a Supervisor?"
                st.warning(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

with tab2:
    st.write("### Exchange Inspection")
    if GCLOUD_API_KEY is None:
        st.error("Vision API Key not configured. Please add it to Streamlit Secrets.")
    else:
        img_file = st.camera_input("Take a clear photo of the device")
        
        if img_file:
            image = Image.open(img_file)
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            vision_url = f"https://vision.googleapis.com/v1/images:annotate?key={GCLOUD_API_KEY}"
            request_data = {
                "requests": [{"image": {"content": base64_image},
                "features": [{"type": "OBJECT_LOCALIZATION", "maxResults": 10}]}]
            }

            with st.spinner("Vayu is scanning surface..."):
                try:
                    res = requests.post(vision_url, json=request_data).json()
                    objects = res['responses'][0].get('localizedObjectAnnotations', [])
                    
                    cv_img = np.array(image.convert('RGB'))
                    h, w, _ = cv_img.shape
                    found_damage = False

                    for obj in objects:
                        label = obj['name'].lower()
                        if any(k in label for k in ["crack", "scratch", "dent", "damage"]):
                            found_damage = True
                            box = obj['boundingPoly']['normalizedVertices']
                            x1, y1 = int(box[0]['x']*w), int(box[0]['y']*h)
                            x2, y2 = int(box[2]['x']*w), int(box[2]['y']*h)
                            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                            cv2.putText(cv_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

                    if found_damage:
                        st.warning("Potential Damage Identified:")
                        st.image(cv_img, use_container_width=True)
                    else:
                        st.success("No major surface damage detected.")
                        st.image(image, use_container_width=True)
                except:
                    st.error("Scanning failed. Please check network or API limits.")

# --- 8. SIDEBAR ---
with st.sidebar:
    st.title("🪽 Portal")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
