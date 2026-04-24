import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

# --- 1. BRANDING & CONFIG ---
st.set_page_config(page_title="Vayu", page_icon="🪽", layout="centered")

# --- 2. AGGRESSIVE CSS FOR MOBILE & DESKTOP BRANDING REMOVAL ---
hide_st_style = """
            <style>
            /* 1. Hide the top header bar and deploy button */
            header, [data-testid="stHeader"] {
                visibility: hidden;
                display: none;
            }

            /* 2. Hide the main hamburger menu (top right) */
            #MainMenu {visibility: hidden;}

            /* 3. Hide the footer (Made with Streamlit) */
            footer {visibility: hidden; display: none !important;}

            /* 4. Hide the floating 'Viewer Badge' (Creator icon) on mobile */
            [data-testid="stViewerBadge"], .viewerBadge_container__1QSob, .st-emotion-cache-1647z6a {
                display: none !important;
                visibility: hidden !important;
            }

            /* 5. Hide the status widget (Hosted by Streamlit) */
            div[data-testid="stStatusWidget"], [data-testid="stStatusWidget"] {
                display: none !important;
                visibility: hidden !important;
            }

            /* 6. Remove the annoying colored line at the top */
            [data-testid="stDecoration"] {
                display: none !important;
            }

            /* 7. Tighten up the top margin so 'Vayu' title starts higher on mobile screens */
            .block-container {
                padding-top: 1.5rem !important;
                padding-bottom: 1rem !important;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- 3. FAQ DATA ---
FAQ_DATA = {
    "How do I mark a delivery as complete?": "Open the 'Active Orders' tab, select your current delivery, and tap 'Confirm Drop-off'.",
    "The customer is not answering the door.": "Wait for 2 minutes and call the customer twice. If there's no response, select 'Customer Unavailable' in the app.",
    "What do I do in case of a vehicle breakdown?": "Safety first! Use the 'Emergency' button in the main menu to report the issue and pause your shift.",
    "How are earnings calculated?": "Your earnings include base pay, distance incentives, and 100% of the tips provided by customers.",
    "What is the policy for damaged packages?": "Take a photo of the package before handing it over and report it under 'Damaged Item' in the support menu.",
    "How do I update my profile details?": "Go to the Settings tab in your main delivery app and select 'Profile Info'.",
    "Where can I see my performance rating?": "Ratings are updated weekly under the 'Performance' tab in your Wishmaster Dashboard.","How can I check if a shipment is eligible for OBD?":"Go to the runsheet in BYOD, search for the tracking ID or scan the QR code on the shipment, the tracking id page will show the shipment details, including OBD."
}

# --- 4. LOAD AI BRAIN ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()
faq_questions = list(FAQ_DATA.keys())
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

# --- 5. VAYU HEADERS ---
st.title("Vayu")
st.subheader("Giving wings to every Wishmaster.")
st.caption("AI-powered assistant for instant delivery help")
st.divider()

# --- 6. CHAT HISTORY LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 7. CHAT INPUT & AI LOGIC ---
if prompt := st.chat_input("How can Vayu help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    query_embedding = model.encode(prompt, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, faq_embeddings)[0]
    best_match_idx = torch.argmax(cos_scores).item()
    top_score = cos_scores[best_match_idx].item()

    with st.chat_message("assistant"):
        if top_score > 0.45:
            response = FAQ_DATA[faq_questions[best_match_idx]]
            st.info(response)
        else:
            response = "I couldn't find a specific answer. Connect with a Fleet Supervisor?"
            st.warning(response)
            if st.button("Contact Support"):
                st.write("📞 1-800-VAYU-HELP")
            
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- 8. SIDEBAR ---
with st.sidebar:
    st.title("🪽 Wishmaster Portal")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
