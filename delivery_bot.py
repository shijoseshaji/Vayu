import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

# --- 1. BRANDING & CONFIG ---
# This sets the browser tab title
st.set_page_config(page_title="Vayu", page_icon="🪽", layout="centered")

# --- 2. UI CLEANUP (REMOVING STREAMLIT BRANDING) ---
# This CSS hides the menu, footer, deploy button, and the 'Hosted by/Creator' badges
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stDeployButton {display:none;}
            #stDecoration {display:none;}
            
            /* Hides the 'Hosted by' and 'Creator' status widgets */
            div[data-testid="stStatusWidget"] {display: none !important;}
            [data-testid="stViewerBadge"] {display: none !important;}
            .viewerBadge_container__1QSob {display: none !important;}
            
            /* Professional spacing */
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- 3. FAQ DATA (WISHMASTER KNOWLEDGE BASE) ---
FAQ_DATA = {
    "How do I mark a delivery as complete?": "Open the 'Active Orders' tab, select your current delivery, and tap 'Confirm Drop-off'.",
    "The customer is not answering the door.": "Wait for 2 minutes and call the customer twice. If there's no response, select 'Customer Unavailable' in the app.",
    "What do I do in case of a vehicle breakdown?": "Safety first! Use the 'Emergency' button in the main menu to report the issue and pause your shift.",
    "How are earnings calculated?": "Your earnings include base pay, distance incentives, and 100% of the tips provided by customers.",
    "What is the policy for damaged packages?": "Take a photo of the package before handing it over and report it under 'Damaged Item' in the support menu.",
    "How do I update my profile details?": "Go to the Settings tab in your main delivery app and select 'Profile Info'.",
    "Where can I see my performance rating?": "Ratings are updated weekly under the 'Performance' tab in your Wishmaster Dashboard."
}

# --- 4. LOAD AI BRAIN ---
@st.cache_resource
def load_model():
    # MiniLM is fast and lightweight for mobile performance
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

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 7. CHAT INPUT & AI LOGIC ---
if prompt := st.chat_input("How can Vayu help you today?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Use Semantic Search to find best FAQ match
    query_embedding = model.encode(prompt, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, faq_embeddings)[0]
    best_match_idx = torch.argmax(cos_scores).item()
    top_score = cos_scores[best_match_idx].item()

    # Generate Response
    with st.chat_message("assistant"):
        if top_score > 0.45:  # Confidence threshold
            response = FAQ_DATA[faq_questions[best_match_idx]]
            st.info(response)
        else:
            response = "I couldn't find a specific answer in the handbook. Would you like to connect with a Fleet Supervisor?"
            st.warning(response)
            if st.button("Contact Support"):
                st.write("📞 Calling Supervisor: 1-800-VAYU-HELP")
            
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- 8. SIDEBAR ---
with st.sidebar:
    st.title("🪽 Wishmaster Portal")
    st.write("This tool is designed to provide real-time support during your delivery run.")
    st.divider()
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
