import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

# --- CONFIGURATION & UI ---
st.set_page_config(page_title="Delivery Support AI", page_icon="📦")

# 1. Pre-fed FAQ Database (Key: Question, Value: Answer)
FAQ_DATA = {
    "How do I mark a delivery as complete?": "Go to 'Active Tasks', select the order, and tap 'Confirm Drop-off'. Ensure you are within 50 meters of the location.",
    "The customer is not answering the door.": "Wait for 2 minutes. Try calling them via the app. If no response after 2 attempts, tap 'Customer Unavailable' to notify support.",
    "My vehicle broke down during delivery.": "Safety first! Move to a safe spot. Use the 'Emergency' button in the menu to report a breakdown and pause incoming orders.",
    "I received a cash payment, what do I do?": "Collect the exact amount. The app will update your 'Cash on Hand' balance. You must deposit this at a partner kiosk by end-of-day.",
    "The delivery address is incorrect.": "Ask the customer for the correct address via chat. If it's more than 2km away, contact Support to adjust your payout.",
    "How do I view my daily earnings?": "Tap your profile picture > Earnings. You can see a breakdown of base pay, tips, and bonuses there.",
    "What is the policy for damaged items?": "If a package is damaged before delivery, take a clear photo and report it under 'Damaged Goods' before handing it to the customer."
}

# 2. Load AI Model (Semantic Search)
@st.cache_resource
def load_model():
    # 'all-MiniLM-L6-v2' is very fast and runs locally on your CPU
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()
faq_questions = list(FAQ_DATA.keys())
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

# --- CHAT INTERFACE ---
st.title("📦 Delivery Partner Support")
st.caption("AI-powered assistant for instant delivery help")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about delivery, payments, or app issues..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI Logic: Find the most similar FAQ
    query_embedding = model.encode(prompt, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, faq_embeddings)[0]
    best_match_idx = torch.argmax(cos_scores).item()
    top_score = cos_scores[best_match_idx].item()

    # Generate Response
    with st.chat_message("assistant"):
        if top_score > 0.45:  # Confidence threshold
            response = FAQ_DATA[faq_questions[best_match_idx]]
            st.markdown(response)
        else:
            response = "I'm not 100% sure about that policy. Would you like to be connected to a live human supervisor?"
            st.warning(response)
            st.button("Connect to Supervisor")
            
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})