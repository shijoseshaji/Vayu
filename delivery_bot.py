import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

# --- 1. BRANDING & CONFIG ---
st.set_page_config(page_title="Vayu", page_icon="🪽")

# --- 2. FAQ DATA ---
# Add your specific Wishmaster FAQs here
FAQ_DATA = {
    "How do I mark a delivery as complete?": "Open the 'Active Orders' tab, select your current delivery, and tap 'Confirm Drop-off'.",
    "The customer is not answering the door.": "Wait for 2 minutes and call the customer twice. If there's no response, select 'Customer Unavailable' in the app.",
    "What do I do in case of a vehicle breakdown?": "Safety first! Use the 'Emergency' button in the main menu to report the issue and pause your shift.",
    "How are earnings calculated?": "Your earnings include base pay, distance incentives, and 100% of the tips provided by customers.",
    "What is the policy for damaged packages?": "Take a photo of the package before handing it over and report it under 'Damaged Item' in the support menu."
}

# --- 3. LOAD AI MODEL ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()
faq_questions = list(FAQ_DATA.keys())
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

# --- 4. UI HEADERS (VAYU BRANDING) ---
st.title("Vayu")
st.subheader("Giving wings to every Wishmaster.")
st.write("AI-powered assistant for instant delivery help")
st.divider()

# --- 5. CHAT SYSTEM ---
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask Vayu a question..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Search Logic
    query_embedding = model.encode(prompt, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, faq_embeddings)[0]
    best_match_idx = torch.argmax(cos_scores).item()
    top_score = cos_scores[best_match_idx].item()

    # Generate Assistant Response
    with st.chat_message("assistant"):
        if top_score > 0.45:
            response = FAQ_DATA[faq_questions[best_match_idx]]
            st.success(response)
        else:
            response = "I couldn't find a specific answer for that. Would you like to connect with a Fleet Supervisor?"
            st.warning(response)
            if st.button("Connect to Support"):
                st.info("Dialing support line: 1-800-VAYU-HELP")
            
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- SIDEBAR ---
with st.sidebar:
    st.title("🪽 Wishmaster Portal")
    st.info("This AI is trained to help you fly through your deliveries.")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
