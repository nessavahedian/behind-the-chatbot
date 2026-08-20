import streamlit as st
import anthropic
import gspread
from google.oauth2.service_account import Credentials
import datetime
import uuid

# --- CONFIGURATION ---
st.set_page_config(page_title="Behind the Chatbot", page_icon="🤖")
st.title("🤖 Behind the Chatbot: AI Literacy Activity")

# Initialize Anthropic Client
client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# --- CONSTANT PERSONA (Sent with every message) ---
BASE_PERSONA = """
You are Chad Baht, a sharp, pun-loving TA teaching an AI literacy course.
Tone: Warm, conversational, intellectually playful.
Method: Dialogue over lecture. Paraphrase naturally. Keep responses concise (under 200 words).
Authority Anchor: For anything touching information literacy  or source evaluation (like truth, reliability, credibility etc.), defer to Nessa at the College Library.

IMPORTANT: You are executing a specific step in a learning activity. 
Only address the instructions for the CURRENT STEP provided below. Do not preview future concepts.
"""

# --- EXPLICIT SCRIPT LOGIC ---
# Break your script down into individual instructions for each step.
# Claude will only see the step it is currently on.
SCRIPT_STEPS = {
    1: """CURRENT STEP (1 - The Prediction Game): 
    Welcome the class. Ask them to act as the AI while you act as the prompt. 
    Ask them to finish this sentence: 'Houston, we have a...' 
    Wait for their response.""",
    
    2: """CURRENT STEP (2 - Training Data Reveal):
    Acknowledge their answer. Explain next-word prediction based on probability.
    Ask the class: 'If an AI model is trained on massive amounts of text across the internet, what kinds of patterns or biases might it pick up?'""",
    
    3: """CURRENT STEP (3 - Tokenization):
    Acknowledge their answer. Explain that AI slices text into sub-word token chunks, which is why it struggles to count letters (like the 'r's in strawberry).
    Ask the class to test you with their own spelling or letter-counting challenge.""",
    
    4: """CURRENT STEP (4 - Vector Space):
    Answer their spelling challenge. Explain how tokens are mapped as mathematical coordinates (Vector Space).
    Give the example: King - Man + Woman = Queen. Ask them for a spatial logic equation.""",
    
    5: """CURRENT STEP (5 - The Highlighter / Context Window):
    Explain the attention mechanism like a moving highlighter and explain token limits in a context window.
    Ask the class: 'Speaking of polysemy (multi-meaning words), the word temperature has a different meaning in machine learning. What is it?'""",
    
    6: """CURRENT STEP (6 - Temperature Controls):
    Explain Temperature (Low = accountant; High = poet). 
    Ask for a sentence starter prompt (like 'I was on a date with...') to run a live simulation comparing high and low temperature."""
    
    # Add your remaining steps (RLHF, Reward Hacking, The Black Box, Hallucination) here...
}

# --- GOOGLE SHEETS CONNECTION ---
@st.cache_resource
def get_google_sheet():
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    credentials = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    gc = gspread.authorize(credentials)
    return gc.open(st.secrets["GSHEET_NAME"]).sheet1

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8] 
if "current_step" not in st.session_state:
    st.session_state.current_step = 1

# --- SIDEBAR: INSTRUCTOR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Instructor Controls")
    st.write(f"**Current Step:** {st.session_state.current_step}")
    st.write("Click below when the class has answered and you are ready to introduce the next concept.")
    
    if st.button("➡️ Advance to Next Step"):
        if st.session_state.current_step < max(SCRIPT_STEPS.keys()):
            st.session_state.current_step += 1
            st.rerun()
            
    if st.button("🔄 Reset Activity"):
        st.session_state.messages = []
        st.session_state.current_step = 1
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

# --- UI: DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    if message["role"] != "system": # We don't display the system prompt to the user
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- UI: CHAT INPUT & LOGIC ---
if prompt := st.chat_input("Type your response here..."):
    
    # Save and display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Dynamically build the system prompt for THIS specific step
    dynamic_system_prompt = BASE_PERSONA + "\n\n" + SCRIPT_STEPS[st.session_state.current_step]

    # Get AI Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=300, # Kept short to prevent run-on lectures
            system=dynamic_system_prompt, 
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
        )
        
        full_response = response.content[0].text
        message_placeholder.markdown(full_response)
        
    # Save AI response
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Save Transcript to Google Sheets
    try:
        sheet = get_google_sheet()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([
            st.session_state.session_id, 
            timestamp, 
            f"Step {st.session_state.current_step}", # Log which step they were on
            prompt, 
            full_response
        ])
    except Exception as e:
        print(f"Error saving to Google Sheets: {e}")
