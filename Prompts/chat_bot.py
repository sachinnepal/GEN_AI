import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

    .main {
        background-color: #f8f9fa;
    }

    .title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #6c757d;
        margin-bottom: 30px;
    }

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Model
# -----------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    max_output_tokens=100
)

# -----------------------------
# Session State
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a helpful assistant.")
    ]

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.title("🤖 AI Assistant")

    st.write(
        "A simple chatbot built with "
        "LangChain + Gemini + Streamlit."
    )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.chat_history = [
            SystemMessage(content="You are a helpful assistant.")
        ]

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.caption("Model: Gemini")
    st.caption("Framework: LangChain")

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="title">🤖 AI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask me anything</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Display Chat History
# -----------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# Chat Input
# -----------------------------
user_input = st.chat_input("Type your message...")

if user_input:

    # User message
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append(
        HumanMessage(content=user_input)
    )

    # AI response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            result = model.invoke(
                st.session_state.chat_history
            )

            # Extract text from Gemini response
            if isinstance(result.content, list):
                response = "".join(
                    block["text"]
                    for block in result.content
                    if block.get("type") == "text"
                )
            else:
                response = result.content

            st.markdown(response)

    # Save AI response
    st.session_state.chat_history.append(
        AIMessage(content=response)
    )

    # Save UI messages
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })