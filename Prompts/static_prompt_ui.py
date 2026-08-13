from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.5,
    max_output_tokens = 100
)

st.header("Research Tool")

user_input = st.text_input("Prompt Here")

if st.button("Summarize"):
    result = model.invoke(user_input)
    st.write(result.text)