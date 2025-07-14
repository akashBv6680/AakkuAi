import streamlit as st
from langchain_together import ChatTogether
import os
import pandas as pd

# Set API key
api_key = st.secrets["TOGETHER_API_KEY"]

# Initialize LLM with Together AI endpoint
llm = ChatTogether(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0,
    max_tokens=512,
    api_key=api_key
)

# Streamlit app
st.title("Aakku AI - Conversational Interface")

# File upload feature
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Process the uploaded file
    data = pd.read_csv(uploaded_file)
    st.write(data.head())

# Conversational interface
user_input = st.text_input("You: ")
if user_input:
    # Get LLM response
    response = llm.invoke(user_input)
    st.write("Aakku AI:", response)
