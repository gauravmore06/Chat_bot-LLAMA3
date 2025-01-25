import os
from dotenv import load_dotenv
from langchain_community.llms import ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# Set LangChain API configuration
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Design the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked."),
        ("user", "Question: {question}"),
    ]
)

# Streamlit Framework for the UI
st.title("LangChain Demo With LLAMA3")
input_text = st.text_input("Ask any Question")  # Use text_input instead of time_input

# Initialize the Ollama LLM
try:
    llm = ollama.Ollama(model="gemma:2b")
except AttributeError:
    st.error("Error: Could not initialize Ollama model. Ensure the `ollama` library is correctly installed.")

# Set up output parser and processing chain
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Process the input and display the output
if input_text:
    try:
        response = chain.invoke({"question": input_text})
        st.write(response)
    except Exception as e:
        st.error(f"An error occurred: {e}")