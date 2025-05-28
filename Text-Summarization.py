# Import required libraries
import streamlit as st
from transformers import pipeline

# Function to load the summarization pipeline using a small, fast model
# This function is cached to avoid reloading the model every time the app runs
@st.cache_resource
def load_model():
    # Load the summarization pipeline using a distilled (lightweight) version of BART
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Load the model once and reuse it throughout the app
summarizer = load_model()

# --------- Streamlit App UI ---------

# Title of the app
st.title("📝 Text Summarizer App")

# Text input area for user to paste content
text = st.text_area("Paste your text here:(limit max 150 Words)", height=300)

# Button to trigger summarization
if st.button("Summarize"):
    # Check if the text input is not empty
    if text.strip():
        # Show a spinner while the model processes the input
        with st.spinner("Summarizing..."):
            # Perform summarization with specified parameters
            summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            # Display the summarized output
            st.subheader("🧠 Summary:")
            st.write(summary)
    else:
        # Show a warning if no input was provided
        st.warning("Please enter some text.")

