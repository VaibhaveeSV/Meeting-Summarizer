import streamlit as st
from summarizer import process_transcript
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Meeting Summarizer", layout="wide")
st.title(" Gemini-Powered Meeting Summarizer")

st.markdown("Paste your meeting transcript below")

transcript = st.text_area("Meeting Transcript", height=300)

if st.button("Summarize"):
    if transcript.strip() == "":
        st.warning("Please enter a meeting transcript.")
    else:
        with st.spinner("Processing with Gemini..."):
            result = process_transcript(transcript)
        st.subheader(" AI Output")
        st.markdown(result)

