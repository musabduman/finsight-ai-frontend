import streamlit as st

def get_api_keys():
    return {
        "gemini": st.session_state.get("gemini_key", ""),
        "ollama": st.session_state.get("ollama_key", "")
    }