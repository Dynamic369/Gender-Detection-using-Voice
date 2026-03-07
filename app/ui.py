import streamlit as st
import requests

# Define the FastAPI backend URL
API_URL = "http://127.0.0.1:8000/predict/"

st.set_page_config(page_title="Voice Gender Recognition", layout="centered")

st.title("🎙️ Voice Gender Recognition System")
st.markdown("Upload a 3-second audio clip to classify the speaker's gender using a custom PyTorch CNN.")

# File uploader widget
uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    if st.button("Predict Gender"):
        with st.spinner("Analyzing audio frequencies..."):
            # Send the file to the FastAPI backend
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "audio/wav")}
            try:
                response = requests.post(API_URL, files=files)
                response.raise_for_status()
                
                result = response.json()
                label = result["prediction"]
                confidence = result["confidence"]
                
                # Display results
                st.success("Analysis Complete!")
                st.metric(label="Predicted Gender", value=label)
                st.metric(label="Confidence", value=f"{confidence}%")
                
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to backend API: {e}")