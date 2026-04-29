import streamlit as st
import requests

# Define the FastAPI backend URL
API_URL = "https://gender-detection-using-voice.onrender.com/predict/"

st.set_page_config(page_title="Voice Gender Recognition", layout="centered")

st.title("🎙️ Voice Gender Recognition System")
st.markdown("Record the audio clip to classify the speaker's gender using a custom PyTorch CNN.")

#Taking the audio input
audio_file = st.audio_input("Record your audio")

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")
    
    if st.button("Predict Gender"):
        with st.spinner("Analyzing audio frequencies..."):
            # Send the file to the FastAPI backend
            files = {"file": (audio_file.name, audio_file.getvalue(), "audio/wav")}
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