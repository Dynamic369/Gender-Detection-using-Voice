import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import streamlit as st
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import sys
import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.model import GenderMLP

st.set_page_config(page_title="Voice Gender Recognition", layout="centered", page_icon="🎙️")

model_path = os.path.join(PROJECT_ROOT, 'models', 'gender_model.pth')
if os.path.exists(model_path):
    mod_time = os.path.getmtime(model_path)
    st.sidebar.success(f"🧠 Robust RAVDESS MLP Loaded:\n{datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')}")

@st.cache_resource
def load_model():
    model = GenderMLP()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

model = load_model()

def process_and_predict(audio_bytes):
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
    
    # 1. Trim silence 
    audio, _ = librosa.effects.trim(audio, top_db=30)
    if len(audio) < 100:
        return "Error", 0.0, 0.0, audio, sr
        
    # 2. Extract Mel and Average
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    mel_mean = np.mean(mel_spec, axis=1)
    
    # 3. Peak Normalization [0.0 to 1.0]
    mel_db = librosa.power_to_db(mel_mean, ref=np.max)
    mel_normalized = (mel_db + 80.0) / 80.0 
    
    # Shape for PyTorch (1 Batch, 128 Features)
    tensor_X = torch.Tensor(mel_normalized).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(tensor_X).item()
        
    label = "Female" if prediction >= 0.5 else "Male"
    confidence = prediction if prediction >= 0.5 else (1 - prediction)
    
    return label, round(confidence * 100, 2), prediction, audio, sr

def plot_audio_features(mel_normalized):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(mel_normalized, color='cyan')
    ax.fill_between(range(len(mel_normalized)), mel_normalized, alpha=0.3, color='cyan')
    ax.set_title('1D Peak-Normalized Frequency Profile (Neural Network Input)')
    ax.set_xlabel('Mel Frequency Bins (Low Pitch -> High Pitch)')
    ax.set_ylabel('Normalized Amplitude [0.0 - 1.0]')
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    return fig

st.title("🎙️ Voice Gender Recognition System")
# tab1, tab2 = st.tabs(["🎙️ Record Audio", "📁 Upload File"])
audio_file = None

# with tab1:
#     recorded_audio = st.audio_input("Record your voice")
#     if recorded_audio is not None:
#         audio_file = recorded_audio

# with tab2:
uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg"])
if uploaded_audio is not None:
    audio_file = uploaded_audio

if audio_file is not None:
    st.audio(audio_file)
    
    if st.button("Predict Gender & Analyze Signal"):
        with st.spinner("Extracting biological pitch..."):
            audio_bytes = audio_file.getvalue()
            label, confidence, raw_pred, audio_data, sr = process_and_predict(audio_bytes)
            
            if label == "Error":
                st.error("Audio too quiet or too short. Please speak louder.")
            else:
                st.success("Analysis Complete!")
                col1, col2, col3 = st.columns(3)
                col1.metric("Predicted Gender", label)
                col2.metric("Confidence", f"{confidence}%")
                col3.metric("Raw Output", f"{raw_pred:.4f}")
                
                st.divider()
                
                mel = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128, fmax=8000)
                mel_db = librosa.power_to_db(np.mean(mel, axis=1), ref=np.max)
                mel_norm = (mel_db + 80.0) / 80.0
                
                fig = plot_audio_features(mel_norm)
                st.pyplot(fig)