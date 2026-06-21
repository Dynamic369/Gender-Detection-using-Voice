import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import streamlit as st
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import io
import sys
import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.model_cnn2d import GenderCNN2D

st.set_page_config(page_title="Gender Detection Using Voice ", layout="centered")

model_path = os.path.join(PROJECT_ROOT, 'models', 'gender_model_cnn2d.pth')
if os.path.exists(model_path):
    mod_time = os.path.getmtime(model_path)
    st.sidebar.success(f" 2D CNN Loaded:\n{datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')}")

@st.cache_resource
def load_model():
    model = GenderCNN2D()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

model = load_model()

def process_and_predict_2d(audio_bytes):
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
    
    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=30)
    if len(audio) < 100:
        return "Error", 0.0, 0.0, None, None, None
        
    # Extract 2D Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=3000)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    

    MAX_TIME_STEPS = 86
    if mel_db.shape[1] < MAX_TIME_STEPS:
        pad_width = MAX_TIME_STEPS - mel_db.shape[1]
        mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=-80.0)
    else:
        mel_db = mel_db[:, :MAX_TIME_STEPS]
        
    mel_normalized = (mel_db + 80.0) / 80.0 
    
    tensor_X = torch.Tensor(mel_normalized).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(tensor_X).item()
        
    label = "Female" if prediction >= 0.5 else "Male"
    confidence = prediction if prediction >= 0.5 else (1 - prediction)

    return label, round(confidence * 100, 2), prediction, mel_normalized, audio, sr

def plot_combined_analysis(mel_normalized, audio, sr):
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    
    time_axis = np.linspace(0, len(audio) / sr, num=len(audio))
    ax1.plot(time_axis, audio, color='cyan')
    ax1.set_title('Raw Audio Waveform (Time Domain)')
    ax1.set_xlabel('Time (Seconds)')
    ax1.set_ylabel('Amplitude')
    
  
    img = ax2.imshow(mel_normalized, aspect='auto', origin='lower', cmap='magma')
    ax2.set_title('2D Peak-Normalized Mel-Spectrogram (Frequency Domain - CNN Input)')
    ax2.set_xlabel('Time Steps (Fixed at 86)')
    ax2.set_ylabel('Mel Frequency Bins (128)')
    fig.colorbar(img, ax=ax2, format="%+2.0f dB")
    
    fig.tight_layout()
    return fig

st.title("Gender Recognition Using Voice.")
st.markdown("### Mathematical Signal Analysis")

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
    
    if st.button("Predict Gender & Analyze Spectrogram"):
        with st.spinner("Extracting spatial features..."):
            audio_bytes = audio_file.getvalue()
            
            
            label, confidence, raw_pred, mel_matrix, raw_audio, sr = process_and_predict_2d(audio_bytes)
            
            if label == "Error":
                st.error("Audio too quiet or too short. Please speak louder.")
            else:
                st.success("Analysis Complete!")
                col1, col2, col3 = st.columns(3)
                col1.metric("Predicted Gender", label)
                col2.metric("Confidence", f"{confidence}%")
                col3.metric("Raw Sigmoid Output", f"{raw_pred:.4f}")
                
                st.divider()
                st.markdown("Represetation of the Your Voice Signal.")
               
                fig = plot_combined_analysis(mel_matrix, raw_audio, sr)
                st.pyplot(fig)