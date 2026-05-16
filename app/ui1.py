import streamlit as st
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import sys

# 1. Point Python to your src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model import GenderCNN

st.set_page_config(page_title="Voice Gender Recognition", layout="centered", page_icon="🎙️")

# 2. Load the model 
@st.cache_resource
def load_model():
    model = GenderCNN()
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'gender_model.pth'))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# 3. Audio processing logic (Now returns the audio array for plotting)
def process_and_predict(audio_bytes, threshold):
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, duration=3.0)
    
    # Dynamic Silence Check
    if np.max(np.abs(audio)) < threshold:
        return "Silence Detected", 0.0, None, None
        
    target_length = 22050 * 3
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
        
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs = mfccs[np.newaxis, np.newaxis, ...] 
    tensor_X = torch.Tensor(mfccs)
    
    tensor_X = (tensor_X - tensor_X.mean()) / (tensor_X.std() + 1e-8)
    
    with torch.no_grad():
        prediction = model(tensor_X).item()
        
    label = "Female" if prediction >= 0.5 else "Male"
    confidence = prediction if prediction >= 0.5 else (1 - prediction)
    
    # We now return the raw audio and sample rate so we can plot them!
    return label, round(confidence * 100, 2), audio, sr

# 4. NEW: Visualization Engine
def plot_audio_features(audio, sr):
    # Create a figure with two subplots stacked vertically
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
    
    # Top Plot: The raw waveform
    librosa.display.waveshow(audio, sr=sr, ax=ax[0], color='cyan')
    ax[0].set_title('Raw Audio Waveform (Time Domain)')
    ax[0].set_ylabel('Amplitude')
    
    # Bottom Plot: The Mel-Spectrogram
    # We calculate a high-res spectrogram specifically for the user to look at
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax[1], cmap='magma')
    fig.colorbar(img, ax=ax[1], format='%+2.0f dB')
    ax[1].set_title('Mel-Spectrogram (Frequency Domain - CNN Input)')
    
    fig.tight_layout()
    return fig

# 5. The User Interface
st.title("🎙️ Voice Gender Recognition System")
st.markdown("Record an audio clip to classify the speaker's gender and visualize the signal processing.")

with st.expander("⚙️ Advanced Settings (Microphone Calibration)"):
    st.write("Adjust this if background noise is triggering false predictions.")
    user_threshold = st.slider("Silence Threshold", min_value=0.01, max_value=0.15, value=0.02, step=0.01)

audio_file = st.audio_input("Record your audio")

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")
    
    if st.button("Predict Gender & Analyze Signal"):
        with st.spinner("Analyzing audio frequencies and rendering graphs..."):
            
            audio_bytes = audio_file.getvalue()
            
            # Catch all 4 variables returned by the updated function
            label, confidence, audio_data, sr = process_and_predict(audio_bytes, threshold=user_threshold)
            
            if label == "Silence Detected":
                st.warning(f"Peak amplitude was below the {user_threshold} threshold. Please speak clearly or lower the threshold.")
            else:
                st.success("Analysis Complete!")
                
                # Create a nice 2-column layout for the metrics
                col1, col2 = st.columns(2)
                col1.metric(label="Predicted Gender", value=label)
                col2.metric(label="Confidence", value=f"{confidence}%")
                
                st.divider()
                st.markdown("Mathematical Signal Analysis")
                st.write("These are the exact frequency patterns extracted from your voice and fed into the Convolutional Neural Network.")
                
                # Render the plots!
                fig = plot_audio_features(audio_data, sr)
                st.pyplot(fig)