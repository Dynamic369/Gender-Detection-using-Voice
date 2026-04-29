from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import torch
import librosa
import numpy as np
import io
import sys
import os

# Adjust path to import the model class from the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model import GenderCNN

app = FastAPI(title="Gender Voice Recognition API")

# Load model weights
model = GenderCNN()
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'gender_model.pth'))
model.load_state_dict(torch.load(model_path))
model.eval()

def preprocess_audio(audio_bytes):
    """Replicates the librosa preprocessing on live audio data."""
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, duration=3.0)
    
    #Silence detection(GateKeeper)
    max_apmlitude = np.max(np.abs(audio))
    if max_apmlitude < 0.02:
        return None
    
    target_length = 22050 * 3
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
        
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    
    # Shape it for PyTorch: (Batch=1, Channels=1, Height=40, Width=130)
    mfccs = mfccs[np.newaxis, np.newaxis, ...] 
    return torch.Tensor(mfccs)

@app.post("/predict/")
async def predict_gender(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    
    # Preprocess and predict
    tensor_X = preprocess_audio(audio_bytes)
    # handle the silent audio
    if tensor_X is None:
        return {
            "filename":file.filename,
            "prediction": "Silence Detected",
            "confidence":0.0
        }
    with torch.no_grad():
        prediction = model(tensor_X).item()
        
    # Interpret the sigmoid output (0 = Male, 1 = Female)
    label = "Female" if prediction >= 0.5 else "Male"
    confidence = prediction if prediction >= 0.5 else (1 - prediction)
    
    return {
        "filename": file.filename,
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    }