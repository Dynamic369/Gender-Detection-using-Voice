import os
import librosa
import numpy as np

SAMPLE_RATE = 22050 
N_MELS = 128 
MAX_TIME_STEPS = 86 

def extract_features_2d(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    

    audio, _ = librosa.effects.trim(audio, top_db=30)
    if len(audio) < 100:
        return None
        
  
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, fmax=3000)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    if mel_db.shape[1] < MAX_TIME_STEPS:
        # Pad the time axis (axis 1) with -80 dB (absolute silence)
        pad_width = MAX_TIME_STEPS - mel_db.shape[1]
        mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=-80.0)
    else:
        # Crop the time axis to MAX_TIME_STEPS
        mel_db = mel_db[:, :MAX_TIME_STEPS]
    
    # Peak Normalization [0.0 to 1.0]
    mel_normalized = (mel_db + 80.0) / 80.0 
    
    return mel_normalized

def process_ravdess_2d(dataset_dir):
    X = []
    y = []
    
    print(f"Extracting 2D Matrices from RAVDESS directory: {dataset_dir}")
    
    for root, dirs, files in os.walk(dataset_dir):
        for filename in files:
            if filename.endswith('.wav'):
                parts = filename.split('.')[0].split('-')
                try:
                    actor_id = int(parts[6])
                    label_value = 1 if actor_id % 2 == 0 else 0
                    
                    file_path = os.path.join(root, filename)
                    features = extract_features_2d(file_path)
                    
                    if features is not None:
                        X.append(features)
                        y.append(label_value)
                except IndexError:
                    continue
                    
    return np.array(X), np.array(y)

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data") 
    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    X, y = process_ravdess_2d(RAW_DATA_DIR)
    
    if len(X) > 0:
        print(f"\nFinal 2D Features Shape (X): {X.shape}") # Should be (N, 128, 86)
        print(f"Final Labels Shape (y): {y.shape}")
        
        # Save with _2d suffix so it doesn't overwrite your MLP arrays
        np.save(os.path.join(PROCESSED_DATA_DIR, 'X_2d.npy'), X)
        np.save(os.path.join(PROCESSED_DATA_DIR, 'y_2d.npy'), y)
        print("\nSuccess! 2D Data saved.")