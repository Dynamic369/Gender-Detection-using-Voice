import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import copy
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.model_cnn2d import GenderCNN2D

def train_cnn2d():
    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    
   
    X = np.load(os.path.join(PROCESSED_DATA_DIR, "X_2d.npy"))
    y = np.load(os.path.join(PROCESSED_DATA_DIR, "y_2d.npy"))
    
    tensor_X = torch.Tensor(X)
    tensor_y = torch.Tensor(y).view(-1, 1)
    dataset = TensorDataset(tensor_X, tensor_y)
    
    total_size = len(dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = GenderCNN2D()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    
    epochs = 40
    patience = 5
    patience_counter = 0
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    
    print("Starting 2D CNN training on RAVDESS...")
    
    for epoch in range(epochs):
        model.train() 
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval() 
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_model_weights)
    model.eval()
    
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            predicted = (outputs >= 0.5).float()
            test_correct += (predicted == batch_y).sum().item()
            test_total += batch_y.size(0)
            
    print(f"\nFinal Test Accuracy (2D CNN): {(test_correct / test_total) * 100:.2f}%")
    
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    SAVE_PATH = os.path.join(MODELS_DIR, "gender_model_cnn2d.pth") 
    
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"2D CNN Model saved to: {SAVE_PATH}")

if __name__ == "__main__":
    train_cnn2d()