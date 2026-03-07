import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
from model import GenderCNN
import os
import copy

def train_model():
    # 1. Load and Normalize Data
    X = np.load(r"C:\Users\Pradum Gupta\OneDrive\Desktop\Coding\Gender Detection\data\processed\X.npy")
    y = np.load(r"C:\Users\Pradum Gupta\OneDrive\Desktop\Coding\Gender Detection\data\processed\y.npy")
    
    X = np.transpose(X, (0, 3, 1, 2)) 
    tensor_X = torch.Tensor(X)
    tensor_X = (tensor_X - tensor_X.mean()) / tensor_X.std()
    tensor_y = torch.Tensor(y).view(-1, 1)
    
    dataset = TensorDataset(tensor_X, tensor_y)
    
    # 2. Train / Validation / Test Split (70% / 15% / 15%)
    total_size = len(dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 3. Model, Loss, and Optimizer Initialization
    model = GenderCNN()
    criterion = nn.BCELoss()
    # Added slight L2 regularization (weight_decay) to further penalize overfitting
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) 
    
    # 4. Early Stopping Configuration
    epochs = 30 # We can set this higher now because early stopping will catch it
    patience = 4 # Stop if validation loss doesn't improve for 4 consecutive epochs
    patience_counter = 0
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    
    print(f"Dataset Split: {train_size} Train | {val_size} Val | {test_size} Test")
    print("Starting training with Early Stopping...")
    
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train() # Activates Dropout layers
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION PHASE ---
        model.eval() # Deactivates Dropout layers for accurate evaluation
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # --- EARLY STOPPING LOGIC ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best weights observed so far
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered! Validation loss hasn't improved in {patience} epochs.")
                break

    # 5. Final Testing Phase
    print("\nLoading best model weights for final Test evaluation...")
    model.load_state_dict(best_model_weights)
    model.eval()
    
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            predicted = (outputs >= 0.5).float()
            test_correct += (predicted == batch_y).sum().item()
            test_total += batch_y.size(0)
            
    test_accuracy = (test_correct / test_total) * 100
    print(f"Final Test Accuracy (Unseen Data): {test_accuracy:.2f}%")
    
    # 6. Save the optimized model
    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), "../models/gender_model.pth")
    print("Optimized model saved to models/gender_model.pth")

if __name__ == "__main__":
    train_model()