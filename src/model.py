import torch
import torch.nn as nn

class GenderMLP(nn.Module):
    def __init__(self):
        super(GenderMLP, self).__init__()
        
        # 128 inputs corresponding to the 128 Mel-frequency bands
        self.fc1 = nn.Linear(128, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        
        # 50% Dropout forces the model to generalize
        self.dropout = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.sigmoid(self.fc3(x))
        return x