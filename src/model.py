import torch
import torch.nn as nn

class GenderCNN(nn.Module):
    def __init__(self):
        super(GenderCNN, self).__init__()
        # Input channel is 1 (grayscale spectrogram)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Adaptive pooling ensures the output is always 10x10, 
        # preventing shape mismatch errors regardless of exact audio length
        self.adaptive_pool = nn.AdaptiveAvgPool2d((10, 10))
        
        # Fully connected layers for binary classification
        self.fc1 = nn.Linear(32 * 10 * 10, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        
        # Flatten the matrix into a 1D vector for the linear layers
        x = x.view(x.size(0), -1) 
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x