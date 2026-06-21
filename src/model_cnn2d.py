import torch
import torch.nn as nn

class GenderCNN2D(nn.Module):
    def __init__(self):
        super(GenderCNN2D, self).__init__()
        
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
    
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
       
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        
        
        self.fc1 = nn.Linear(32 * 32 * 21, 128)
        self.dropout = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      
        x = x.unsqueeze(1) 
        
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        
        x = self.flatten(x)
        
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x