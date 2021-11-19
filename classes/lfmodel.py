import torch
import torch.nn as nn
import torch.nn.functional as F

class LFModel(nn.Module):
    
    def __init__(self, n_features):
        super(LFModel, self).__init__()
        self.fc1 = nn.Linear(n_features, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))