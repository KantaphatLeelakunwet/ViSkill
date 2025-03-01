import torch
import torch.nn as nn
import torch.nn.functional as F

class SubtaskClassifier(nn.Module):
    def __init__(self):
        super(SubtaskClassifier, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 84),
            nn.ReLU(),
            nn.Linear(84, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.net(x)
    
    def predict(self, x):
        # Forward pass to get logits
        logits = self.forward(x)
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        # Get the predicted class index
        predicted_class = torch.argmax(probs, dim=1)
        return predicted_class