import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import SubtaskClassifier

tasks = ['BiPegTransfer-v0', 'BiPegBoard-v0',]
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--task', type=str, choices=tasks, default='BiPegBoard-v0')
args = parser.parse_args()

# Declare device to train on
device = torch.device('cuda:0'
                      if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Load the combined dataset
print(f"Task: {args.task}")
data = np.load(f'{args.task}.npz')
X = data['X']
y = data['y']

# Convert numpy arrays to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
# Remove extra dimension for labels
y_tensor = torch.tensor(y, dtype=torch.long).squeeze().to(device)

# Create a TensorDataset
dataset = TensorDataset(X_tensor, y_tensor)

# Define the sizes for the training and dev sets
train_size = int(0.9 * len(dataset))  # 90% for training
dev_size = len(dataset) - train_size  # 10% for dev

# Split the dataset into training and dev sets
train_dataset, dev_dataset = random_split(dataset, [train_size, dev_size])

# Create DataLoader instances for training and dev sets
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

# Print the sizes of the training and dev sets
print(f"Training set size: {len(train_dataset)}")
print(f"Dev set size: {len(dev_dataset)}")

model = SubtaskClassifier().to(device)
print(model)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Combines softmax and cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Adam optimizer

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    train_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        outputs = model(data)  # Shape: [batch_size, 3]
        loss = criterion(outputs, target)  # Compute the loss

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        train_loss += loss.item()

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for batch_idx, (data, target) in enumerate(dev_loader):
            # Forward pass
            outputs = model(data)  # Shape: [batch_size, 3]
            loss = criterion(outputs, target)  # Compute the loss
            test_loss += loss.item()

    # Print average loss for the epoch
    print(f"Epoch [{epoch + 1:3d}/{num_epochs}] | Train Loss: {train_loss / len(train_loader):.4f} | Test Loss: {test_loss / len(dev_loader):.4f}")

torch.save(model.state_dict(), f"{args.task}_weight.pth")
print("Training complete!")
