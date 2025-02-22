import torch
from torch.utils.data import DataLoader, TensorDataset
from model import SubtaskClassifier
import numpy as np
import argparse

def evaluate_model(model, data_loader, criterion):
    """
    Evaluate the model on the given dataset.

    Args:
        model (nn.Module): The trained model.
        data_loader (DataLoader): DataLoader for the evaluation dataset.
        criterion: Loss function (e.g., nn.CrossEntropyLoss).

    Returns:
        float: Average loss over the evaluation dataset.
        float: Accuracy over the evaluation dataset.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for data, target in data_loader:
            # Forward pass: compute predictions
            outputs = model(data)

            # Compute loss
            loss = criterion(outputs, target)
            total_loss += loss.item()

            # Compute accuracy
            _, predicted_subtask = torch.max(outputs, 1)  # Get the predicted class
            total += target.size(0)
            correct += (predicted_subtask == target).sum().item()

    # Compute average loss and accuracy
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

# Example usage
if __name__ == "__main__":
    # Select task to evaluate
    tasks = ['BiPegTransfer-v0', 'BiPegBoard-v0',]
    parser = argparse.ArgumentParser('ODE demo')
    parser.add_argument('--task', type=str, choices=tasks, default='BiPegBoard-v0')
    args = parser.parse_args()
    
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load the trained model (replace with your actual model)
    model = SubtaskClassifier()
    model.load_state_dict(torch.load(f"{args.task}_weight.pth"))  # Load saved model weights
    model.to(device)

    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Evaluation dataset
    data = np.load(f'{args.task}.npz')
    X = torch.tensor(data['X'], dtype=torch.float32).to(device)
    y = torch.tensor(data['y'], dtype=torch.long).squeeze().to(device)

    # Create a TensorDataset and DataLoader for evaluation
    test_dataset = TensorDataset(X, y)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # Evaluate the model
    avg_loss, accuracy = evaluate_model(model, test_loader, criterion)
    print(f"Evaluation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")