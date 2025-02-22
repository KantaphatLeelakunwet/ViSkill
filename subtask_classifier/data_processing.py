import numpy as np
import argparse

tasks = ['BiPegTransfer-v0', 'BiPegBoard-v0',]
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--task', type=str, choices=tasks, default='BiPegBoard-v0')
args = parser.parse_args()

# Load the data from the three files
grasp_data = np.load(f'../data/{args.task}/grasp/obs_pos.npy')  # Shape: [200, 35, 6]
handover_data = np.load(f'../data/{args.task}/handover/obs_pos.npy')  # Shape: [200, 35, 6]
release_data = np.load(f'../data/{args.task}/release/obs_pos.npy')  # Shape: [200, 35, 6]

# Reshape the data to [7000, 6] for each subtask
grasp_data = grasp_data.reshape(-1, 6)  # Shape: [7000, 6]
handover_data = handover_data.reshape(-1, 6)  # Shape: [7000, 6]
release_data = release_data.reshape(-1, 6)  # Shape: [7000, 6]

# Assign labels to each subtask
grasp_labels = np.zeros((grasp_data.shape[0], 1), dtype=int)  # Label 0 for grasp
handover_labels = np.ones((handover_data.shape[0], 1), dtype=int)  # Label 1 for handover
release_labels = np.full((release_data.shape[0], 1), 2, dtype=int)  # Label 2 for release

# Combine the data and labels
X = np.vstack((grasp_data, handover_data, release_data))  # Shape: [21000, 6]
y = np.vstack((grasp_labels, handover_labels, release_labels))  # Shape: [21000, 1]

# Save the combined dataset to a single file
np.savez(f'{args.task}.npz', X=X, y=y)
