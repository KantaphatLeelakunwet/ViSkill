import numpy as np
import argparse
import socket
import os

parser = argparse.ArgumentParser('Data Processing')
parser.add_argument(
    '--task',
    type=str,
    choices=['BiPegBoard-v0', 'BiPegTransfer-v0'],
    default='BiPegBoard-v0'
)
args = parser.parse_args()


def makedir(task: str, subtask: str):
    # Create a task directory if it doesn't exist
    if not os.path.exists(task):
        os.makedirs(task)

    # Define the subtask directory path
    subtask_dir = os.path.join(task, subtask)

    # Check if the subtask directory exists
    if not os.path.exists(subtask_dir):
        # Create the subtask directory if it doesn't exist
        os.makedirs(subtask_dir)
    else:
        # Check if the subtask directory is empty
        if not os.listdir(subtask_dir):
            # Directory exists and is empty, do nothing
            pass
        else:
            # Directory exists and is not empty, print error message and exit
            print(f"'{subtask_dir}' already exists and is not empty.")
            exit(1)


subtasks = ['grasp', 'handover', 'release']
demo_path = f'/bd_{socket.gethostname()}/users/kleelakunwet/demo/'

# Load demonstration for each subtask
for subtask in subtasks:
    filename = f'data_{args.task}_random_200_primitive_new{subtask}.npz'
    data_path = demo_path + filename

    makedir(args.task, subtask)
    data = np.load(data_path, allow_pickle=True)

    # Action
    episodes, total_time, _ = data['actions'].shape
    acs_pos = data['actions'][:, :, [0, 1, 2, 5, 6, 7]]
    assert acs_pos.shape == (episodes, total_time, 6)
    acs_orn = data['actions'][:, :, [3, 8]]
    assert acs_orn.shape == (episodes, total_time, 2)
    np.save(os.path.join(args.task, subtask, 'acs_pos.npy'), acs_pos)
    np.save(os.path.join(args.task, subtask, 'acs_orn.npy'), acs_orn)

    # Observation
    episodes, total_time = data['observations'].shape
    obs_pos = np.zeros((episodes, total_time, 6))
    obs_orn = np.zeros((episodes, total_time, 6))
    for ep in range(episodes):
        for t in range(total_time):
            obs_pos[ep, t] = data['observations'][ep, t]['observation'][[0, 1, 2, 7, 8, 9]]
            obs_orn[ep, t] = data['observations'][ep, t]['observation'][[3, 4, 5, 10, 11, 12]]
    np.save(os.path.join(args.task, subtask, 'obs_pos.npy'), obs_pos)
    np.save(os.path.join(args.task, subtask, 'obs_orn.npy'), obs_orn)
