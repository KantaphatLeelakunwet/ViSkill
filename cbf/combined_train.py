import os
import torch
import argparse
import numpy as np
from cbf import CBF
import torch.optim as optim
from torchdiffeq import odeint

torch.autograd.set_detect_anomaly(True)

subtasks = ['combined']
tasks = ['BiPegTransfer-v0', 'BiPegBoard-v0',]

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str,
                    choices=['dopri8', 'adams'], default='dopri8')  # dopri5
parser.add_argument('--activation', type=str,
                    choices=['gelu', 'silu', 'tanh'], default='gelu')
parser.add_argument('--task', type=str, choices=tasks, default='BiPegBoard-v0')
parser.add_argument('--subtask', type=str,
                    choices=subtasks, default='combined')
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--niters', type=int, default=200)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--resume', type=str, default=None,
                    help='Path to pretrained weights')
parser.add_argument('--train_counter', type=int, default=0)
args = parser.parse_args()

# Setting up device used for training
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:' + str(args.gpu)
                      if torch.cuda.is_available() else 'cpu')

# Load dataset
grasp_obs = np.load(f'../data/{args.task}/grasp/obs_pos.npy')
grasp_acs = np.load(f'../data/{args.task}/grasp/acs_pos.npy')
handover_obs = np.load(f'../data/{args.task}/handover/obs_pos.npy')
handover_acs = np.load(f'../data/{args.task}/handover/acs_pos.npy')
release_obs = np.load(f'../data/{args.task}/release/obs_pos.npy')
release_acs = np.load(f'../data/{args.task}/release/acs_pos.npy')

# Scaling data
SCALING = 5.0 if args.task in tasks else 1.0
grasp_acs *= 0.01 * SCALING
handover_acs *= 0.01 * SCALING
release_acs *= 0.01 * SCALING

num_episode, grasp_data_size, _ = grasp_acs.shape
handover_data_size = handover_acs.shape[1]
release_data_size = release_acs.shape[1]

grasp_obs = torch.tensor(grasp_obs).float()
grasp_acs = torch.tensor(grasp_acs).float()
handover_obs = torch.tensor(handover_obs).float()
handover_acs = torch.tensor(handover_acs).float()
release_obs = torch.tensor(release_obs).float()
release_acs = torch.tensor(release_acs).float()

# Training data
grasp_x_train = grasp_obs.unsqueeze(2).to(device)
grasp_u_train = grasp_acs.unsqueeze(2).to(device)
handover_x_train = handover_obs.unsqueeze(2).to(device)
handover_u_train = handover_acs.unsqueeze(2).to(device)
release_x_train = release_obs.unsqueeze(2).to(device)
release_u_train = release_acs.unsqueeze(2).to(device)

# Testing data
grasp_x_test = grasp_x_train[-1, :, :, :]
grasp_u_test = grasp_u_train[-1, :, :, :]
handover_x_test = handover_x_train[-1, :, :, :]
handover_u_test = handover_u_train[-1, :, :, :]
release_x_test = release_x_train[-1, :, :, :]
release_u_test = release_u_train[-1, :, :, :]

# Initial condition for testing
grasp_x_test0 = grasp_x_train[-1, 0, :, :]
grasp_u_test0 = grasp_u_train[-1, 0, :, :]
handover_x_test0 = handover_x_train[-1, 0, :, :]
handover_u_test0 = handover_u_train[-1, 0, :, :]
release_x_test0 = release_x_train[-1, 0, :, :]
release_u_test0 = release_u_train[-1, 0, :, :]


def get_batch(bitr):  # get training data from bitr-th trajectory
    grasp_u = grasp_u_train[bitr, :, :, :]
    grasp_x = grasp_x_train[bitr, :, :, :]
    handover_u = handover_u_train[bitr, :, :, :]
    handover_x = handover_x_train[bitr, :, :, :]
    release_u = release_u_train[bitr, :, :, :]
    release_x = release_x_train[bitr, :, :, :]

    # Get random starting time for a time batch
    g_random = torch.from_numpy(
        np.random.choice(
            np.arange(grasp_data_size - args.batch_time, dtype=np.int64),
            args.batch_size // 3 - 1,
            replace=False
        )
    )
    h_random = torch.from_numpy(
        np.random.choice(
            np.arange(handover_data_size - args.batch_time, dtype=np.int64),
            args.batch_size // 3 - 1,
            replace=False
        )
    )
    r_random = torch.from_numpy(
        np.random.choice(
            np.arange(release_data_size - args.batch_time, dtype=np.int64),
            args.batch_size // 3 - 1,
            replace=False
        )
    )

    g = torch.cat([g_random, torch.tensor([0])], dim=0)  # g.shape = [10]
    h = torch.cat([h_random, torch.tensor([0])], dim=0)
    r = torch.cat([r_random, torch.tensor([0])], dim=0)
    batch_u0 = torch.cat(
        [grasp_u[g], handover_u[h], release_u[r]], dim=0)  # [30, 1, 1]
    batch_x0 = torch.cat(
        [grasp_x[g], handover_x[h], release_x[r]], dim=0)  # [30, 1, 3]

    batch_u = torch.cat([
        torch.stack([grasp_u[g + i] for i in range(args.batch_time)], dim=0),
        torch.stack([handover_u[h + i]
                    for i in range(args.batch_time)], dim=0),
        torch.stack([release_u[r + i] for i in range(args.batch_time)], dim=0)
    ], dim=1)

    batch_x = torch.cat([
        torch.stack([grasp_x[g + i] for i in range(args.batch_time)], dim=0),
        torch.stack([handover_x[h + i]
                    for i in range(args.batch_time)], dim=0),
        torch.stack([release_x[r + i] for i in range(args.batch_time)], dim=0)
    ], dim=1)

    return batch_u0.to(device), batch_x0.to(device), batch_u.to(device), batch_x.to(device)


def makedirs(task: str, subtask: str, train_counter: int):
    if not os.path.exists("saved_model"):
        os.makedirs("saved_model")

    task_dir = os.path.join("saved_model", task)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)

    subtask_dir = os.path.join(task_dir, subtask)
    if not os.path.exists(subtask_dir):
        os.makedirs(subtask_dir)

    saved_dir = os.path.join(subtask_dir, f"{train_counter}")
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    elif not os.listdir(saved_dir):
        pass
    elif not args.resume:
        print(f"'{saved_dir}' already exists and is not empty.")
        print("If you want to retrain, please increment 'train_counter'")
        exit(1)

    return saved_dir


if __name__ == '__main__':
    test_count = 0

    # Create directory to store trained weights
    saved_folder = makedirs(args.task, args.subtask, args.train_counter)
    print(f"Weights will be saved at {saved_folder}")

    # Set up the dimension of the network
    x_dim = grasp_x_train.shape[-1]
    u_dim = grasp_u_train.shape[-1]
    fc_param = [x_dim, 64, x_dim + x_dim * u_dim]

    # Initialize neural ODE
    func = CBF(fc_param).to(device)
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)

    # Load pretrained weights if resuming
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"{args.resume} does not exist.")
            print(f"It should be in {saved_folder}/CLFxx.pth")
            exit(1)

        func.load_state_dict(torch.load(args.resume))
        print(f"Loaded pretrained weights from {args.resume}")
        filename = os.path.basename(args.resume)
        # Check train_counter
        train_counter = int(os.path.basename(os.path.dirname(args.resume)))
        assert train_counter == args.train_counter
        # Set test_count
        test_count = int(''.join(filter(str.isdigit, filename.split('.')[0])))

    test_loss = torch.tensor([0]).to(device)
    tt = torch.tensor([0.0, 0.1]).to(device)

    # Training Loop
    for iter in range(1, args.niters + 1):
        # Loop over each trajectory
        for traj in range(num_episode):
            func.train()
            optimizer.zero_grad()
            batch_u0, batch_x0, batch_u, batch_x = get_batch(traj)

            # Setup input for network
            x0 = batch_x0             # [20, 1, 3]
            pred_x = x0.unsqueeze(0)  # [1, 20, 1, 3]

            # Loop over each time step in a time batch
            for i in range(args.batch_time-1):

                # Store this to use u in forward() later
                batch_u_i = batch_u[i, :, :, :]
                func.u = batch_u_i

                # pred: [ x0, x0 at next timestep ]
                # returns next observation in the last index
                # (depend on what tt is)
                pred = odeint(func, x0, tt).to(device)  # [2, 20, 1, 3]

                # Concat predicted next state into pred_x
                pred_x = torch.cat(
                    [pred_x, pred[-1, :, :, :].unsqueeze(0)], dim=0)

                # Update input to the network
                x0 = pred[-1, :, :, :]

            # Compute loss [10, 20, 1, 3]
            loss = torch.mean(torch.abs(pred_x - batch_x))

            # Display loss
            print('iteration: ', iter,
                  '| traj: ', traj,
                  '| train loss: ', loss.item())

            loss.backward()
            optimizer.step()

        if iter % args.test_freq == 0:
            with torch.no_grad():
                func.eval()

                # ======================================================
                # GRASP
                # ======================================================

                # Set input for network
                x0 = grasp_x_test0
                pred_x = x0.unsqueeze(0)

                # Test over the whole test trajectory
                # Predict every timestep
                # Compute cost with real data
                for i in range(grasp_data_size):
                    # Setup u for forward()
                    # u_test[i,:,:] is the actions of test trajectory that at time i
                    func.u = grasp_u_test[i, :, :]

                    pred = odeint(func, x0, tt)
                    pred_x = torch.cat(
                        [pred_x, pred[-1, :, :].unsqueeze(0)], dim=0)

                    # Update input
                    x0 = pred[-1, :, :]

                # [data_size+1, 1, 6]
                test_loss_grasp = torch.mean(torch.abs(pred_x - grasp_x_test))

                # Display info
                print('Iter {:04d} | Test Loss Grasp {:.6f}'.format(
                    iter, test_loss_grasp.item()))

                # ======================================================
                # HANDOVER
                # ======================================================

                x0 = handover_x_test0
                pred_x = x0.unsqueeze(0)

                for i in range(handover_data_size):
                    func.u = handover_u_test[i, :, :]  # u_test_i

                    pred = odeint(func, x0, tt)
                    pred_x = torch.cat(
                        [pred_x, pred[-1, :, :].unsqueeze(0)], dim=0)

                    x0 = pred[-1, :, :]

                test_loss_handover = torch.mean(
                    torch.abs(pred_x - handover_x_test))

                # Display info
                print('Iter {:04d} | Test Loss Handover {:.6f}'.format(
                    iter, test_loss_handover.item()))

                # ======================================================
                # RELEASE
                # ======================================================

                x0 = release_x_test0
                pred_x = x0.unsqueeze(0)

                for i in range(release_data_size):
                    func.u = release_u_test[i, :, :]  # u_test_i

                    pred = odeint(func, x0, tt)
                    pred_x = torch.cat(
                        [pred_x, pred[-1, :, :].unsqueeze(0)], dim=0)

                    x0 = pred[-1, :, :]

                test_loss_release = torch.mean(
                    torch.abs(pred_x - release_x_test))

                # Display info
                print('Iter {:04d} | Test Loss Release {:.6f}'.format(
                    iter, test_loss_release.item()))

                test_count += 1

            torch.save(func.state_dict(),
                       f"{saved_folder}/CBF{format(test_count, '02d')}.pth")
