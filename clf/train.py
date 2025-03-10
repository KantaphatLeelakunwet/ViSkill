import os
import torch
import argparse
import numpy as np
from clf import CLF
import torch.optim as optim
from torchdiffeq import odeint

torch.autograd.set_detect_anomaly(True)

subtasks = ['grasp', 'handover', 'release']
tasks = ['BiPegTransfer-v0', 'BiPegBoard-v0',]

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str,
                    choices=['dopri8', 'adams'], default='dopri8')
parser.add_argument('--activation', type=str,
                    choices=['gelu', 'silu', 'tanh'], default='gelu')
parser.add_argument('--task', type=str, choices=tasks, default='BiPegBoard-v0')
parser.add_argument('--subtask', type=str, choices=subtasks, default='grasp')
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=200)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--resume', type=str, default=None,
                    help='Path to pretrained weights')
parser.add_argument('--train_counter', type=int, default=0)
parser.add_argument('--psm', type=int, choices=[0, 1, 2], default=1)
args = parser.parse_args()

# Setting up device used for training
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:' + str(args.gpu)
                      if torch.cuda.is_available() else 'cpu')

# Load dataset
obs = np.load(f'../data/{args.task}/{args.subtask}/obs_orn.npy')
acs = np.load(f'../data/{args.task}/{args.subtask}/acs_orn.npy')
num_episode, data_size, _ = acs.shape

if args.task == 'BiPegBoard-v0':
    acs[:, :, 0] *= np.deg2rad(15)
    acs[:, :, 1] *= np.deg2rad(30)
else:
    acs *= np.deg2rad(30)

if args.psm == 1:
    obs = obs[:, :, :3]
    acs = acs[:, :, 0].reshape(num_episode, data_size, 1)
elif args.psm == 2:
    obs = obs[:, :, 3:]
    acs = acs[:, :, 1].reshape(num_episode, data_size, 1)

obs = torch.tensor(obs).float()
acs = torch.tensor(acs).float()

# Training data
x_train = obs.unsqueeze(2).to(device)
u_train = acs.unsqueeze(2).to(device)

# Testing data
x_test = x_train[-1, :, :, :]
u_test = u_train[-1, :, :, :]

# Initial condition for testing
x_test0 = x_train[-1, 0, :, :]
u_test0 = u_train[-1, 0, :, :]

t = torch.arange(0., 0.1 * data_size, 0.1).to(device)


def get_batch(bitr):  # get training data from bitr-th trajectory
    # u = [ time, 1, actions ]
    u = u_train[bitr, :, :, :]
    x = x_train[bitr, :, :, :]

    # data_size  = length of traj = 50
    # batch_time = 10
    # batch_size = 20
    # np.arange(50-10) = [0, 1, 2, 3, ..., 39]
    # bb = array of size 19 with random numbers from [0, 39]
    # bb.shape = [19]
    bb = torch.from_numpy(
        np.random.choice(
            np.arange(data_size - args.batch_time, dtype=np.int64),
            args.batch_size-1,
            replace=False
        )
    )
    aa = torch.tensor([0])          # augment 0, aa.shape = [1]

    # Concat 0 to the end of bb
    s = torch.cat([bb, aa], dim=0)  # s.shape = [20]
    batch_u0 = u[s]  # (M, D) # [20, 1, 1]
    batch_x0 = x[s]  # [20, 1, 3]
    batch_t = t[:args.batch_time]  # (T)

    # u[s].shape = [20, 1, 1]
    batch_u = torch.stack(
        [u[s + i] for i in range(args.batch_time)], dim=0
    )  # (T, M, D) # [10, 20, 1, 1]
    # batch_u[time] returns actions of trajectory bitr starting from time "s" to time "s + 10"
    # u[s] returns the actions at time s, where s contains multiple time
    # u[s + i] returns the actions at time s + i
    # batch_u[0][start] returns actions at time start
    # batch_u[i][start] returns actions at time start + i

    batch_x = torch.stack(
        [x[s + i] for i in range(args.batch_time)], dim=0
    )  # [10, 20, 1, 3]

    return batch_u0.to(device), batch_x0.to(device), batch_t.to(device), batch_u.to(device), batch_x.to(device)


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

    saved_folder = makedirs(args.task, args.subtask, args.train_counter)
    print(f"Weights will be saved at {saved_folder}")

    # Set up the dimension of the network
    x_dim = x_train.shape[-1]
    u_dim = u_train.shape[-1]
    fc_param = [x_dim, 64, x_dim + x_dim * u_dim]

    # Initialize neural ODE
    func = CLF(fc_param).to(device)
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
        # Check if train_counter are the same as args
        train_counter = int(os.path.basename(os.path.dirname(args.resume)))
        assert train_counter == args.train_counter
        # Set test_count
        test_count = int(''.join(filter(str.isdigit, filename.split('.')[0])))

    test_loss = torch.tensor([0]).to(device)
    tt = torch.tensor([0.0, 0.1]).to(device)

    # Training Loop
    for iter in range(1, args.niters + 1):
        # Loop over each trajectory
        for traj in range(x_train.shape[0]):
            func.train()
            optimizer.zero_grad()
            batch_u0, batch_x0, batch_t, batch_u, batch_x = get_batch(traj)

            # Setup input for network
            x0 = batch_x0
            pred_x = x0.unsqueeze(0)

            # Loop over each time step in a time batch
            for i in range(args.batch_time-1):
                # Store this to use u in forward() later
                batch_u_i = batch_u[i, :, :, :]
                func.u = batch_u_i

                # pred: [ x0, x0 at next timestep ]
                # returns next observation in the last index
                # (depend on what tt is)
                pred = odeint(func, x0, tt).to(device)  # [2, 20, 1, 6]

                # Concat predicted next state into pred_x
                pred_x = torch.cat(
                    [pred_x, pred[-1, :, :, :].unsqueeze(0)], dim=0)

                # Update input to the network
                x0 = pred[-1, :, :, :]

            # NOTE: Orientation lies between [-pi, pi]
            #       Let's say changing from 0.99 pi to -0.99 pi
            #       is actually 0.02 pi of difference. However, if you
            #       directly compute the difference, it turns out to be 1.98 pi.
            #       Hence, we should do some modification to target value (-0.99 in this case)
            #       such that we can get the difference of 0.02 pi.
            # NOTE: Once a value is changed, every value after that must also be changed.

            while True:
                diff = batch_x[:-1, :, :, :] - batch_x[1:, :, :, :]

                mask_greater = diff > np.pi
                if mask_greater.any():
                    batch_x[1:, :, :, :] = torch.where(mask_greater,
                                                       batch_x[1:, :, :,
                                                               :] + 2 * np.pi,
                                                       batch_x[1:, :, :, :])

                mask_lesser = diff < -np.pi
                if mask_lesser.any():
                    batch_x[1:, :, :, :] = torch.where(mask_lesser,
                                                       batch_x[1:, :, :,
                                                               :] - 2 * np.pi,
                                                       batch_x[1:, :, :, :])

                if mask_greater.any() == False and mask_lesser.any() == False:
                    break

            # Compute loss [10, 20, 1, 6]
            loss = torch.mean(torch.abs(pred_x - batch_x))

            # Display loss
            print('iteration: ', iter,
                  '| traj: ', traj,
                  '| train loss: ', loss.item())

            loss.backward()
            optimizer.step()

        if iter % args.test_freq == 0:
            with torch.no_grad():
                # Set input for network
                x0 = x_test0
                pred_x = x0.unsqueeze(0)
                func.eval()

                # Test over the whole test trajectory
                # Predict every timestep
                # Compute cost with real data
                for i in range(data_size):
                    # Setup u for forward()
                    # u_test[i,:,:] is the actions of test trajectory that at time i
                    u_test_i = u_test[i, :, :]
                    func.u = u_test_i

                    pred = odeint(func, x0, tt)
                    pred_x = torch.cat(
                        [pred_x, pred[-1, :, :].unsqueeze(0)], dim=0)

                    # Update input
                    x0 = pred[-1, :, :]

                while True:
                    diff = x_test[:-1, :, :] - x_test[1:, :, :]

                    mask_greater = diff > np.pi
                    if mask_greater.any():
                        x_test[1:, :, :] = torch.where(
                            mask_greater,
                            x_test[1:, :, :] + 2 * np.pi,
                            x_test[1:, :, :])

                    mask_lesser = diff < -np.pi
                    if mask_lesser.any():
                        x_test[1:, :, :] = torch.where(
                            mask_lesser,
                            x_test[1:, :, :] - 2 * np.pi,
                            x_test[1:, :, :])

                    if mask_greater.any() == False and mask_lesser.any() == False:
                        break

                test_loss = torch.mean(torch.abs(pred_x - x_test))

                # Display info
                print('Iter {:04d} | Test Loss {:.6f}'.format(
                    iter, test_loss.item()))

                test_count += 1

            torch.save(func.state_dict(),
                       f"{saved_folder}/CLF{format(test_count, '02d')}.pth")
