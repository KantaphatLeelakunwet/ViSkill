import os
import argparse
import numpy as np
import torch
from torchdiffeq import odeint
from cbf import CBF

subtasks = ['grasp', 'handover', 'release']
tasks = ['BiPegTransfer-v0', 'BiPegBoard-v0']

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str,
                    choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--task', type=str, choices=tasks, default='BiPegBoard-v0')
parser.add_argument('--subtask', type=str, choices=subtasks, default='grasp')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use_dcbf', action='store_true')
parser.add_argument('--train_counter', type=int, default=0)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device(
    'cuda:' + str(args.gpu)
    if torch.cuda.is_available() else 'cpu'
)

# Load dataset
obs = np.load(f'../data/{args.task}/{args.subtask}/obs_pos.npy')
acs = np.load(f'../data/{args.task}/{args.subtask}/acs_pos.npy')
SCALING = 5.0 if args.task in tasks else 1.0
acs = acs * 0.01 * SCALING

_, data_size, _ = acs.shape

obs = torch.tensor(obs).float()
acs = torch.tensor(acs).float()

# Training data
x_train = obs.unsqueeze(2).to(device)  # [100, 51, 1, 3]
u_train = acs.unsqueeze(2).to(device)  # [100, 50, 1, 3]

# Testing data
x_test = x_train[-1, :, :, :]  # [51, 1, 3]
u_test = u_train[-1, :, :, :]  # [50, 1, 3]

# Initial condition for testing
x_test0 = x_train[-1, 0, :, :]  # [1, 3]
u_test0 = u_train[-1, 0, :, :]  # [1, 3]

# Set up the dimension of the network
x_dim = x_train.shape[-1]
u_dim = u_train.shape[-1]
fc_param = [x_dim, 64, x_dim + x_dim * u_dim]

# Initialize neural ODE
func = CBF(fc_param).to(device)
latest_model = max(os.listdir(
    f"saved_model/{args.task}/{args.subtask}/{args.train_counter}/"), key=lambda f: int(f[3:5]))
func.load_state_dict(torch.load(
    f"saved_model/{args.task}/{args.subtask}/{args.train_counter}/{latest_model}"))
func.eval()

# Display selected choices
print(f"Task   : {args.task}")
print(f"Subtask: {args.subtask}")
print(f"Counter: {args.train_counter}")

# Set up initial state
tt = torch.tensor([0., 0.1]).to(device)
x0 = x_test0.clone().detach().to(device)
pred_x = x0.unsqueeze(0)

safety = []
total_loss = []

with torch.no_grad():
    constraint_center = x_test[data_size // 2]
    radius = 0.05

    for i in range(data_size):
        # Setup u for forward()
        # u_test[i,:,:] is the actions of test trajectory that at time i
        u_test_i = u_test[i, :, :]  # [1, 6]

        if args.use_dcbf:
            net_out = func.net(x0)  # [1, 42]
            fx = net_out[:, :x_dim]  # [1, 6]
            gx = net_out[:, x_dim:]  # [1, 36]
            func.u = func.dCBF_sphere(
                x0, u_test_i, fx, gx, constraint_center, radius)
        else:
            func.u = u_test_i

        pred = odeint(func, x0, tt)
        pred_x = torch.cat(
            [pred_x, pred[-1, :, :].unsqueeze(0)], dim=0)
        x0 = pred[-1, :, :]

        # Compute loss on how much it deviate from actual trajectory
        loss = torch.sum((x0 - x_test[i + 1]) ** 2)
        total_loss.append(loss.cpu().item())
        if not args.use_dcbf:
            print(f'timestep: {i:02d} | loss: {loss.item()}')
        
        # Compute the distance between robot and obstacle point
        barrier = torch.sum((x0 - constraint_center) ** 2) - radius ** 2
        safety.append(barrier)


if not args.use_dcbf:
    # Print total loss
    print('total loss:   ', np.sum(total_loss))
    print('Average loss: ', np.mean(total_loss))
    
print("====== Safety ======")
num_violations = torch.sum(torch.tensor(safety) < 0)
if num_violations == 0:
    print("\033[32mSafe!\033[0m")
else:
    print(f"\033[31mViolate {num_violations} times!\033[0m")
