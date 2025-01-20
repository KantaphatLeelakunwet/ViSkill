import os
import torch
import argparse
import numpy as np
from cbf import CBF
from torchdiffeq import odeint

subtasks = ['combined']
tasks = ['BiPegTransfer-v0', 'BiPegBoard-v0']

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--task', type=str, choices=tasks, default='BiPegBoard-v0')
parser.add_argument('--subtask', type=str, choices=subtasks, default='combined')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use_dcbf', action='store_true')
parser.add_argument('--train_counter', type=int, default=0)
args = parser.parse_args()

# Setup device
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

grasp_obs = torch.tensor(grasp_obs).float().unsqueeze(2).to(device)
grasp_acs = torch.tensor(grasp_acs).float().unsqueeze(2).to(device)
handover_obs = torch.tensor(handover_obs).float().unsqueeze(2).to(device)
handover_acs = torch.tensor(handover_acs).float().unsqueeze(2).to(device)
release_obs = torch.tensor(release_obs).float().unsqueeze(2).to(device)
release_acs = torch.tensor(release_acs).float().unsqueeze(2).to(device)

# Testing data
grasp_x_test = grasp_obs[-1, :, :, :]
grasp_u_test = grasp_acs[-1, :, :, :]
handover_x_test = handover_obs[-1, :, :, :]
handover_u_test = handover_acs[-1, :, :, :]
release_x_test = release_obs[-1, :, :, :]
release_u_test = release_acs[-1, :, :, :]

# Initial condition for testing
grasp_x_test0 = grasp_obs[-1, 0, :, :]
grasp_u_test0 = grasp_acs[-1, 0, :, :]
handover_x_test0 = handover_obs[-1, 0, :, :]
handover_u_test0 = handover_acs[-1, 0, :, :]
release_x_test0 = release_obs[-1, 0, :, :]
release_u_test0 = release_acs[-1, 0, :, :]

# Set up the dimension of the network
x_dim = grasp_obs.shape[-1]
u_dim = grasp_acs.shape[-1]
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

tt = torch.tensor([0., 0.1]).to(device)
safety = []
total_loss = []

with torch.no_grad():
    # ======================================================
    # GRASP
    # ======================================================
    
    constraint_center = grasp_x_test[grasp_data_size // 2]
    radius = 0.05

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
        
        if args.use_dcbf:
            net_out = func.net(x0)
            fx = net_out[:, :x_dim]
            gx = net_out[:, x_dim:]
            func.u = func.dCBF_sphere(x0, func.u, fx, gx, constraint_center, radius)

        pred = odeint(func, x0, tt)
        pred_x = torch.cat(
            [pred_x, pred[-1, :, :].unsqueeze(0)], dim=0)

        # Update input
        x0 = pred[-1, :, :]
        
        # Compute the distance between robot and obstacle point
        barrier = torch.sum((x0 - constraint_center) ** 2) - radius ** 2
        safety.append(barrier)

    print("=============== Grasp ===============")
    
    if not args.use_dcbf:
        # Compute loss on how much it deviate from actual trajectory
        mae_loss = torch.nn.functional.l1_loss(pred_x, grasp_x_test)
        mse_loss = torch.nn.functional.mse_loss(pred_x, grasp_x_test)
        print("L1 loss:", mae_loss.item())
        print("L2 loss:", mse_loss.item())

    num_violations = torch.sum(torch.tensor(safety) < 0)
    if num_violations == 0:
        print("\033[32mSafe!\033[0m")
    else:
        print(f"\033[31mViolate {num_violations} times!\033[0m")
    
    print()
    
    # ======================================================
    # HANDOVER
    # ======================================================
    
    constraint_center = handover_x_test[handover_data_size // 2]
    radius = 0.05

    x0 = handover_x_test0
    pred_x = x0.unsqueeze(0)

    for i in range(handover_data_size):
        func.u = handover_u_test[i, :, :]
        
        if args.use_dcbf:
            net_out = func.net(x0)
            fx = net_out[:, :x_dim]
            gx = net_out[:, x_dim:]
            func.u = func.dCBF_sphere(x0, func.u, fx, gx, constraint_center, radius)

        pred = odeint(func, x0, tt)
        pred_x = torch.cat(
            [pred_x, pred[-1, :, :].unsqueeze(0)], dim=0)

        x0 = pred[-1, :, :]
        
        # Compute the distance between robot and obstacle point
        barrier = torch.sum((x0 - constraint_center) ** 2) - radius ** 2
        safety.append(barrier)

    print("============== Handover =============")
    
    if not args.use_dcbf:
        # Compute loss on how much it deviate from actual trajectory
        mae_loss = torch.nn.functional.l1_loss(pred_x, handover_x_test)
        mse_loss = torch.nn.functional.mse_loss(pred_x, handover_x_test)
        print("L1 loss:", mae_loss.item())
        print("L2 loss:", mse_loss.item())

    num_violations = torch.sum(torch.tensor(safety) < 0)
    if num_violations == 0:
        print("\033[32mSafe!\033[0m")
    else:
        print(f"\033[31mViolate {num_violations} times!\033[0m")
    
    print()

    # ======================================================
    # RELEASE
    # ======================================================

    constraint_center = release_x_test[release_data_size // 2]
    radius = 0.05
    
    x0 = release_x_test0
    pred_x = x0.unsqueeze(0)

    for i in range(release_data_size):
        func.u = release_u_test[i, :, :]
        
        if args.use_dcbf:
            net_out = func.net(x0)
            fx = net_out[:, :x_dim]
            gx = net_out[:, x_dim:]
            func.u = func.dCBF_sphere(x0, func.u, fx, gx, constraint_center, radius)

        pred = odeint(func, x0, tt)
        pred_x = torch.cat(
            [pred_x, pred[-1, :, :].unsqueeze(0)], dim=0)

        x0 = pred[-1, :, :]

        # Compute the distance between robot and obstacle point
        barrier = torch.sum((x0 - constraint_center) ** 2) - radius ** 2
        safety.append(barrier)

    print("============== Release ==============")
    
    if not args.use_dcbf:
        # Compute loss on how much it deviate from actual trajectory
        mae_loss = torch.nn.functional.l1_loss(pred_x, release_x_test)
        mse_loss = torch.nn.functional.mse_loss(pred_x, release_x_test)
        print("L1 loss:", mae_loss.item())
        print("L2 loss:", mse_loss.item())

    num_violations = torch.sum(torch.tensor(safety) < 0)
    if num_violations == 0:
        print("\033[32mSafe!\033[0m")
    else:
        print(f"\033[31mViolate {num_violations} times!\033[0m")
