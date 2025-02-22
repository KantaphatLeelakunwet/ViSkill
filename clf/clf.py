import torch
import numpy as np
import torch.nn as nn
from cvxopt import solvers, matrix


def cvx_solver(P, q, G, h):
    mat_P = matrix(P.cpu().numpy())
    mat_q = matrix(q.cpu().numpy())
    mat_G = matrix(G.cpu().numpy())
    mat_h = matrix(h.cpu().numpy())

    solvers.options['show_progress'] = False

    sol = solvers.qp(mat_P, mat_q, mat_G, mat_h)

    return sol['x']


class CLF(nn.Module):
    def __init__(self, fc_param):
        super(CLF, self).__init__()

        self.net = self.build_mlp(fc_param)
        self.x_dim = fc_param[0]
        self.u_dim = (fc_param[-1] - fc_param[0]) // fc_param[0]

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

        self.u = None
        self.device = torch.device(
            'cuda:' + str(0)
            if torch.cuda.is_available() else 'cpu'
        )

    def forward(self, t, x):
        net_out = self.net(x)
        if self.training:
            fx = net_out[:, :, :self.x_dim]
            gx = net_out[:, :, self.x_dim:]
            gx = torch.reshape(gx, (x.shape[0], self.u_dim, self.x_dim))
        else:
            fx = net_out[:, :self.x_dim]
            gx = net_out[:, self.x_dim:]
            gx = torch.reshape(gx, (self.u_dim, self.x_dim))
        return fx + self.u @ gx

    def dCLF(self, robot, desired, u, f, g):
        # assert robot.shape == (1, 3)
        # assert desired.shape == (1, 3)
        # assert u.shape == (1, 1)
        # assert f.shape == (1, 3)
        # assert g.shape == (1, 3)

        # Compute CLF
        V = ((robot - desired) ** 2).sum()

        # Partial derivative
        dV = 2 * torch.tensor([[
            robot[0, i] - desired[0, i] for i in range(robot.shape[1])]]).float().to(self.device)

        dotV_f = dV @ f.T  # [1, 1]

        g = torch.reshape(g, (self.u_dim, self.x_dim))
        dotV_g = dV @ g.T  # [1, 2]

        # dotV + epsilon * V <= 0
        # (dotV_f + dotV_g * u) + epsilon * V <= 0
        # dotV_g * u <= -dotV_f - epsilon * V
        # Gx <= h
        epsilon = 10.
        delta = 1.
        b_safe = -dotV_f - epsilon * V + delta
        A_safe = dotV_g

        assert A_safe.shape == (1, u.shape[1])
        assert b_safe.shape == (1, 1)

        dim = u.shape[1]
        G = A_safe.to(self.device)
        h = b_safe.to(self.device)
        P = torch.eye(dim).to(self.device)
        q = -u.T  # [2, 1]

        # NOTE: different x from above now
        x = cvx_solver(P.double(), q.double(), G.double(), h.double())

        out = []
        for i in range(dim):
            out.append(x[i])
        out = np.array(out)
        out = torch.tensor(out).float().to(self.device)
        out = out.unsqueeze(0)
        return out  # [1, 2]

    def build_mlp(self, filters, no_act_last_layer=True, activation='gelu'):
        if activation == 'gelu':
            activation = nn.GELU()
        elif activation == 'silu':
            activation = nn.SiLU()
        elif activation == 'tanh':
            activation = nn.Tanh()
        else:
            raise NotImplementedError(
                f'Not supported activation function {activation}')
        modules = nn.ModuleList()
        for i in range(len(filters)-1):
            modules.append(nn.Linear(filters[i], filters[i+1]))
            if not (no_act_last_layer and i == len(filters)-2):
                modules.append(activation)

        modules = nn.Sequential(*modules)
        return modules
