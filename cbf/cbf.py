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


class CBF(nn.Module):
    def __init__(self, fc_param):
        super(CBF, self).__init__()

        self.net = self.build_mlp(fc_param)
        self.x_dim = fc_param[0]
        self.u_dim = (fc_param[-1] - fc_param[0]) // fc_param[0]

        # Initializing weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

        self.u = None
        self.device = torch.device(
            'cuda:' + str(0)
            if torch.cuda.is_available() else 'cpu'
        )

    def forward(self, t, x):  # x: [20, 1, 6]
        if self.training:
            net_out = self.net(x)  # [20, 1, 42]
            fx = net_out[:, :, :self.x_dim]
            gx = net_out[:, :, self.x_dim:]
            gx = torch.reshape(gx, (x.shape[0], self.u_dim, self.x_dim))
            return fx + self.u @ gx  # [20, 1, 6]
        else:
            net_out = self.net(x)  # [1, 42]
            fx = net_out[:, :self.x_dim]  # [1, 6]
            gx = net_out[:, self.x_dim:]  # [1, 36]
            gx = torch.reshape(gx, (self.u_dim, self.x_dim))
            return fx + self.u @ gx  # [1, 6]

    def dCBF_sphere(self, robot, u, f, g, constraint_center, radius):
        assert robot.shape == (1, self.x_dim)
        assert u.shape == (1, self.u_dim)
        assert f.shape == (1, self.x_dim)
        assert g.shape == (1, self.x_dim * self.u_dim)
        # assert constraint_center.shape == (1, self.x_dim)
        # assert len(constraint_center) == 3
        assert radius > 0

        # Convert list to torch.tensor
        # constraint_center = torch.tensor(
        #     constraint_center).float().to(self.device)
        # constraint_center = torch.reshape(constraint_center, (1, self.x_dim))

        r = radius

        # Compute barrier function
        b1 = ((robot[:, :3] - constraint_center[:, :3]) ** 2).sum().reshape(1, 1) - r ** 2
        b2 = ((robot[:, 3:] - constraint_center[:, 3:]) ** 2).sum().reshape(1, 1) - r ** 2
        assert b1.shape == b2.shape == (1, 1)

        zeros = torch.zeros((1, 3)).to(self.device)
        db1 = 2 * (robot[:, :3] - constraint_center[:, :3])
        db1 = torch.cat([db1, zeros], dim=1)
        db2 = 2 * (robot[:, 3:] - constraint_center[:, 3:])
        db2 = torch.cat([zeros, db2], dim=1)
        assert db1.shape == db2.shape == (1, self.x_dim)
        
        Lfb1 = db1 @ f.T
        Lfb2 = db2 @ f.T
        assert Lfb1.shape == Lfb2.shape == (1, 1)

        g = torch.reshape(g, (self.u_dim, self.x_dim))
        Lgb1 = db1 @ g.T
        Lgb2 = db2 @ g.T
        assert Lgb1.shape == Lgb2.shape == (1, self.u_dim)
        
        Lfb = torch.cat([Lfb1, Lfb2], dim=0)
        Lgb = torch.cat([Lgb1, Lgb2], dim=0)
        b = torch.cat([b1, b2], dim=0)
        
        gamma = 1
        b_safe = Lfb + gamma * b
        A_safe = -Lgb

        dim = self.u_dim
        G = A_safe.to(self.device)
        h = b_safe.to(self.device)
        P = torch.eye(dim).to(self.device)
        q = -u.T

        # NOTE: different x from above now
        x = cvx_solver(P.double(), q.double(), G.double(), h.double())

        out = []
        for i in range(dim):
            out.append(x[i])
        out = np.array(out)
        out = torch.tensor(out).float().to(self.device)
        out = out.unsqueeze(0)
        return out

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
