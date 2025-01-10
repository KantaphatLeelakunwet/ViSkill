import torch
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
