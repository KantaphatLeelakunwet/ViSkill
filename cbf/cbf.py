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

    def constraint_valid(self, constraint_type, robot, constraint_center=None, radius=None, ori_vector=None):

        # Sphere constraint
        if constraint_type == 'sphere':
            assert constraint_center is not None
            assert radius is not None
            assert robot.shape == (6,)
            
            constraint_center = np.array(constraint_center)
            b1 = np.sum((robot[:3] - constraint_center) ** 2) - radius ** 2
            b2 = np.sum((robot[3:] - constraint_center) ** 2) - radius ** 2
            violate = (b1 <= 0) or (b2 <= 0)
        
        # Cylinder constraint
        elif constraint_type == 'cylinder':
            assert constraint_center is not None
            assert ori_vector is not None
            assert radius is not None
            assert robot.shape == (3,)
            
            robot = np.array(robot)
            ori_vector = np.array(ori_vector)
            constraint_center = np.array(constraint_center)
            
            proj_vec = np.dot(ori_vector, robot - constraint_center) * ori_vector
            norm_vec = robot - (constraint_center + proj_vec)
            violate = (np.sum(norm_vec ** 2) - radius ** 2 >= 0)
        else:
            violate = False

        return violate

    def dCBF_sphere(self, robot, u, f, g, constraint_center, radius):
        assert robot.shape == (1, self.x_dim)
        assert u.shape == (1, self.u_dim)
        assert f.shape == (1, self.x_dim)
        assert g.shape == (1, self.x_dim * self.u_dim)
        assert len(constraint_center) == 3
        assert radius > 0

        # Convert list to torch.tensor
        constraint_center = torch.tensor(constraint_center).float().reshape(1, 3).to(self.device)

        r = radius

        # Compute barrier function
        b1 = ((robot[:, :3] - constraint_center) ** 2).sum().reshape(1, 1) - r ** 2
        b2 = ((robot[:, 3:] - constraint_center) ** 2).sum().reshape(1, 1) - r ** 2
        assert b1.shape == b2.shape == (1, 1)

        zeros = torch.zeros((1, 3)).to(self.device)
        db1 = 2 * (robot[:, :3] - constraint_center)
        db1 = torch.cat([db1, zeros], dim=1)
        db2 = 2 * (robot[:, 3:] - constraint_center)
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

    def dCBF_cylinder(self, robot, u, f, g, ori_vec, center, radius, psm1_area, psm2_area):
        assert robot.shape == (1, self.x_dim)
        assert u.shape == (1, self.u_dim)
        assert f.shape == (1, self.x_dim)
        assert g.shape == (1, self.x_dim * self.u_dim)
        assert len(center) == 3 # List
        assert len(ori_vec) == 3 # List
        assert radius > 0
        
        # "ori_vec" is a unit vector.
        
        # PSM1
        x1, y1, z1 = robot[0, 0], robot[0, 1], robot[0, 2]
        # PSM2
        x2, y2, z2 = robot[0, 3], robot[0, 4], robot[0, 5]

        # Obstacle point position
        x0, y0, z0 = center
        # Cylinder orientation vector
        orix, oriy, oriz = ori_vec

        r = radius

        '''
        # proj_factor = orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)
        # norm_vec_x = x - x0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * orix
        # norm_vec_x = x - (x0 + proj_factor * orix)
        # norm_vec_y = y - y0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * oriy
        # norm_vec_z = z - z0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * oriz

        # Compute barrier function
        # derivation
        # b = (x - x0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * orix) ** 2 +\
        #     (y - y0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * oriy) ** 2 +\
        #     (z - z0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * oriz) ** 2 - r ** 2
        # b = ((1 - orix ** 2)(x - x0) - orix * (oriy * (y - y0) + oriz * (z - z0))) ** 2 +\
        #     ((1 - oriy ** 2)(y - y0) - oriy * (orix * (x - x0) + oriz * (z - z0))) ** 2 +\
        #     ((1 - oriz ** 2)(z - z0) - oriz * (orix * (x - x0) + oriy * (y - y0))) ** 2 - r ** 2
        # b = ((1 - orix ** 2)(x - x0) - orix * oriy * (y - y0) - orix * oriz * (z - z0)) ** 2 +\
        #     (- oriy * orix * (x - x0) + (1 - oriy ** 2)(y - y0) - oriy * oriz * (z - z0)) ** 2 +\
        #     (- oriz * orix * (x - x0) - oriz * oriy * (y - y0) + (1 - oriz ** 2)(z - z0)) ** 2 - r ** 2
        '''
        
        # Coefficient for norm_vec_x
        c1x = (1 - orix ** 2)
        c1y = - orix * oriy
        c1z = - orix * oriz
        # Coefficient for norm_vec_y
        c2x = - oriy * orix
        c2y = (1 - oriy ** 2)
        c2z = - oriy * oriz
        # Coefficient for norm_vec_z
        c3x = - oriz * orix
        c3y = - oriz * oriy
        c3z = (1 - oriz ** 2)
        
        b1 = (
            (c1x * (x1 - x0) + c1y * (y1 - y0) + c1z * (z1 - z0)) ** 2 +
            (c2x * (x1 - x0) + c2y * (y1 - y0) + c2z * (z1 - z0)) ** 2 +
            (c3x * (x1 - x0) + c3y * (y1 - y0) + c3z * (z1 - z0)) ** 2 - r ** 2
        ).clone().detach().reshape(1, 1).to(self.device)
        
        b2 = (
            (c1x * (x2 - x0) + c1y * (y2 - y0) + c1z * (z2 - z0)) ** 2 + \
            (c2x * (x2 - x0) + c2y * (y2 - y0) + c2z * (z2 - z0)) ** 2 + \
            (c3x * (x2 - x0) + c3y * (y2 - y0) + c3z * (z2 - z0)) ** 2 - r ** 2
        ).clone().detach().reshape(1, 1).to(self.device)
            
        db1 = torch.tensor([
            2 * c1x * (c1x * (x1 - x0) + c1y * (y1 - y0) + c1z * (z1 - z0)) +
            2 * c2x * (c2x * (x1 - x0) + c2y * (y1 - y0) + c2z * (z1 - z0)) +
            2 * c3x * (c3x * (x1 - x0) + c3y * (y1 - y0) + c3z * (z1 - z0)) ,
            
            2 * c1y * (c1x * (x1 - x0) + c1y * (y1 - y0) + c1z * (z1 - z0)) +
            2 * c2y * (c2x * (x1 - x0) + c2y * (y1 - y0) + c2z * (z1 - z0)) +
            2 * c3y * (c3x * (x1 - x0) + c3y * (y1 - y0) + c3z * (z1 - z0)) ,
            
            2 * c1z * (c1x * (x1 - x0) + c1y * (y1 - y0) + c1z * (z1 - z0)) +
            2 * c2z * (c2x * (x1 - x0) + c2y * (y1 - y0) + c2z * (z1 - z0)) +
            2 * c3z * (c3x * (x1 - x0) + c3y * (y1 - y0) + c3z * (z1 - z0)) ,
            
            0., 0., 0.
        ]).unsqueeze(0).to(self.device)
        
        db2 = torch.tensor([ 0., 0., 0.,
            2 * c1x * (c1x * (x1 - x0) + c1y * (y1 - y0) + c1z * (z1 - z0)) +
            2 * c2x * (c2x * (x1 - x0) + c2y * (y1 - y0) + c2z * (z1 - z0)) +
            2 * c3x * (c3x * (x1 - x0) + c3y * (y1 - y0) + c3z * (z1 - z0)) ,
            
            2 * c1y * (c1x * (x1 - x0) + c1y * (y1 - y0) + c1z * (z1 - z0)) +
            2 * c2y * (c2x * (x1 - x0) + c2y * (y1 - y0) + c2z * (z1 - z0)) +
            2 * c3y * (c3x * (x1 - x0) + c3y * (y1 - y0) + c3z * (z1 - z0)) ,
            
            2 * c1z * (c1x * (x1 - x0) + c1y * (y1 - y0) + c1z * (z1 - z0)) +
            2 * c2z * (c2x * (x1 - x0) + c2y * (y1 - y0) + c2z * (z1 - z0)) +
            2 * c3z * (c3x * (x1 - x0) + c3y * (y1 - y0) + c3z * (z1 - z0))
        ]).unsqueeze(0).to(self.device)
        
        Lfb1 = db1 @ f.T
        Lfb2 = db2 @ f.T
        
        g = torch.reshape(g, (self.u_dim, self.x_dim))
        Lgb1 = db1 @ g.T
        Lgb2 = db2 @ g.T

        if psm1_area == 0:
            Lfb1 = torch.empty(0).to(self.device)
            Lgb1 = torch.empty(0).to(self.device)
            b1 = torch.empty(0).to(self.device)
        elif psm1_area == 2:
            Lfb1 = -Lfb1
            Lgb1 = -Lgb1
            b1 = -b1

        if psm2_area == 0:
            Lfb2 = torch.empty(0).to(self.device)
            Lgb2 = torch.empty(0).to(self.device)
            b2 = torch.empty(0).to(self.device)
        elif psm1_area == 2:
            Lfb2 = -Lfb2
            Lgb2 = -Lgb2
            b2 = -b2

        Lfb = torch.cat([Lfb1, Lfb2], dim=0)
        Lgb = torch.cat([Lgb1, Lgb2], dim=0)
        b = torch.cat([b1, b2], dim=0)
        
        gamma = 1
        A_safe = -Lgb
        b_safe = Lfb + gamma * b

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
