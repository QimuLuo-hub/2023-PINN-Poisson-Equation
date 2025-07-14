import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import time
import pandas as pd

'''  
Ensure that the result is the same every time,  each initialization is fixed. 
'''


def setup_seed(seed):
    torch.manual_seed(seed)  # Results of random operations on the CPU will be deterministic and reproducible.


setup_seed(888888)  # Choose a value large enough to ensure randomness


class PINN(nn.Module):
    def __init__(self, input_size, neurons, hidden_layers, output_size):
        super().__init__()
        # input layer
        self.net = nn.Sequential(nn.Linear(input_size, neurons), nn.Tanh())

        # hidden layer
        for _ in range(hidden_layers):
            self.net.add_module('hidden_layer',
                                nn.Linear(neurons, neurons))  # add layers below the input layer in a specific order
            self.net.add_module('hidden_activation', nn.Tanh())

        # output layer
        self.net.add_module('output_layer', nn.Linear(neurons, output_size))

    def forward(self, x):
        return self.net(x)


# neurons = 64
# hidden_layers = 3
# PINNnet = PINN(2, neurons, hidden_layers, 1)

class PDE_solver:
    def __init__(self, N_b, N_f, h):
        self.N_b = N_b
        self.N_f = N_f
        self.h = h

        self.u = PINNnet
        self.loss = torch.nn.MSELoss()

        xm, ym, xx, yy, xy = self.gridpoint()
        self.u_real = self.exactsol(xx, yy)

    '''Function to modify'''

    def exactsol(self, x, y):
        raise NotImplementedError

    def rhs(self, x, y):
        raise NotImplementedError

    '''Data preparation: Boundary, Interior, Reference points '''

    # Boundary data
    def up(self):  # boundary1 u(x,1)=0
        x = torch.rand(self.N_b, 1)
        y = torch.ones_like(x)
        b_up = self.exactsol(x, y)
        b_up_loss = self.loss(self.u(torch.cat([x, y], dim=1)), b_up)
        return x.requires_grad_(True), y.requires_grad_(True), b_up, b_up_loss

    def down(self):  # boundary2 u(x, 0)=0
        x = torch.rand(self.N_b, 1)
        y = torch.zeros_like(x)  # the same size as x
        b_down = self.exactsol(x, y)
        b_down_loss = self.loss(self.u(torch.cat([x, y], dim=1)), b_down)
        return x.requires_grad_(True), y.requires_grad_(True), b_down, b_down_loss

    def left(self):  # boundary3 u(0, y)=0
        y = torch.rand(self.N_b, 1)
        x = torch.zeros_like(y)
        b_left = self.exactsol(x, y)
        b_left_loss = self.loss(self.u(torch.cat([x, y], dim=1)), b_left)
        return x.requires_grad_(True), y.requires_grad_(True), b_left, b_left_loss

    def right(self):  # boundary4 u(1, y)=0
        y = torch.rand(self.N_b, 1)
        x = torch.ones_like(y)
        b_right = self.exactsol(x, y)
        b_right_loss = self.loss(self.u(torch.cat([x, y], dim=1)), b_right)
        return x.requires_grad_(True), y.requires_grad_(True), b_right, b_right_loss

    # pde data
    def interior(self):
        x = torch.rand(self.N_f, 1)  # randomly generate x value from [0, 1)
        y = torch.rand(self.N_f, 1)
        f = self.rhs(x, y)
        return x.requires_grad_(True), y.requires_grad_(True), f

    # reference points
    def gridpoint(self):
        xc = torch.linspace(0, 1, self.h)
        xm, ym = torch.meshgrid(xc, xc, indexing='ij')
        xx = xm.reshape(-1, 1)
        yy = ym.reshape(-1, 1)
        xy = torch.cat([xx, yy], dim=1)
        return xm, ym, xx, yy, xy

    '''Compute loss: boundary loss, gradient, pde loss, l2error '''

    def bloss(self):
        xu, yu, b_up, b_up_loss = self.up()
        xd, yd, b_down, b_down_loss = self.down()
        xl, yl, b_left, b_left_loss = self.left()
        xr, yr, b_right, b_right_loss = self.right()

        # version 1: add loss separately
        loss_b = b_up_loss + b_down_loss + b_left_loss + b_right_loss

        # version2: compute integrately
        # u_boundary = (self.u(torch.cat([xd, yd], dim=1))
        #               + self.u(torch.cat([xu, yu], dim=1))
        #               + self.u(torch.cat([xl, yl], dim=1))
        #               + self.u(torch.cat([xr, yr], dim=1)))
        # b_real = b_up + b_down + b_left + b_right
        # loss_b = self.loss(u_boundary, b_real)
        return loss_b

    def gradients(self, u, x, order=1):
        if order == 1:
            return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, only_inputs=True, )[0]
        else:
            return self.gradients(self.gradients(u, x), x, order=order - 1)

    def pdeloss(self):
        x, y, f = self.interior()
        uh = self.u(torch.cat([x, y], dim=1))
        delta_uh = self.gradients(uh, x, 2) + self.gradients(uh, y, 2)
        return self.loss(delta_uh, f)  # delta u_h-delta u

    def loss_func(self):  # 50好
        loss = 50 * self.bloss() + self.pdeloss()
        return loss

    def l2_error(self, u_pred):
        l2 = torch.norm(self.u_real - u_pred, 2)
        # l2 = torch.norm(self.u_real - u_pred, 2) / torch.norm(self.u_real, 2)
        # l2 = torch.sqrt(self.loss(u_pred, self.u_real))
        return l2


class examples(PDE_solver):
    def exactsol(self, x, y):
        u = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
        return u

    def rhs(self, x, y):
        f = -2 * torch.pi ** 2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
        return f


'''1. Compare points, iterations = 20000'''
# loss_data = pd.DataFrame()
# N_b_list = [50, 100, 250, 500]
# N_f_list = [2500, 4000, 6000]
# # N_b_list = [50, 500]
# # N_f_list = [4000, 6000]
#
# combined_list_points = [(x, y) for x in N_b_list for y in N_f_list]
# epochs = 10000
# h = 50
# fig = plt.figure(figsize=(10, 6))
#
# for k in range(len(combined_list_points)):
#     PINNnet = PINN(2, 64, 3, 1)
#     opt = torch.optim.Adam(PINNnet.parameters(), lr=1e-3)
#     pde_loss_list = []
#     L2_loss_list = []
#     loss_data_iteration = []
#
#     start_time = time.time()
#     for i in range(epochs):
#         PINNnet.train()
#         example_2D = examples(combined_list_points[k][0], combined_list_points[k][1], 50)
#         opt.zero_grad()
#         pdeloss = example_2D.loss_func()
#         pdeloss.backward(retain_graph=True)
#         opt.step()
#         pde_loss_list.append(pdeloss.item())
#
#         '''record l2 error'''
#         PINNnet.eval()
#         xm, ym, xx, yy, xy = example_2D.gridpoint()
#         u_pred = PINNnet(xy)
#         L2_loss_list.append(example_2D.l2_error(u_pred).item())
#
#         if i % 100 == 0:
#             end_time_100 = time.time()
#             run_time_100 = end_time_100 - start_time
#             print('iterations ', i, 'loss: ', pdeloss, 'l2 loss: ', example_2D.l2_error(u_pred).item(), 'time',
#                   run_time_100)
#             loss_data_iteration.append(
#                 {'Iteration': i, 'BD nodes': combined_list_points[k][0], 'Inner nodes': combined_list_points[k][1],
#                  'PDE Loss': pdeloss.item(), 'L2 Loss': example_2D.l2_error(u_pred).item(), 'time': run_time_100})
#     print('points(N_b, N_f): ', combined_list_points[k])
#     loss_data = pd.concat([loss_data, pd.DataFrame(loss_data_iteration)], ignore_index=True)



# loss_data.to_csv('loss_data_points.csv', index=False)



'''2. compare architecture of neural, N_b = 100, N_f = 2500'''
loss_data = pd.DataFrame()
neurons_list = [20, 64, 120]
hidden_layers_list = [2, 3, 4]
# neurons_list = [20, 60]
# hidden_layers_list = [1, 2]

combined_list_neural = [(x, y) for x in neurons_list for y in hidden_layers_list]
N_f = 2500
N_b = 100
epochs = 10
h = 50
fig = plt.figure(figsize=(10, 6))
# fig2 = plt.figure(figsize=(10, 6))
print('N_f: ', N_f, 'compare boundary nodes')
for k in range(len(combined_list_neural)):
    PINNnet = PINN(2, combined_list_neural[k][0], combined_list_neural[k][1], 1)
    opt = torch.optim.Adam(PINNnet.parameters(), lr=1e-3)
    pde_loss_list = []
    L2_loss_list = []
    loss_data_iteration = []

    for i in range(epochs):
        PINNnet.train()
        start_time = time.time()
        example_2D = examples(N_b, N_f, 50)
        opt.zero_grad()
        pdeloss = example_2D.loss_func()
        pdeloss.backward(retain_graph=True)
        opt.step()
        pde_loss_list.append(pdeloss.item())
        '''record l2 error'''
        PINNnet.eval()
        xm, ym, xx, yy, xy = example_2D.gridpoint()
        u_pred = PINNnet(xy)
        L2_loss_list.append(example_2D.l2_error(u_pred).item())

        if i % 100 == 0:
            print('i= ', i, 'loss: ', pdeloss)
            loss_data_iteration.append(
                {'Iteration': i, 'Neurons': combined_list_neural[k][0], 'hidden layer': combined_list_neural[k][1],
                 'PDE Loss': pdeloss.item(), 'L2 Loss': example_2D.l2_error(u_pred).item()})
    print('points(neurons, hidden layers): ', combined_list_neural[k])
    loss_data = pd.concat([loss_data, pd.DataFrame(loss_data_iteration)], ignore_index=True)
    '''pde loss'''
    plt.subplot(2, 1, 1)
    plt.plot(range(epochs), pde_loss_list, label=f'(neurons, hidden layers)={combined_list_neural[k]}')
    plt.legend()
    plt.title("PDE Loss for Different Numbers of Inner Nodes")
    plt.xlabel("Update Step")
    plt.ylabel("PDE Loss")
    plt.yscale("log")
    plt.grid(True)

    '''l2 loss'''
    plt.subplot(2, 1, 2)
    plt.plot(range(epochs), L2_loss_list, label=f'(neurons, hidden layers)={combined_list_neural[k]}')
    plt.title("L2 Loss for Different Numbers of Inner Nodes")
    plt.xlabel("Update Step")
    plt.ylabel("L2 Loss")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()

plt.savefig("loss_architecture_comparison.pdf")
plt.subplots_adjust(hspace=0.5)
plt.show()
plt.close()

loss_data.to_csv('loss_data_architecture.csv', index=False)





