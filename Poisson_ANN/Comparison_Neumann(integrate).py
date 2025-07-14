import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import pandas as pd
'''  
Ensure that the result is the same every time,  each initialization is fixed. 
'''
def setup_seed(seed):
    torch.manual_seed(seed)
setup_seed(888888)


class PINN(nn.Module):
    def __init__(self, input_size, neurons, hidden_layers, output_size):
        super().__init__()
        # input layer
        self.net = nn.Sequential(nn.Linear(input_size, neurons), nn.Tanh())

        # hidden layer
        for _ in range(hidden_layers):
            self.net.add_module('hidden_layer', nn.Linear(neurons, neurons))#add layers below the input layer in a specific order
            self.net.add_module('hidden_activation', nn.Tanh())

        # output layer
        self.net.add_module('output_layer', nn.Linear(neurons, output_size))

    def forward(self, x):
        return self.net(x)

# neurons = 64
# hidden_layers = 3
# PINNnet = PINN(2, neurons, hidden_layers, 1)

class PDE_solver_Neumann:
    def __init__(self, N_b, N_f, h):
        self.N_b = N_b #self for later func's call in this class
        self.N_f = N_f
        self.h = h
        self.u = PINNnet
        self.loss = torch.nn.MSELoss(reduction='mean')

        xm, ym, xx, yy, xy = self.gridpoint()
        self.u_real = self.exactsol(xx, yy)


    '''Function to modify'''
    def exactsol(self, x, y):
        raise NotImplementedError
    def rhs(self, x, y):
        raise NotImplementedError

    '''Data preparation: Boundary, Interior, Reference points '''

    #Boundary data
    def up(self):  # boundary1 u(x,1)
        x = torch.rand(self.N_b, 1)
        y = torch.ones_like(x)
        return x.requires_grad_(True), y.requires_grad_(True)

    def down(self):  # boundary2 u(x, 0)=0
        x = torch.rand(self.N_b, 1)
        y = torch.zeros_like(x)  # the same size as x
        return x.requires_grad_(True), y.requires_grad_(True)

    def left(self):  # boundary3 u(0, y)
        y = torch.rand(self.N_b, 1)
        x = torch.zeros_like(y)
        return x.requires_grad_(True), y.requires_grad_(True)

    def right(self):  # boundary4 u(1, y)
        y = torch.rand(self.N_b, 1)
        x = torch.ones_like(y)
        return x.requires_grad_(True), y.requires_grad_(True)

    #pde data
    def interior(self):
        x = torch.rand(self.N_f, 1)
        y = torch.rand(self.N_f, 1)
        f = self.rhs(x, y)
        return x.requires_grad_(True), y.requires_grad_(True), f

    #reference points
    def gridpoint(self):
        xc = torch.linspace(0, 1, self.h)
        xm, ym = torch.meshgrid(xc, xc, indexing='ij')
        xx = xm.reshape(-1, 1)
        yy = ym.reshape(-1, 1)
        xy = torch.cat([xx, yy], dim=1)
        return xm, ym, xx, yy, xy

    '''Compute loss: boundary loss, gradient, pde loss, l2error '''
    def gradients(self, u, x, order = 1):
        if order == 1:
            return torch.autograd.grad(u, x, grad_outputs = torch.ones_like(u),
                                       create_graph=True, only_inputs=True,)[0]
        else:
            return self.gradients(self.gradients(u, x), x, order = order-1)

    def neumann_derivatives(self):
        u_x = lambda x, y: self.gradients(self.u(torch.cat([x, y], 1)), x, 1)
        u_y = lambda x, y: self.gradients(self.u(torch.cat([x, y], 1)), y, 1)
        xl, yl = self.left()#(0, y)
        xr, yr = self.right()#(1, y)
        xu, yu = self.up()#(x, 1)

        du_dx_0 = u_x(xl, yl)#(0, y)
        du_dx_1 = u_x(xr, yr)#(1, y)
        du_dy_1 = u_y(xu, yu)#(x, 1)

        return du_dx_0, du_dx_1, du_dy_1

    def neumann_loss(self):
        xl, yl = self.left()  # (0, y)
        xr, yr = self.right()  # (1, y)
        xu, yu = self.up()  # (x, 1)

        boundary_exact_left = self.exactsol(xl, yl)
        boundary_exact_right = self.exactsol(xr, yr)
        boundary_exact_up = self.exactsol(xu, yu)

        du_dx_0, du_dx_1, du_dy_1 = self.neumann_derivatives()
        loss_neumann = self.loss(du_dx_0, boundary_exact_left) + self.loss(du_dx_1, boundary_exact_right) + self.loss(du_dy_1, boundary_exact_up)
        # loss_neumann = torch.mean(du_dx_0**2) + torch.mean(du_dx_1**2) + torch.mean(du_dy_1**2)
        return loss_neumann

    def dirichlet_loss(self):
        xd, yd = self.down()
        u_down = self.u(torch.cat([xd, yd], 1))
        boundary_exact = self.exactsol(xd, yd)
        # loss_dirichlet = torch.mean(u_down**2)
        loss_dirichlet = self.loss(u_down, boundary_exact)
        return loss_dirichlet

    def pdeloss(self):
        x, y, f = self.interior()
        uh = self.u(torch.cat([x, y], dim=1))
        delta_uh = self.gradients(uh, x, 2) + self.gradients(uh, y, 2)
        return self.loss(delta_uh, f)  # delta u_h-delta u

    def loss_func(self):
        loss = self.pdeloss() + 10 * self.neumann_loss() + 250 * self.dirichlet_loss()
        return loss

    def l2_error(self, u_pred):
        # l2 = torch.norm(self.u_real - u_pred, 2) / torch.norm(self.u_real, 2)##relative l2 error
        l2 = torch.norm(self.u_real - u_pred, 2)
        # l2 = torch.sqrt(self.loss(u_pred, self.u_real))
        return l2

class examples(PDE_solver_Neumann):
    def exactsol(self, x, y):
        u = (x**2) * ((x-1)**2) * y * ((y-1)**2)
        # u = x*(1-x)*y*(1-y)
        return u

    def rhs(self, x, y):
        f = 2*((x**4)*(3*y-2) + (x**3)*(4-6*y) +
               (x**2)*(6*(y**3)-12*(y**2)+9*y-2) - 6*x*((y-1)**2)*y + ((y-1)**2)*y)
        # f = 2*(x**2+y**2-x-y)
        return f


'''1. Compare points, iterations = 10000'''
# loss_data = pd.DataFrame()
#
# #compare boundary
# # N_b_list = [50, 100, 250, 500]
# # N_f_list = [2500]
#
# #compare inner
# N_b_list = [100]
# N_f_list = [2500, 4000, 6000]
#
# combined_list_points = [(x, y) for x in N_b_list for y in N_f_list]
# epochs = 30000
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
#             print('iterations ', i, 'loss: ', pdeloss, 'l2 loss: ',example_2D.l2_error(u_pred).item(), 'time', run_time_100)
#             loss_data_iteration.append({'Iteration': i, 'BD nodes': combined_list_points[k][0], 'Inner nodes': combined_list_points[k][1],
#                                         'PDE Loss': pdeloss.item(), 'L2 Loss': example_2D.l2_error(u_pred).item(), 'time': run_time_100})
#     print('points(N_b, N_f): ', combined_list_points[k])
#     loss_data = pd.concat([loss_data, pd.DataFrame(loss_data_iteration)], ignore_index=True)
#
#     '''pde loss'''
#     #plt.plot(range(epochs), pde_loss_list, label = f'(Boundary, Inner)={combined_list_points[k]}')
#
#     '''l2 loss'''
#     plt.plot(range(epochs), L2_loss_list, label = f'(Boundary, Inner)={combined_list_points[k]}')
#
# #loss_data.to_csv('loss_data_points.csv', index=False)
#
# '''pde loss'''
# # plt.legend()
# # plt.title("PDE Loss for Different Numbers of points")
# # plt.xlabel("Update Step")
# # plt.ylabel("PDE Loss")
# # plt.yscale("log")
# # plt.grid(True)
# # plt.savefig('PDE Loss for Different Numbers of points(Neumann).pdf')
# # plt.savefig('PDE loss for Different Numbers of boundary points(Neumann).pdf')
# #plt.savefig('PDE loss for Different Numbers of Inner points(Neumann).pdf')
#
# '''l2 loss'''
# plt.legend()
# plt.title("L2 Error for Different Numbers of training data")
# plt.xlabel("Update Step")
# plt.ylabel("L2 Error")
# plt.yscale("log")
# plt.grid(True)
# # plt.savefig('PDE Loss for Different Numbers of points(Neumann).pdf')
# # plt.savefig('L2 Error for Different Numbers of boundary points(Neumann).pdf')
# plt.savefig('L2 Error for Different Numbers of Inner points(Neumann).pdf')
# plt.show()


'''2. compare architecture of neural, N_b = 100, N_f = 2500'''
loss_data = pd.DataFrame()

# #combine
neurons_list = [20, 64, 120]
hidden_layers_list = [1, 3, 9]

# compare neurons
# neurons_list = [20, 64, 120]
# hidden_layers_list = [3]

# compare hidden layer
# neurons_list = [64]
# hidden_layers_list = [1, 3, 9]

combined_list_neural = [(x, y) for x in neurons_list for y in hidden_layers_list]
epochs = 10000
h = 50
fig = plt.figure(figsize=(10, 6))

for k in range(len(combined_list_neural)):
    PINNnet = PINN(2, combined_list_neural[k][0], combined_list_neural[k][1], 1)
    opt = torch.optim.Adam(PINNnet.parameters(), lr=1e-3)
    pde_loss_list = []
    L2_loss_list = []
    loss_data_iteration = []

    start_time = time.time()
    for i in range(epochs):
            PINNnet.train()
            example_2D = examples(100, 2500, 50)
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
                end_time_100 = time.time()
                run_time_100 = end_time_100 - start_time
                print('iterations ', i, 'loss: ', pdeloss, 'l2 loss: ',example_2D.l2_error(u_pred).item(), 'time', run_time_100)
                loss_data_iteration.append({'Iteration': i, 'neurons': combined_list_neural[k][0], 'layers': combined_list_neural[k][1],
                                            'PDE Loss': pdeloss.item(), 'L2 Loss': example_2D.l2_error(u_pred).item(), 'time': run_time_100})
    print('points(neurons, hidden layers): ', combined_list_neural[k])
    loss_data = pd.concat([loss_data, pd.DataFrame(loss_data_iteration)], ignore_index=True)
    '''pde loss'''
    plt.plot(range(epochs), pde_loss_list, label = f'(neuron, hidden layer)={combined_list_neural[k]}')

    '''l2 loss'''
    #plt.plot(range(epochs), L2_loss_list, label = f'(neuron, hidden layer)={combined_list_neural[k]}')

#loss_data.to_csv('loss_data_neural.csv', index=False)

'''pde loss'''
plt.legend()
plt.title("PDE Loss for Different architectures")
plt.xlabel("Update Step")
plt.ylabel("PDE Loss")
plt.yscale("log")
plt.grid(True)
plt.savefig('PDE Loss for Different architectures(Neumann).pdf')

'''l2 loss'''
# plt.legend()
# plt.title("L2 Error for Different architectures")
# plt.xlabel("Update Step")
# plt.ylabel("L2 Error")
# plt.yscale("log")
# plt.grid(True)
#plt.savefig('L2 Error for Different architectures(Neumann).pdf')
# plt.savefig('L2 Error for Different neurons(Neumann).pdf')
# plt.savefig('L2 Error for Different hidden layers(Neumann).pdf')

plt.show()
