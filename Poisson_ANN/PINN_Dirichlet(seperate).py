import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import time
# import pandas as pd

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
            self.net.add_module('hidden_layer', nn.Linear(neurons, neurons))#add layers below the input layer in a specific order
            self.net.add_module('hidden_activation', nn.Tanh())

        # output layer
        self.net.add_module('output_layer', nn.Linear(neurons, output_size))

    def forward(self, x):
        return self.net(x)

neurons = 64
hidden_layers = 3
PINNnet = PINN(2, neurons, hidden_layers, 1)

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

    #Boundary data
    def up(self):  # boundary1 u(x,1)=0
        x = torch.rand(self.N_b, 1)
        y = torch.ones_like(x)
        b_up = self.exactsol(x, y)
        b_up_loss = self.loss(self.u(torch.cat([x, y], dim=1)), b_up)
        return x.requires_grad_(True), y.requires_grad_(True), b_up, b_up_loss
    #
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

    #pde data
    def interior(self):
        x = torch.rand(self.N_f, 1)  # randomly generate x value from [0, 1)
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

    def bloss(self):
        xu, yu, b_up, b_up_loss = self.up()
        xd, yd, b_down, b_down_loss = self.down()
        xl, yl, b_left, b_left_loss = self.left()
        xr, yr, b_right, b_right_loss = self.right()

        #version 1: add loss separately
        loss_b = b_up_loss + b_down_loss + b_left_loss + b_right_loss

        #version2: compute integrately
        # u_boundary = (self.u(torch.cat([xd, yd], dim=1))
        #               + self.u(torch.cat([xu, yu], dim=1))
        #               + self.u(torch.cat([xl, yl], dim=1))
        #               + self.u(torch.cat([xr, yr], dim=1)))
        # b_real = b_up + b_down + b_left + b_right
        # loss_b = self.loss(u_boundary, b_real)
        return loss_b


    def gradients(self, u, x, order = 1):
        if order == 1:
            return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, only_inputs=True,)[0]
        else:
            return self.gradients(self.gradients(u, x), x, order = order-1)

    def pdeloss(self):
        x, y, f = self.interior()
        uh = self.u(torch.cat([x, y], dim=1))
        delta_uh = self.gradients(uh, x, 2) + self.gradients(uh, y, 2)
        return self.loss(delta_uh, f)  # delta u_h-delta u

    def loss_func(self):
        loss = 500 * self.bloss() + self.pdeloss()
        return loss

    def l2_error(self, u_pred):
        l2 = torch.norm(self.u_real - u_pred, 2)
        #l2 = torch.norm(self.u_real - u_pred, 2) / torch.norm(self.u_real, 2)
        # l2 = torch.sqrt(self.loss(u_pred, self.u_real))
        return l2

# class examples(PDE_solver):
#     def exactsol(self, x, y):
#         u = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
#         return u
#     def rhs(self, x, y):
#         f = -2 * torch.pi ** 2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
#         return f
class examples(PDE_solver):
    def exactsol(self, x, y):
        u = (x**2) * ((x-1)**2) * y * ((y-1)**2)
        return u

    def rhs(self, x, y):
        f = 2*((x**4)*(3*y-2) + (x**3)*(4-6*y) +
               (x**2)*(6*(y**3)-12*(y**2)+9*y-2) - 6*x*((y-1)**2)*y + ((y-1)**2)*y)
        return f

'''plug in value'''
epochs = 50000
N_b = 100
N_f = 2500
h = 50

def train(epochs, N_b, N_f):
    MSE = []
    l2 = []
    opt = torch.optim.Adam(PINNnet.parameters(), lr=0.001)

    start_time = time.time()
    for i in range(epochs):
        PINNnet.train()
        example_2D = examples(N_b, N_f, h)
        '''train pde network, record ploss'''
        opt.zero_grad()
        l_mse = example_2D.loss_func()
        l_mse.backward()
        opt.step()
        MSE.append(l_mse.item())

        '''record l2 error'''
        PINNnet.eval()
        xm, ym, xx, yy, xy = example_2D.gridpoint()
        u_pred = PINNnet(xy)
        l2.append(example_2D.l2_error(u_pred).item())
        if i % 100 == 0:
            end_time_100 = time.time()
            run_time_100 = end_time_100 - start_time
            print('interation= ', i, 'pde_loss: ', l_mse, 'l2 error',example_2D.l2_error(u_pred).item(),'run time:', run_time_100)

    end_time = time.time()
    run_time = end_time - start_time
    l2_error = np.min(example_2D.l2_error(u_pred).item())

    iterations_list = [i for i in range(1, epochs + 1)]
    xm, ym, xx, yy, xy = example_2D.gridpoint()

    '''Numerical and Analytical'''
    u_pred_fig = u_pred.reshape(h, h).detach().numpy()
    u_real_fig = example_2D.u_real.reshape(h, h).detach().numpy()

    fig1 = plt.figure(figsize=(15, 5))

    ax1 = fig1.add_subplot(131, projection='3d')
    ax1.plot_surface(xm, ym, u_real_fig, cmap='jet')
    ax1.set_title("Analytic Solution")

    ax2 = fig1.add_subplot(132, projection='3d')
    ax2.plot_surface(xm, ym, u_pred_fig, cmap='jet')
    ax2.set_title("Numerical Solution by Neural Networks")

    ax3 = fig1.add_subplot(133, projection='3d')
    ax3.plot_surface(xm, ym, u_pred_fig, cmap='Blues')  # blue for pred
    ax3.plot_surface(xm, ym, u_real_fig, cmap='Reds')  # red for real network
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title("Comparison")

    # fig1.savefig('Comparison_of_Numerical_and_Analytical.pdf')

    '''Record loss'''
    fig2 = plt.figure()
    fig3 = plt.figure()

    # MSEloss: MSE_u + MSE_f
    ax1 = fig2.add_subplot(111)
    ax1.plot(iterations_list, MSE, 'r-+', linewidth=2, label='loss')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('MSE Loss')
    ax1.set_xscale('linear')
    ax1.set_yscale('log')
    ax1.legend()

    # l2 error
    ax2 = fig3.add_subplot(111)
    ax2.plot(iterations_list, l2, 'b-+', linewidth=2, label='L2 Error')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('L2 Error')
    ax2.set_xscale('linear')
    ax2.set_yscale('log')
    ax2.legend()

    # fig2.savefig('MSE_for_Poisson(Dirichlet)_Solution.pdf')
    # fig3.savefig('L2_for_Poisson(Dirichlet)_Solution.pdf')

    plt.show()
    return MSE, l2_error, run_time, fig1, fig2, fig3

train(epochs, N_b, N_f)

'''compare separately'''

'''#1.epochs'''
# epochs_list = [100, 200, 300, 400]
# epochs_list = [(i+1) * 10 **4 for i in range(5)]
# l2_error_list_epochs = np.zeros(len(epochs_list))
# time_list_epochs = np.zeros(len(epochs_list))
#
# for j in range(len(epochs_list)):
#     PINNnet = PINN(2, 64, 3, 1)
#     loss, l2_error, run_time, compare_fig, mse_fig, l2error_fig = train(epochs_list[j], 100, 2500)
#     l2_error_list_epochs[j] = l2_error
#     time_list_epochs[j] = run_time
#
#     compare_fig_name = f'Comparison_of_Numerical_and_Analytical_epochs{epochs_list[j]}.pdf'
#     mse_fig_name = f'MSE_for_Poisson(Dirichlet)_Solution_epochs{epochs_list[j]}.pdf'
#     l2error_fig_name = f'L2_for_Poisson(Dirichlet)_Solution_epochs{epochs_list[j]}.pdf'
#
#     # save fig
#     # compare_fig.savefig(compare_fig_name)
#     # mse_fig.savefig(mse_fig_name)
#     # l2error_fig.savefig(l2error_fig_name)
#
#     print('iteration: ', epochs_list[j], 'runtime is', time_list_epochs[j])
#
# # Table for epochs
# df = pd.DataFrame({'Iterations': epochs_list, 'Error': l2_error_list_epochs, 'Runtime': time_list_epochs})
# '''convert data type'''
# df['Error'] = df['Error'].apply(lambda x: f"{x:.3e}")
# df['Runtime'] = df['Runtime'].apply(lambda x: f"{x:.3f}")  # （Keep 2 decimal place）
# print(df)
# df.to_csv('Table for epochs(D).csv', index=False)

'''#2.points'''
# N_b_list = [50, 500]
# N_f_list = [2500]
# N_b_list = [50, 100, 250, 500]
# N_f_list = [2500, 4000, 6000]
# combined_list_points = [(x, y) for x in N_b_list for y in N_f_list]
# l2_error_list_points = np.zeros((len(combined_list_points)))
# time_list_points = np.zeros((len(combined_list_points)))
# for k in range(len(combined_list_points)):
#             PINNnet = PINN(2, 64, 3, 1)
#             loss, l2_error, run_time, compare_fig, mse_fig, l2error_fig = train(10000, combined_list_points[k][0], combined_list_points[k][1])
#             l2_error_list_points[k] = l2_error
#             time_list_points[k] = run_time

            # save fig
            # compare_fig_name = f'Comparison_of_Numerical_and_Analytical_(N_b, N_f)_{combined_list_points[k]}.pdf'
            # mse_fig_name = f'MSE_for_Poisson(Dirichlet)_Solution_(N_b, N_f)_{combined_list_points[k]}.pdf'
            # l2error_fig_name = f'L2_for_Poisson(Dirichlet)_Solution_(N_b, N_f)_{combined_list_points[k]}.pdf'
            #
            # compare_fig.savefig(compare_fig_name)
            # mse_fig.savefig(mse_fig_name)
            # l2error_fig.savefig(l2error_fig_name)
            # print('boundary: ', combined_list_points[k][0], 'Interior: ', combined_list_points[k][1], '\n',
            #               'run time', run_time, 'l2 error', l2_error)

# df = pd.DataFrame({'Points': combined_list_points, 'Error': l2_error_list_points, 'Runtime': time_list_points})
# '''convert data type'''
# df['Error'] = df['Error'].apply(lambda x: f"{x:.3e}")
# df['Runtime'] = df['Runtime'].apply(lambda x: f"{x:.3f}")  # （Keep 2 decimal place）
# print(df)
# df.to_csv('Table for points.csv', index=False)


'''#3.architecture'''
# neurons_list = [20, 64, 120]
# hidden_layers_list = [1, 3, 9]
# neurons_list = [20]
# hidden_layers_list = [1]
# combined_list_neural = [(x, y) for x in neurons_list for y in hidden_layers_list]
# l2_error_list_neural = np.zeros((len(combined_list_neural)))
# time_list_neural = np.zeros((len(combined_list_neural)))
#
# for a in range(len(combined_list_neural)):
#     PINNnet = PINN(2, combined_list_neural[a][0], combined_list_neural[a][1], 1)
#     loss, l2_error, run_time, compare_fig, mse_fig, l2error_fig = train(10000, 100, 2500)
#     time_list_neural[a] = run_time
#     l2_error_list_neural[a]=l2_error
#
#     # save fig
#     # compare_fig_name = f'Comparison_(Dirichlet)_(neurons, hidden layers)_{combined_list_neural[a]}.pdf'
#     # mse_fig_name = f'MSE_for_Poisson(Dirichlet)_(neurons, hidden layers)_{combined_list_neural[a]}.pdf'
#     # l2error_fig_name = f'L2_for_Poisson(Dirichlet)_(neurons, hidden layers)_{combined_list_neural[a]}.pdf'
#     #
#     # compare_fig.savefig(compare_fig_name)
#     # mse_fig.savefig(mse_fig_name)
#     # l2error_fig.savefig(l2error_fig_name)
#
#     print('interation: ', 10000, 'neurons: ', combined_list_neural[a][0], 'hidden layers: ',
#           combined_list_neural[a][1], '\n',
#           'run time', run_time, 'l2 error', l2_error)
# df = pd.DataFrame({'Neural structure(neurons, layers)': combined_list_neural, 'Error': l2_error_list_neural,
#                    'Runtime': time_list_neural})
#
# '''convert data type'''
# df['Error'] = df['Error'].apply(lambda x: f"{x:.3e}")
# df['Runtime'] = df['Runtime'].apply(lambda x: f"{x:.3f}")  # （Keep 2 decimal place）
# print(df)
# df.to_csv('Table for architecture.csv', index=False)



