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

class PDE_solver:
    def __init__(self, N_b, N_f, h, D_weight,  N_weight):
        self.N_b = N_b
        self.N_f = N_f
        self.h = h
        self.D_weight = D_weight
        self.N_weight = N_weight


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
        loss = self.pdeloss() + self.D_weight * self.dirichlet_loss() + self.N_weight * self.neumann_loss()
        return loss

    def l2_error(self, u_pred):
        # l2 = torch.norm(self.u_real - u_pred, 2) / torch.norm(self.u_real, 2)
        l2 = torch.norm(self.u_real - u_pred, 2)
        # l2 = torch.sqrt(self.loss(u_pred, self.u_real))
        return l2

class examples(PDE_solver):
    def exactsol(self, x, y):
        u = (x**2) * ((x-1)**2) * y * ((y-1)**2)
        # u = x*(1-x)*y*(1-y)
        return u

    def rhs(self, x, y):
        f = 2*((x**4)*(3*y-2) + (x**3)*(4-6*y) +
               (x**2)*(6*(y**3)-12*(y**2)+9*y-2) - 6*x*((y-1)**2)*y + ((y-1)**2)*y)
        # f = 2*(x**2+y**2-x-y)
        return f

h = 50
def train(epochs, N_b, N_f, D_weight, N_weight):
    MSE = []
    l2 = []
    opt = torch.optim.Adam(PINNnet.parameters(), lr=1e-3)

    start_time = time.time()
    for i in range(epochs):
        PINNnet.train()
        example_2D = examples(N_b, N_f, h, D_weight, N_weight)
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
    ax2.set_title(f"Numerical Solution by Neural Networks(D={D_weight},N={N_weight})")

    ax3 = fig1.add_subplot(133, projection='3d')
    ax3.plot_surface(xm, ym, u_pred_fig, cmap='Blues')  # blue for pred
    ax3.plot_surface(xm, ym, u_real_fig, cmap='Reds')  # red for real network
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title("Comparison")

    # fig1.savefig(f'Comparison_for_Poisson(Neumann)_epochs{epochs}_weights{D_weight}{N_weight}.pdf')

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

    # fig2.savefig(f'MSE_for_Poisson(Neumann)_epochs{epochs}_weights_{D_weight}{N_weight}.pdf')
    # fig3.savefig(f'L2_for_Poisson(Neumann)_epochs{epochs}_weights_{D_weight}{N_weight}.pdf')

    plt.show()
    return MSE, l2_error, run_time, fig1, fig2, fig3



'''compare separately'''

#compare weights
# Dirichlet_list = [50, 100, 250, 300]
# Neumann_list = [10, 20]
# weights_list = [(x, y) for x in Dirichlet_list for y in Neumann_list]
# # epochs_list = [(i+1) * 10 **4 for i in range(5)]
# l2_error_list_weights = np.zeros(len(weights_list))
# time_list_weights = np.zeros(len(weights_list))
#
# for j in range(len(weights_list)):
#     PINNnet = PINN(2, 64, 3, 1)
#     loss, l2_error, run_time, compare_fig, mse_fig, l2error_fig = train(40000, 100, 2500,
#                                                                         weights_list[j][0], weights_list[j][1])
#     l2_error_list_weights[j] = l2_error
#     time_list_weights[j] = run_time
#
#     # save fig
#     compare_fig_name = f'Comparison_(Robin)__(Weight_D, Weight_N){weights_list[j]}.pdf'
#     mse_fig_name = f'MSE_for_Poisson(Robin)_(Weight_D, Weight_N){weights_list[j]}.pdf'
#     l2error_fig_name = f'L2_for_Poisson(Robin)_(Weight_D, Weight_N){weights_list[j]}.pdf'
#
#     compare_fig.savefig(compare_fig_name)
#     mse_fig.savefig(mse_fig_name)
#     l2error_fig.savefig(l2error_fig_name)
#
#     print('weights: ', weights_list[j], 'runtime is', time_list_weights[j])
#
# # Table for epochs
# df = pd.DataFrame({'Weights': weights_list, 'Error': l2_error_list_weights, 'Runtime': time_list_weights})
# '''convert data type'''
# df['Error'] = df['Error'].apply(lambda x: f"{x:.3e}")
# df['Runtime'] = df['Runtime'].apply(lambda x: f"{x:.3f}")  # （Keep 2 decimal place）
# print(df)
# df.to_csv('Table for Weights(Neumann).csv', index=False)

'''1.epochs'''
# # epochs_list = [5000, 10000, 50000]
epochs_list = [(i+1) * 10 **4 for i in range(5)]
l2_error_list_epochs = np.zeros(len(epochs_list))
time_list_epochs = np.zeros(len(epochs_list))

for j in range(len(epochs_list)):
    PINNnet = PINN(2, 64, 3, 1)
    loss, l2_error, run_time, compare_fig, mse_fig, l2error_fig = train(epochs_list[j], 100, 2500)
    l2_error_list_epochs[j] = l2_error
    time_list_epochs[j] = run_time

    # save fig
    compare_fig_name = f'Comparison_(Neumann)_epochs{epochs_list[j]}.pdf'
    mse_fig_name = f'MSE_for_Poisson(Neumann)_Solution_epochs{epochs_list[j]}.pdf'
    l2error_fig_name = f'L2_for_Poisson(Neumann)_Solution_epochs{epochs_list[j]}.pdf'

    # compare_fig.savefig(compare_fig_name)
    # mse_fig.savefig(mse_fig_name)
    # l2error_fig.savefig(l2error_fig_name)

    print('iteration: ', epochs_list[j], 'runtime is', time_list_epochs[j])

# Table for epochs
df = pd.DataFrame({'Iterations': epochs_list, 'Error': l2_error_list_epochs, 'Runtime': time_list_epochs})
'''convert data type'''
df['Error'] = df['Error'].apply(lambda x: f"{x:.3e}")
df['Runtime'] = df['Runtime'].apply(lambda x: f"{x:.3f}")  # （Keep 2 decimal place）
print(df)
df.to_csv('Table for epochs(Neumann).csv', index=False)

'''#2.points'''
# # N_b_list = [50, 500]
# # N_f_list = [2500]
#
# N_b_list = [50, 100, 250, 500]
# N_f_list = [2500, 4000, 6000]
#
#
# combined_list_points = [(x, y) for x in N_b_list for y in N_f_list]
# l2_error_list_points = np.zeros((len(combined_list_points)))
# time_list_points = np.zeros((len(combined_list_points)))
# for k in range(len(combined_list_points)):
#             PINNnet = PINN(2, 64, 3, 1)
#             loss, l2_error, run_time, compare_fig, mse_fig, l2error_fig = train(10000, combined_list_points[k][0], combined_list_points[k][1])
#             l2_error_list_points[k] = l2_error
#             time_list_points[k] = run_time
#
#             # save fig
#             compare_fig_name = f'Comparison_(Neumann)_(N_b, N_f)_{combined_list_points[k]}.pdf'
#             mse_fig_name = f'MSE_for_Poisson(Neumann)_Solution_(N_b, N_f)_{combined_list_points[k]}.pdf'
#             l2error_fig_name = f'L2_for_Poisson(Neumann)_Solution_(N_b, N_f)_{combined_list_points[k]}.pdf'
#
#             compare_fig.savefig(compare_fig_name)
#             mse_fig.savefig(mse_fig_name)
#             l2error_fig.savefig(l2error_fig_name)
#             print('boundary: ', combined_list_points[k][0], 'Interior: ', combined_list_points[k][1], '\n',
#                           'run time', run_time, 'l2 error', l2_error)
#
# df = pd.DataFrame({'Points': combined_list_points, 'Error': l2_error_list_points, 'Runtime': time_list_points})
#
# '''convert data type'''
# df['Error'] = df['Error'].apply(lambda x: f"{x:.3e}")
# df['Runtime'] = df['Runtime'].apply(lambda x: f"{x:.3f}")  # （Keep 2 decimal place）
# print(df)
# df.to_csv('Table for points(Neumann).csv', index=False)

'''#3.architecture'''
# neurons_list = [20, 64, 120]
# hidden_layers_list = [1, 3, 9]
# # neurons_list = [20]
# # hidden_layers_list = [9]
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
#     compare_fig_name = f'Comparison_(Neumann)_(neurons, hidden layers)_{combined_list_neural[a]}.pdf'
#     mse_fig_name = f'MSE_for_Poisson(Neumann)_(neurons, hidden layers)_{combined_list_neural[a]}.pdf'
#     l2error_fig_name = f'L2_for_Poisson(Neumann)_(neurons, hidden layers)_{combined_list_neural[a]}.pdf'
#
#     compare_fig.savefig(compare_fig_name)
#     mse_fig.savefig(mse_fig_name)
#     l2error_fig.savefig(l2error_fig_name)
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
# df.to_csv('Table for architecture(Neumann).csv', index=False)