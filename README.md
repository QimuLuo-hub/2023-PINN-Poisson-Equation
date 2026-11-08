# Physics-Informed Neural Networks for the 2D Poisson Equation

PyTorch implementations of physics-informed neural networks (PINNs) for the two-dimensional Poisson equation on the unit square $\Omega = (0,1)^2$:

$$\Delta u = f \quad \text{in } \Omega,$$

with **Dirichlet**, **Neumann** and **Robin** boundary conditions. The scripts study how training iterations, the number of collocation points, network architecture and boundary-loss weights affect accuracy.

Research assistant project, School of Mathematics and Physics, XJTLU (Sept 2023 – Jan 2024).

## Method

A fully connected network $u_\theta(x, y)$ (tanh activations; 64 neurons × 3 hidden layers by default) is trained by minimising

$$\mathcal{L}(\theta) = \underbrace{\frac{1}{N_f}\sum_{i=1}^{N_f}\big(\Delta u_\theta(x_i, y_i) - f(x_i, y_i)\big)^2}_{\text{PDE residual}} + \sum_{k} w_k\, \mathcal{L}_{\text{BC},k},$$

where $\Delta u_\theta$ is computed with automatic differentiation and $w_k$ are the boundary-loss weights. Collocation points are re-sampled uniformly at every iteration, and the $L^2$ error is measured against the exact solution on a $50 \times 50$ reference grid.

| Boundary condition | Test problem |
|---|---|
| **Dirichlet** | $u = x^2(x-1)^2\,y\,(y-1)^2$ with exact boundary values on all four sides (the report also uses $u = \sin \pi x \sin \pi y$) |
| **Neumann** | Same $u$; $u = 0$ on $y = 0$ and $\partial_n u = 0$ on the other sides |
| **Robin** | As Neumann, but $\partial_n u + u = 0$ on the top side $y = 1$ |

## Repository structure

```
Poisson_ANN/
├── PINN_Dirichlet(seperate).py      # single run + sweeps for the Dirichlet problem
├── PINN_Neumann(seperate).py        # mixed Dirichlet/Neumann problem, weighted BC losses
├── PINN_Robin(seperate).py          # Dirichlet + Neumann + Robin boundary losses
├── Comparison_Dirichlet(integrate).py
├── Comparison_Neumann(integrate).py # parameter studies (epochs, points, architecture, weights)
├── Comparison_Robin(integrate).py
├── compareneumann_v2.py             # revised Neumann comparison
├── PINN_Comparison_Neumann(import_bug).py  # early draft (kept for reference)
└── try.py                           # scratch experiments
```

- **`*(seperate).py`** — each boundary term is added as its own loss term. The sweeps over epochs, $(N_b, N_f)$ and network width/depth are included as commented blocks.
- **`Comparison_*(integrate).py`** — loops over settings and writes error/runtime tables with pandas.

## Running

```bash
pip install torch numpy matplotlib pandas
cd Poisson_ANN
python "PINN_Dirichlet(seperate).py"
```

Each script trains a network, prints the loss and $L^2$ error every 100 iterations, and plots the analytic solution, the PINN solution, and the loss and error curves. Hyperparameters (`epochs`, `N_b`, `N_f`, `neurons`, `hidden_layers`, boundary weights) are set at the top of each script. The random seed is fixed (`888888`) so runs can be reproduced.

## Results

The PINNs reach an $L^2$ error below $10^{-3}$ on these test problems. In the Neumann and Robin experiments, the boundary-loss weights had the largest effect on accuracy, more than the iteration count, the number of points or the architecture. The full write-up, with figures and tables, is in the companion repository [2023-ANN](https://github.com/QimuLuo-hub/2023-ANN).

## References

- M. Raissi, P. Perdikaris, G. E. Karniadakis. Physics-informed neural networks. *J. Comput. Phys.* 378 (2019) 686–707.
- S. Wang, X. Yu, P. Perdikaris. When and why PINNs fail to train: a neural tangent kernel perspective. *J. Comput. Phys.* 449 (2022) 110768.
