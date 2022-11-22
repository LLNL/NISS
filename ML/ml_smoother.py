# from torch.autograd import Variable
# import numpy as np
# import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from UTILS.utils import *
from LFA.smooth import SmoothSymbl2D
from LFA.stencil import StencilSymbl2D
from LFA.theta import Theta2D

# For reproducibility
torch.manual_seed(0)
np.random.seed(0)
torch.set_printoptions(precision=5)
# precision
# torch.set_default_dtype(torch.float64)


class Net(nn.Module):
    def __init__(self, size):
        super(Net, self).__init__()
        self.fc = nn.Linear(size, 1, bias=False)
        # self.flatten = nn.Flatten(0, 1)

    def forward(self, x):
        x = self.fc(x)
        # x = self.flatten(x)
        return x


def symbol_measure(symbol, measure_type):
    if measure_type == 1:
        return torch.max(torch.norm(symbol, dim=0))
    elif measure_type == 2:
        # return torch.sum(torch.norm(symbol, dim=0)) / symbol.size(1)
        return torch.sum(torch.square(symbol)) / symbol.numel()


class SmootherLoss(nn.Module):
    def __init__(self, operator_symbol, measure_type=1):
        super(SmootherLoss, self).__init__()
        self.operator_symbol = operator_symbol
        self.type = measure_type

    def forward(self, smoother_symbol):
        smoother_symbol = smoother_symbol.squeeze(-1)
        symbol = complex_multiply(self.operator_symbol, smoother_symbol)
        if symbol.size(0) == 1:
            symbol = 1 - symbol
        else:
            symbol = torch.tensor([[1], [0]]) - symbol
        return symbol_measure(symbol, self.type)


# centrosymmetric
A_symmetric = True
M_symmetric = True
#
loss_measure = 2

# 1. test the given smoother
# theta grid
theta_grid = Theta2D(128, quadrant=torch.tensor([1, 2, 3]))
# stencil for A
A_size = 3
pattern_A = torch.ones(A_size, A_size)
center_A = torch.tensor([A_size // 2, A_size // 2])
A = -1 / 3 * torch.ones([A_size, A_size])
A[center_A[0], center_A[1]] = 8 / 3
# zero out "strict-upper" part to test centrosymmetric
if A_symmetric and A_size > 1:
    A[centrosymmetric_strict_upper_coord(A_size)] = 0
# stencil for M (given the optimal weight to learn)
M_size = 3
if M_size == 1:
    pattern_M = torch.ones([M_size, M_size])
    center_M = torch.tensor([M_size // 2, M_size // 2])
    M = torch.zeros([M_size, M_size])
    M[center_M[0], center_M[1]] = 1 / 3
else:
    pattern_M = torch.ones([M_size, M_size])
    center_M = torch.tensor([M_size // 2, M_size // 2])
    M = torch.ones([M_size, M_size])
    M[center_M[0], center_M[1]] = 10
    M = M * (2 / 51)
# zero out "strict-upper" part to test centrosymmetric
if M_symmetric and M_size > 1:
    M[centrosymmetric_strict_upper_coord(M_size)] = 0
# the optimal smooth factor
smooth_operator = SmoothSymbl2D(pattern_A, center_A, pattern_M, center_M,
                                mat_a=A, centrosymmetric_a=A_symmetric,
                                mat_m=M, centrosymmetric_m=M_symmetric)
smooth_operator.setup_theta(theta_grid)
smooth_symbol = smooth_operator.symbol()
smooth_symbol_mod = torch.norm(smooth_symbol, dim=0)
smooth_factor = torch.max(smooth_symbol_mod)
print(f'* Optimal Inf-norm Smoothing factor = {smooth_factor.item():.6e}')
smooth_factor = symbol_measure(smooth_symbol, measure_type=loss_measure)
print(f'* Optimal Measured Smoothing factor = {smooth_factor.item():.6e}')

# 2. Use NN to learn M
stencil_A = StencilSymbl2D(pattern_A, center_A, mat=A, centrosymmetric=A_symmetric)
stencil_M = StencilSymbl2D(pattern_M, center_M, mat=None, centrosymmetric=M_symmetric)
stencil_A.setup_theta(theta_grid)
stencil_M.setup_theta(theta_grid)
# create a NN to learn the stencil weights of M
net = Net(size=stencil_M.stencil_value_len)
print(net)
# define loss function
A_symbol = stencil_A.symbol().flatten(start_dim=1)
criterion = SmootherLoss(A_symbol, measure_type=loss_measure)
# NN Input
NNInput = stencil_M.cos_sin_theta_stencil.flatten(start_dim=1, end_dim=3)
if stencil_M.is_symmetric():
    NNInput = torch.matmul(NNInput, stencil_M.sym_extend.transpose(0, 1))
# create a stochastic gradient descent optimizer
optimizer = optim.SGD(net.parameters(), lr=0.001)
# training loop
num_steps = 1000
num_print = 20
print_interval = num_steps // num_print
for t in range(num_steps):
    optimizer.zero_grad()  # zero the gradient buffers
    NNOutput = net(NNInput)
    loss = criterion(NNOutput)
    if np.remainder(t, print_interval) == 0:
        print(f'Step {t:5} : Measured Smoothing factor = {loss.item():.4e}')
    # compute gradient
    loss.backward()
    # Does the update
    optimizer.step()
# Get smoother weights from NN
linear_layer = net.fc
print(f'= = = = = = = = = = = = = = = = =')
print(f'* Final Measured Smoothing factor = {criterion(net(NNInput)).item():.6e} ')
print(f'* Smoother Weights')
print(f' {np.array2string(linear_layer.weight.detach().numpy())}')
# verify with the smoother class
if stencil_M.is_symmetric():
    M = torch.zeros([M_size, M_size])
    M[centrosymmetric_lower_coord(M_size)] = linear_layer.weight.detach().clone()
    centrosymmetrize_upper(M)
else:
    M = linear_layer.weight.detach().clone().reshape(M_size, M_size)
print(f'* Smoother Stencil')
print(f'{np.array2string(M.detach().numpy())}')
# print(f'= = = = = = = = = = = = = = = = =')
# Verify with SmoothSymbl2D class
smooth_operator = SmoothSymbl2D(pattern_A, center_A, pattern_M, center_M,
                                mat_a=A, centrosymmetric_a=A_symmetric,
                                mat_m=M, centrosymmetric_m=M_symmetric)
theta_grid = Theta2D(128, quadrant=torch.tensor([0, 1, 2, 3]))
smooth_operator.setup_theta(theta_grid)
smooth_symbol = smooth_operator.symbol()
smooth_symbol_mod = torch.norm(smooth_symbol, dim=0)
smooth_factor = torch.max(smooth_symbol_mod[:, :, 1:4])
print(f'* Learned Inf-norm Smoothing factor = {smooth_factor.item():.6e}')
smooth_factor = symbol_measure(smooth_symbol[:, :, :, 1:4], measure_type=loss_measure)
print(f'* Learned Measured Smoothing factor = {smooth_factor.item():.6e}')

# plot
theta_grid.plot(smooth_symbol_mod, title='Smoothing factor', num_levels=64)
plt.show()
