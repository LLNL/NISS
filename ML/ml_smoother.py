# from torch.autograd import Variable
# import numpy as np
# import torch
import torch.nn as nn
# import torch.nn.functional as F
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
        # return torch.sum(torch.square(symbol)) # / symbol.numel()
        return torch.mean(torch.square(symbol))


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


# 0. test the "optimal" smoother
def test0(a_symmetric=True, m_symmetric=True, loss_measure=2, m_size=3):
    # theta grid
    theta_grid = Theta2D(128, quadrant=torch.tensor([1, 2, 3]))
    # stencil for A
    a_size = 3
    pattern_a = torch.ones(a_size, a_size)
    center_a = torch.tensor([a_size // 2, a_size // 2])
    mat_a = -1 / 3 * torch.ones([a_size, a_size])
    mat_a[center_a[0], center_a[1]] = 8 / 3
    # zero out "strict-upper" part to test centrosymmetric
    if a_symmetric and a_size > 1:
        mat_a[centrosymmetric_strict_upper_coord(a_size)] = 0
    # stencil for M (given the optimal weight to learn)
    if m_size != 3:
        pattern_m = torch.ones([m_size, m_size])
        center_m = torch.tensor([m_size // 2, m_size // 2])
        mat_m = torch.zeros([m_size, m_size])
        mat_m[center_m[0], center_m[1]] = 1 / 3
    else:
        pattern_m = torch.ones([m_size, m_size])
        center_m = torch.tensor([m_size // 2, m_size // 2])
        mat_m = torch.ones([m_size, m_size])
        mat_m[center_m[0], center_m[1]] = 10
        mat_m = mat_m * (2 / 51)
    print(f'* Optimal Smoother Stencil')
    print(f'{np.array2string(mat_m.detach().numpy(), precision=3)}')
    # zero out "strict-upper" part to test centrosymmetric
    if m_symmetric and m_size > 1:
        mat_m[centrosymmetric_strict_upper_coord(m_size)] = 0
    # the optimal smooth factor
    smooth_operator = SmoothSymbl2D(pattern_a, center_a, pattern_m, center_m,
                                    mat_a=mat_a, centrosymmetric_a=a_symmetric,
                                    mat_m=mat_m, centrosymmetric_m=m_symmetric)
    smooth_operator.setup_theta(theta_grid)
    smooth_symbol = smooth_operator.symbol()
    smooth_symbol_mod = torch.norm(smooth_symbol, dim=0)
    smooth_factor = torch.max(smooth_symbol_mod)
    print(f'* Optimal Inf-norm Smoothing factor = {smooth_factor.item():.6e}')
    smooth_factor = symbol_measure(smooth_symbol, measure_type=loss_measure)
    print(f'* Optimal Measured Smoothing factor = {smooth_factor.item():.6e}')
    # plot
    theta_grid.plot(smooth_symbol_mod, title='Smoothing factor', num_levels=16)


# 1. Use NN to learn M
def test1(a_symmetric=True, m_symmetric=True, loss_measure=2, m_size=3):
    # theta grid
    theta_grid = Theta2D(128, quadrant=torch.tensor([1, 2, 3]))
    # stencil for A
    a_size = 3
    pattern_a = torch.ones(a_size, a_size)
    center_a = torch.tensor([a_size // 2, a_size // 2])
    mat_a = -1 / 3 * torch.ones([a_size, a_size])
    mat_a[center_a[0], center_a[1]] = 8 / 3
    # zero out "strict-upper" part to test centrosymmetric
    if a_symmetric and a_size > 1:
        mat_a[centrosymmetric_strict_upper_coord(a_size)] = 0
    # stencil shape for M
    pattern_m = torch.ones([m_size, m_size])
    center_m = torch.tensor([m_size // 2, m_size // 2])
    stencil_a = StencilSymbl2D(pattern_a, center_a, mat=mat_a, centrosymmetric=a_symmetric)
    stencil_m = StencilSymbl2D(pattern_m, center_m, mat=None, centrosymmetric=m_symmetric)
    stencil_a.setup_theta(theta_grid)
    stencil_m.setup_theta(theta_grid)
    # create a NN to learn the stencil weights of M
    net = Net(size=stencil_m.stencil_value_len)
    print(net)
    # define loss function
    a_symbol = stencil_a.symbol().flatten(start_dim=1)
    criterion = SmootherLoss(a_symbol, measure_type=loss_measure)
    # NN Input
    nn_input = stencil_m.cos_sin_theta_stencil.flatten(start_dim=1, end_dim=3)
    if stencil_m.is_symmetric():
        nn_input = torch.matmul(nn_input, stencil_m.sym_extend.transpose(0, 1))
    # create a stochastic gradient descent optimizer.
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.0005)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0004)
    # optimizer = torch.optim.LBFGS(net.parameters(), lr=0.1)
    print(f'Optimizer used: {type(optimizer).__name__}')
    # training loop
    num_steps = 3000
    num_print = 20
    print_interval = num_steps // num_print
    if type(optimizer).__name__ == "LBFGS":
        for t in range(num_steps):
            def closure():
                optimizer.zero_grad()
                nn_output_ = net(nn_input)
                loss_ = criterion(nn_output_)
                loss_.backward()
                return loss_
            loss = closure()
            if np.remainder(t, print_interval) == 0:
                print(f'Step {t:5} : Measured Smoothing factor = {loss.item():.4e}')
            optimizer.step(closure)
    else:
        for t in range(num_steps):
            optimizer.zero_grad()  # zero the gradient buffers
            nn_output = net(nn_input)
            loss = criterion(nn_output)
            if np.remainder(t, print_interval) == 0:
                print(f'Step {t:5} : Measured Smoothing factor = {loss.item():.4e}')
            # compute gradient
            loss.backward()
            # Does the update
            optimizer.step()
    # Get smoother weights from NN
    linear_layer = net.fc
    print(f'= = = = = = = = = = = = = = = = =')
    print(f'* Final Measured Smoothing factor = {criterion(net(nn_input)).item():.6e} ')
    print(f'* Smoother Weights')
    print(f'{np.array2string(linear_layer.weight.detach().numpy())}')
    # verify with the smoother class
    if stencil_m.is_symmetric():
        mat_m = torch.zeros([m_size, m_size])
        mat_m[centrosymmetric_lower_coord(m_size)] = linear_layer.weight.detach().clone()
        centrosymmetrize_upper(mat_m)
    else:
        mat_m = linear_layer.weight.detach().clone().reshape(m_size, m_size)
    print(f'* Smoother Stencil')
    print(f'{np.array2string(mat_m.detach().numpy(), precision=3)}')
    # print(f'= = = = = = = = = = = = = = = = =')
    # Verify with SmoothSymbl2D class
    smooth_operator = SmoothSymbl2D(pattern_a, center_a, pattern_m, center_m,
                                    mat_a=mat_a, centrosymmetric_a=a_symmetric,
                                    mat_m=mat_m, centrosymmetric_m=m_symmetric)
    theta_grid = Theta2D(128, quadrant=torch.tensor([1, 2, 3]))
    smooth_operator.setup_theta(theta_grid)
    smooth_symbol = smooth_operator.symbol()
    smooth_symbol_mod = torch.norm(smooth_symbol, dim=0)
    smooth_factor = torch.max(smooth_symbol_mod[:, :, 0:3])
    print(f'* Learned Inf-norm Smoothing factor = {smooth_factor.item():.6e}')
    smooth_factor = symbol_measure(smooth_symbol[:, :, :, 0:3], measure_type=loss_measure)
    print(f'* Learned Measured Smoothing factor = {smooth_factor.item():.6e}')
    # plot
    theta_grid.plot(smooth_symbol_mod, title='Smoothing factor', num_levels=16)


if __name__ == "__main__":
    smoother_size = 3
    # centrosymmetric
    operator_symmetric = True
    smoother_symmetric = True
    # type of loss measure (L-inf: 1, L-2: 2)
    loss_measure_type = 1

    for test in [test0, test1]:
        try:
            test(operator_symmetric, smoother_symmetric, loss_measure_type, m_size=smoother_size)
            # print(f"{test.__name__} has been successfully run.\n")

        except Exception as e:
            # print(traceback.format_exc())
            print(f"\nError msg for {test.__name__}: {e}\n")

    plt.show()
