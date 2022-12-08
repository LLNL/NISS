import sys
import torch.nn as nn
import matplotlib.pyplot as plt
import logging

from niss.config import NISSConfig
from niss.utils.tensor import *  # noqa: E402
from niss.lfa.theta import Theta2D  # noqa: E402
from niss.lfa.stencil import StencilSymbl2D  # noqa: E402
from niss.lfa.smooth import SmoothSymbl2D  # noqa: E402
from niss.ml.trainer import NNTrainer  # noqa: E402


# np.random.seed(0)
torch.set_printoptions(precision=5)
# precision
# torch.set_default_dtype(torch.float64)


# 0. Return smoother stencil values
def smoother_optimal(smoother_stencil: StencilSymbl2D, operator_stencil=None,
                     trainer=None, symbol_measure=None):
    return smoother_stencil.stencil_value


class Net(nn.Module):
    def __init__(self, size):
        super(Net, self).__init__()
        self.fc = nn.Linear(size, 1, bias=False)
        torch.nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        x = self.fc(x)
        return x


# 1. Use NN to learn M
def smoother_nn1(trainer, smoother_stencil: StencilSymbl2D, operator_stencil: StencilSymbl2D = None,
                 symbol_measure=None):
    # For reproducibility
    torch.manual_seed(0)
    # Clear smoother stencil's values
    smoother_stencil.stencil_value = None
    # theta grid only for high freq
    theta_grid = Theta2D(128, quadrant=torch.tensor([1, 2, 3]))
    # setup theta grid for only high freq.
    operator_stencil.setup_theta(theta_grid)
    smoother_stencil.setup_theta(theta_grid)

    # create a NN to learn the stencil weights of M
    net = Net(size=smoother_stencil.stencil_value_len)
    # print(net)

    # define loss function
    class SmootherLoss(nn.Module):
        def __init__(self, operator_symbol_):
            super(SmootherLoss, self).__init__()
            self.operator_symbol = operator_symbol_

        def forward(self, smoother_symbol):
            smoother_symbol = smoother_symbol.squeeze(-1)
            symbol = complex_multiply(self.operator_symbol, smoother_symbol)
            if symbol.size(0) == 1:
                symbol = 1 - symbol
            else:
                symbol = torch.tensor([[1], [0]]) - symbol
            return symbol_measure(symbol)

    operator_symbol = operator_stencil.symbol().flatten(start_dim=1)
    criterion = SmootherLoss(operator_symbol)
    # NN Input
    nn_input = smoother_stencil.cos_sin_theta_stencil.flatten(start_dim=1, end_dim=3)
    if smoother_stencil.is_symmetric():
        nn_input = torch.matmul(nn_input, smoother_stencil.sym_extend.transpose(0, 1))
    # Training NN
    trainer.train(net, criterion, nn_input)
    # Get smoother weights from NN
    linear_layer = net.fc
    logging.info(f'* Final Measured Smoothing factor = {criterion(net(nn_input)).item():.6e} ')
    # logging.info(f'* Smoother Weights')
    # logging.info(f'{np.array2string(linear_layer.weight.detach().numpy())}')

    return linear_layer.weight


# 2. Use NN to learn M (Preferred. Use SmoothSymbl2D class)
def smoother_nn2(trainer, smoother_stencil: StencilSymbl2D, operator_stencil: StencilSymbl2D = None,
                 symbol_measure=None):
    # For reproducibility
    torch.manual_seed(0)
    # Clear smoother stencil's values
    smoother_stencil.stencil_value = None
    # Create smoother class
    smooth_operator = SmoothSymbl2D(stencil_a=operator_stencil, stencil_m=smoother_stencil)
    # theta grid only for high freq
    theta_grid = Theta2D(128, quadrant=torch.tensor([1, 2, 3]))
    smooth_operator.setup_theta(theta_grid)

    # create a NN to learn the stencil weights of M
    net = Net(size=smooth_operator.smoother_stencil.stencil_value_len)
    # print(net)

    # define loss function
    class SmootherLoss(nn.Module):
        def __init__(self, smoother_operator):
            super(SmootherLoss, self).__init__()
            self.smoother_operator = smoother_operator

        def forward(self, smoother_weights):
            self.smoother_operator.smoother_stencil.set_values(smoother_weights.squeeze())
            symbol = self.smoother_operator.symbol()
            return symbol_measure(symbol)

    criterion = SmootherLoss(smooth_operator)
    # NN Input
    nn_input = torch.eye(smooth_operator.smoother_stencil.stencil_value_len)
    # Training NN
    trainer.train(net, criterion, nn_input)
    # Get smoother weights from NN
    linear_layer = net.fc
    logging.info(f'* Final Measured Smoothing factor = {criterion(net(nn_input)).item():.6e} ')
    # logging.info(f'* Smoother Weights')
    # logging.info(f'{np.array2string(linear_layer.weight.detach().numpy().squeeze())}')

    return linear_layer.weight


def main(smoother_size=3, num_steps=8000):
    # centrosymmetric
    operator_symmetric = True
    smoother_symmetric = True
    # size of the stencils
    operator_size = 3
    # stencil for A
    operator_pattern = torch.ones(operator_size, operator_size)
    operator_center = torch.tensor([operator_size // 2, operator_size // 2])
    operator_matrix = -1 / 3 * torch.ones([operator_size, operator_size])
    operator_matrix[operator_center[0], operator_center[1]] = 8 / 3
    # zero out "strict-upper" part to test centrosymmetric
    if operator_symmetric and operator_size > 1:
        operator_matrix[centrosymmetric_strict_upper_coord(operator_size)] = 0
    operator_stencil = StencilSymbl2D(operator_pattern, operator_center, vmatrix=operator_matrix,
                                      centrosymmetric=operator_symmetric)
    # stencil for M
    smoother_pattern = torch.ones(smoother_size, smoother_size)
    smoother_center = torch.tensor([smoother_size // 2, smoother_size // 2])
    # "optimal" smoother
    if smoother_size != 3:
        optimal_smoother_matrix = torch.zeros(smoother_size, smoother_size)
        optimal_smoother_matrix[smoother_center[0], smoother_center[1]] = 1 / 3
    else:
        optimal_smoother_matrix = torch.ones(smoother_size, smoother_size)
        optimal_smoother_matrix[smoother_center[0], smoother_center[1]] = 10
        optimal_smoother_matrix = optimal_smoother_matrix * (2 / 51)

    smoother_stencil = StencilSymbl2D(smoother_pattern, smoother_center, vmatrix=optimal_smoother_matrix,
                                      centrosymmetric=smoother_symmetric)
    # NN
    # optimizer_type = torch.optim.SGD
    # learning_rate = 0.0005
    # optimizer_type = torch.optim.LBFGS
    # learning_rate = 0.1
    # type of loss measure (L-inf: 1, L-2: 2)

    def symbol_measure_linf(symbol): return torch.max(torch.norm(symbol, dim=0))
    # def symbol_measure_l2(symbol): return torch.sqrt(torch.mean(torch.square(symbol)))
    # def symbol_measure(symbol): return torch.mean(torch.square(symbol))

    optimizer_type = torch.optim.Adam
    learning_rate = 0.0004
    num_prints = 20
    symbol_measure = symbol_measure_linf
    trainer = NNTrainer(optimizer=optimizer_type, learning_rate=learning_rate, num_steps=num_steps,
                        num_prints=num_prints)
    err = 0
    ret = []
    theta_grid = Theta2D(128, quadrant=torch.tensor([1, 2, 3]))
    for test in [smoother_optimal, smoother_nn1, smoother_nn2]:
        try:
            logging.info(f'= = = = = = = = = {test.__name__} = = = = = = = = = = =')
            # get smoother matrix from test
            smoother_weights = test(trainer=trainer, smoother_stencil=smoother_stencil,
                                    operator_stencil=operator_stencil,
                                    symbol_measure=symbol_measure)
            # smoother matrix
            if smoother_symmetric:
                smoother_matrix = torch.zeros(smoother_size, smoother_size)
                smoother_matrix[centrosymmetric_lower_coord(smoother_size)] = \
                    smoother_weights.squeeze().detach().clone()
                centrosymmetrize_upper(smoother_matrix)
            else:
                smoother_matrix = smoother_weights.detach().clone().reshape(smoother_size, smoother_size)
            logging.info(f'* Smoother stencil matrix')
            logging.info(f'{np.array2string(smoother_matrix.detach().numpy(), precision=6)}')
            # get smooth factor using smoother class
            smooth_operator = SmoothSymbl2D(pattern_a=operator_pattern, center_a=operator_center,
                                            pattern_m=smoother_pattern, center_m=smoother_center,
                                            mat_a=operator_matrix, centrosymmetric_a=operator_symmetric,
                                            mat_m=smoother_matrix, centrosymmetric_m=smoother_symmetric)
            smooth_operator.setup_theta(theta_grid)
            smooth_symbol = smooth_operator.symbol()
            smooth_symbol_mod = torch.norm(smooth_symbol, dim=0)
            inf_smooth_factor = torch.max(smooth_symbol_mod[..., -3:]).item()
            smooth_factor = symbol_measure(smooth_symbol[..., -3:]).item()
            ret.append(inf_smooth_factor)
            logging.info(f'* Inf-norm Smoothing factor = {inf_smooth_factor:.8e}')
            logging.info(f'* Measured Smoothing factor = {smooth_factor:.8e}')
            # plot
            if NISSConfig.plotting:
                theta_grid.plot(smooth_symbol_mod, title=f'Smoothing factor {inf_smooth_factor:.2e}',
                                num_levels=NISSConfig.contour_num_levels)
        except Exception as e:
            # print(traceback.format_exc())
            err += 1
            logging.error(f"\nError msg for {test.__name__}: {e}\n")

    return err, ret


if __name__ == "__main__":
    NISSConfig.plotting = True
    ierr = main(smoother_size=3, num_steps=2000)[0]
    plt.show()
    sys.exit(ierr)
