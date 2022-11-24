# import numpy as np
# import torch
from LFA.stencil import StencilSymbl2D
from LFA.theta import Theta2D
from UTILS.utils import *
import matplotlib.pyplot as plt


class SmoothSymbl2D:
    smoother_stencil: StencilSymbl2D
    operator_stencil: StencilSymbl2D
    nu_pre: int
    nu_post: int

    def __init__(self, pattern_a=None, center_a=None, pattern_m=None, center_m=None,
                 mat_a=None, centrosymmetric_a=False, mat_m=None, centrosymmetric_m=False,
                 stencil_a=None, stencil_m=None, nu_pre=1, nu_post=0):
        """
        :param pattern_a:
        :param center_a:
        :param pattern_m:
        :param center_m:
        :param mat_a:
        :param centrosymmetric_a:
        :param mat_m:
        :param centrosymmetric_m:
        :param stencil_a:
        :param stencil_m:
        :param nu_pre:
        :param nu_post:
        """
        if stencil_m is None:
            self.smoother_stencil = StencilSymbl2D(pattern_m, center_m, mat_m, centrosymmetric_m)
        else:
            self.smoother_stencil = stencil_m

        if stencil_a is None:
            self.operator_stencil = StencilSymbl2D(pattern_a, center_a, mat_a, centrosymmetric_a)
        else:
            self.operator_stencil = stencil_a

        self.nu_pre = nu_pre
        self.nu_post = nu_post

    def setup_theta(self, theta):
        self.smoother_stencil.setup_theta(theta)
        self.operator_stencil.setup_theta(theta)

    def operator_symbol(self):
        symbol = self.operator_stencil.symbol()
        return symbol

    def smoother_symbol(self):
        symbol = self.smoother_stencil.symbol()
        return symbol

    def is_symmetric(self):
        return self.operator_stencil.is_symmetric() and self.smoother_stencil.is_symmetric()

    def symbol(self):
        symbol = complex_multiply(self.operator_symbol(), self.smoother_symbol())
        if symbol.size(0) == 1:
            symbol = 1 - symbol
        else:
            symbol = torch.tensor([1, 0])[:, None, None, None] - symbol
        return symbol


if __name__ == "__main__":
    # centrosymmetric
    A_symmetric = True
    M_symmetric = True
    # theta grid
    theta_grid = Theta2D(128, quadrant=torch.tensor([0, 1, 2, 3]))
    # stencil for A
    A_size = 3
    pattern_A = torch.ones([A_size, A_size])
    A = -torch.ones([A_size, A_size]) / 3
    center_A = torch.tensor([A_size // 2, A_size // 2])
    A[center_A[0], center_A[1]] = 8 / 3
    # zero out "strict-upper" part to test centrosymmetric
    if A_symmetric and A_size > 1:
        A[centrosymmetric_strict_upper_coord(A_size)] = 0

    # Smoother 1:
    # stencil for M
    M_size = 1
    pattern_M = torch.ones([M_size, M_size])
    M = torch.zeros([M_size, M_size])
    center_M = torch.tensor([M_size // 2, M_size // 2])
    M[center_M[0], center_M[1]] = 1 / 3
    # zero out "strict-upper" part to test centrosymmetric
    if M_symmetric and M_size > 1:
        M[centrosymmetric_strict_upper_coord(M_size)] = 0
    # smoothing operator
    smooth_operator = SmoothSymbl2D(pattern_a=pattern_A, center_a=center_A,
                                    pattern_m=pattern_M, center_m=center_M,
                                    mat_a=A, centrosymmetric_a=A_symmetric,
                                    mat_m=M, centrosymmetric_m=M_symmetric)
    smooth_operator.setup_theta(theta_grid)
    # smoother LFA
    smooth_symbol = smooth_operator.symbol()
    smooth_symbol_mod = torch.norm(smooth_symbol, dim=0)
    smooth_factor = torch.max(smooth_symbol_mod[:, :, 1:4])
    print('Smoothing factor is', smooth_factor.item())
    # plot
    theta_grid.plot(smooth_symbol_mod, title='Smoothing factor', num_levels=32)

    # Smoother 2:
    # stencil for M
    M_size = 3
    pattern_M = torch.ones([M_size, M_size])
    M = torch.ones([M_size, M_size])
    center_M = torch.tensor([M_size // 2, M_size // 2])
    M[center_M[0], center_M[1]] = 10
    M = M * (2 / 51)
    # zero out "strict-upper" part to test centrosymmetric
    if M_symmetric and M_size > 1:
        M[centrosymmetric_strict_upper_coord(M_size)] = 0
    # smoother
    smooth_operator = SmoothSymbl2D(pattern_a=pattern_A, center_a=center_A,
                                    pattern_m=pattern_M, center_m=center_M,
                                    mat_a=A, centrosymmetric_a=A_symmetric,
                                    mat_m=M, centrosymmetric_m=M_symmetric)
    smooth_operator.setup_theta(theta_grid)
    # smoother LFA
    smooth_symbol = smooth_operator.symbol()
    smooth_symbol_mod = torch.norm(smooth_symbol, dim=0)
    smooth_factor = torch.max(smooth_symbol_mod[:, :, 1:4])
    print('Smoothing factor is', smooth_factor.item())
    # plot
    theta_grid.plot(smooth_symbol_mod, title='Smoothing factor', num_levels=32)

    plt.show()
