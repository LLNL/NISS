import sys
import matplotlib.pyplot as plt
import logging
from niss.config import NISSConfig
from niss.lfa.stencil import StencilSymbl2D  # noqa: E402
from niss.lfa.theta import Theta2D  # noqa: E402
from niss.utils.tensor import *  # noqa: E402


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


def main():
    # centrosymmetric
    a_symmetric = True
    m_symmetric = True
    # theta grid
    theta_grid = Theta2D(128, quadrant=torch.tensor([0, 1, 2, 3]))
    # stencil for A
    a_size = 3
    pattern_a = torch.ones([a_size, a_size])
    mat_a = -torch.ones([a_size, a_size]) / 3
    center_a = torch.tensor([a_size // 2, a_size // 2])
    mat_a[center_a[0], center_a[1]] = 8 / 3
    # zero out "strict-upper" part to test centrosymmetric
    if a_symmetric and a_size > 1:
        mat_a[centrosymmetric_strict_upper_coord(a_size)] = 0

    # Smoother 1 (point-wise):
    # stencil for M
    m_size = 1
    pattern_m = torch.ones([m_size, m_size])
    mat_m = torch.zeros([m_size, m_size])
    center_m = torch.tensor([m_size // 2, m_size // 2])
    mat_m[center_m[0], center_m[1]] = 1 / 3
    # zero out "strict-upper" part to test centrosymmetric
    if m_symmetric and m_size > 1:
        mat_m[centrosymmetric_strict_upper_coord(m_size)] = 0
    # smoothing operator
    smooth_operator = SmoothSymbl2D(pattern_a=pattern_a, center_a=center_a,
                                    pattern_m=pattern_m, center_m=center_m,
                                    mat_a=mat_a, centrosymmetric_a=a_symmetric,
                                    mat_m=mat_m, centrosymmetric_m=m_symmetric)
    smooth_operator.setup_theta(theta_grid)
    # smoother LFA
    smooth_symbol = smooth_operator.symbol()
    smooth_symbol_mod = torch.norm(smooth_symbol, dim=0)
    smooth_factor1 = torch.max(smooth_symbol_mod[:, :, 1:4]).item()
    logging.info(f'Smoothing factor is {smooth_factor1:.6f}')
    # plot
    if NISSConfig.plotting:
        theta_grid.plot(smooth_symbol_mod, title=f'Smoothing factor {smooth_factor1:.3f}', num_levels=32)

    # Smoother 2 (block):
    # stencil for M
    m_size = 3
    pattern_m = torch.ones([m_size, m_size])
    mat_m = torch.ones([m_size, m_size])
    center_m = torch.tensor([m_size // 2, m_size // 2])
    mat_m[center_m[0], center_m[1]] = 10
    mat_m = mat_m * (2 / 51)
    # zero out "strict-upper" part to test centrosymmetric
    if m_symmetric and m_size > 1:
        mat_m[centrosymmetric_strict_upper_coord(m_size)] = 0
    # smoother
    smooth_operator = SmoothSymbl2D(pattern_a=pattern_a, center_a=center_a,
                                    pattern_m=pattern_m, center_m=center_m,
                                    mat_a=mat_a, centrosymmetric_a=a_symmetric,
                                    mat_m=mat_m, centrosymmetric_m=m_symmetric)
    smooth_operator.setup_theta(theta_grid)
    # smoother LFA
    smooth_symbol = smooth_operator.symbol()
    smooth_symbol_mod = torch.norm(smooth_symbol, dim=0)
    smooth_factor2 = torch.max(smooth_symbol_mod[:, :, 1:4]).item()
    logging.info(f'Smoothing factor is {smooth_factor2:.6f}')
    # plot
    if NISSConfig.plotting:
        theta_grid.plot(smooth_symbol_mod, title=f'Smoothing factor {smooth_factor2:.3f}', num_levels=32)
        plt.show()

    ret = [smooth_factor1, smooth_factor2]
    return 0, ret


if __name__ == "__main__":
    NISSConfig.plotting = True
    err = main()[0]
    plt.show()
    sys.exit(err)
