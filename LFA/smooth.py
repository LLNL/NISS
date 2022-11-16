import torch as torch
from LFA.stencil import StencilSymbl2D
from LFA.theta import Theta2D


class SmoothSymbl2D:
    def __init__(self, mat_a, center_a, mat_m, center_m, nu_pre=1, nu_post=0):
        """
        :param mat_a:
        :param center_a:
        :param mat_m:
        :param center_m:
        :param nu_pre:
        :param nu_post:
        """
        self.stencil_m = StencilSymbl2D(mat_m, center_m)
        self.stencil_a = StencilSymbl2D(mat_a, center_a)
        self.nu_pre = nu_pre
        self.nu_post = nu_post

    def setup_theta(self, theta):
        self.stencil_m.setup_theta(theta)
        self.stencil_a.setup_theta(theta)

    def operator_symbol(self):
        symbl = self.stencil_a.symbol()
        return symbl

    def smoother_symbol(self):
        symbl = self.stencil_m.symbol()
        return symbl

    def symmetric(self):
        return self.stencil_a.symmetric() and self.stencil_m.symmetric()

    def symbol(self):
        operator_symbl = self.operator_symbol()
        smoother_symbl = self.smoother_symbol()
        symbol = torch.zeros_like(operator_symbl)
        if self.symmetric():
            symbol[0, :] = 1 - operator_symbl[0, :] * smoother_symbl[0, :]
        else:
            symbol[0, :] = 1 - operator_symbl[0, :] * smoother_symbl[0, :] + operator_symbl[1, :] * smoother_symbl[1, :]
            symbol[1, :] = operator_symbl[1, :] * smoother_symbl[0, :] + operator_symbl[0, :] * smoother_symbl[1, :]

        return symbol

    @staticmethod
    def smoothing_factor(symbol):
        smoothing_factor = torch.max(torch.norm(symbol, dim=0))
        return smoothing_factor


if __name__ == "__main__":
    # stencil for A
    A = -1 / 3 * torch.ones([3, 3])
    A[1, 1] = 8 / 3
    center_A = torch.tensor([1, 1])
    # stencil for M
    stencil_M_size = 3
    if stencil_M_size == 3:
        M = torch.zeros([3, 3])
        M[1, 1] = 1 / 3
        center_M = torch.tensor([1, 1])
    elif stencil_M_size == 1:
        M = 1 / 3 * torch.ones([1, 1])
        center_M = torch.tensor([0, 0])
    # smoother
    smooth_operator = SmoothSymbl2D(A, center_A, M, center_M)
    # theta grid
    theta_grid = Theta2D(128, quadrant=torch.tensor([0, 1, 2, 3]))
    smooth_operator.setup_theta(theta_grid)
    # smoother LFA
    smooth_symbol = smooth_operator.symbol()
    smooth_factor = smooth_operator.smoothing_factor(smooth_symbol[:, :, :, 1:4])
    print('Smoothing factor is', smooth_factor.item())
    # plot
    theta_grid.plot(torch.norm(smooth_symbol, dim=0), title='Smoothing factor', num_levels=10)
