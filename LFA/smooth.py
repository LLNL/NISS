import torch as torch
from LFA.stencil import StencilSymbl2D
from LFA.lfa import LFA2D


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

    def operator_symbol(self, theta_grid):
        symbl = self.stencil_a.symbol(theta_grid)
        return symbl

    def smoother_symbol(self, theta_grid):
        symbl = self.stencil_m.symbol(theta_grid)
        return symbl

    def symbol(self, theta_grid):
        symbl_op = self.operator_symbol(theta_grid)
        symbl_sm = self.smoother_symbol(theta_grid)
        return abs(1 - symbl_op * symbl_sm)


if __name__ == "__main__":
    # stencil for A
    A = -1 / 3 * torch.ones([3, 3])
    A[1, 1] = 8 / 3
    # stencil for M
    M = -1 / 3 * torch.zeros([3, 3])
    M[1, 1] = 1 / 3
    #
    center = torch.tensor([1, 1])
    # smoother
    smooth_operator = SmoothSymbl2D(A, center, M, center)
    # lfa
    num_theta = 128
    lfa = LFA2D(num_theta)
    # smoother LFA
    smooth_symbol = lfa.lfa(smooth_operator)
    smooth_factor = torch.max(smooth_symbol[:, :, 1:4])
    print('Smoothing factor is', smooth_factor.item())
    # plot
    lfa.plot(smooth_symbol, title='Smoothing factor', num_levels=10)

