import torch
from LFA.stencil import StencilSymbl2D


class SmoothSymbl2D:
    def __init__(self, mat_a, center_a, mat_m, center_m, nu_pre=1, nu_post=0):
        """
        """
        self.stencil_m = StencilSymbl2D(mat_m, center_m)
        self.stencil_a = StencilSymbl2D(mat_a, center_a)
        self.nu_pre = nu_pre
        self.nu_post = nu_post

    def operator_symbl(self, theta0, theta1):
        symbl = self.stencil_a.symbol(theta0, theta1)
        return symbl

    def smoother_symbl(self, theta0, theta1):
        symbl = self.stencil_m.symbol(theta0, theta1)
        return symbl

    def smoothing_symbl(self, theta0, theta1):
        symbl_op = self.operator_symbl(theta0, theta1)
        symbl_sm = self.smoother_symbl(theta0, theta1)
        return abs(1 - symbl_op * symbl_sm)


if __name__ == "__main__":
    # A
    A = -1 / 3 * torch.ones([3, 3])
    A[1, 1] = 8 / 3
    # M
    M = -1 / 3 * torch.zeros([3, 3])
    M[1, 1] = 1 / 3
    #
    center = torch.tensor([1, 1])
    smooth_op = SmoothSymbl2D(A, center, M, center)
