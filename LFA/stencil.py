import torch
import numpy as np
from LFA.theta import Theta2D


class StencilSymbl2D:
    def __init__(self, mat, center):
        """
        :param mat: stencil matrix
        :param center[2]: stencil center in (dim0, dim1), i.e., (y, x)
        """
        # stores stencils in k-by-3 array
        ind2d = np.unravel_index(range(mat.size(0) * mat.size(1)), mat.shape, 'C')
        self.stencil = torch.zeros(mat.size(0) * mat.size(1), 3)
        self.stencil[:, 0] = torch.tensor(ind2d[0]) - center[0]
        self.stencil[:, 1] = torch.tensor(ind2d[1]) - center[1]
        self.stencil[:, 2] = mat.flatten()
        # test if stencil is symmetric
        ind2d_sym = (mat.size(0) - 1 - ind2d[0], mat.size(1) - 1 - ind2d[1])
        self.symmetry = torch.all(mat[ind2d] == mat[ind2d_sym])
        # the inner product of lfa theta_quad and stencil kappa
        self.theta_stencil = None
        # cos and sin of theta_stencil
        self.cos_theta_stencil = None
        self.sin_theta_stencil = None

    def symmetric(self):
        return self.symmetry

    def setup_theta(self, theta):
        self.theta_stencil = torch.sum(theta.theta_quad * self.stencil[:, None, None, 0:2, None], dim=3)
        self.cos_theta_stencil = torch.cos(self.theta_stencil)
        if not self.symmetric():
            self.sin_theta_stencil = torch.sin(self.theta_stencil)

    def symbol(self):
        symbol = torch.zeros((2,) + self.theta_stencil.shape[1:])
        symbol[0, :] = torch.sum(self.stencil[:, 2, None, None, None] * self.cos_theta_stencil, dim=0)
        if not self.symmetric():
            symbol[1, :] = torch.sum(self.stencil[:, 2, None, None, None] * self.sin_theta_stencil, dim=0)

        return symbol


if __name__ == "__main__":
    # stencil for A
    A = -1 / 3 * torch.ones([3, 3])
    A[1, 1] = 8 / 3
    stencil_A = StencilSymbl2D(A, torch.tensor([1, 1]))
    # (pi/4, pi/4) symbol
    theta_grid = Theta2D(1, start=torch.pi / 4, end=torch.pi / 4, quadrant=torch.tensor([0]))
    stencil_A.setup_theta(theta_grid)
    symbol_A = stencil_A.symbol()[0, :]
    print("LFA of (%f, %f) is %f" % (theta_grid.theta[0], theta_grid.theta[0], symbol_A.item()))
    # theta grid
    theta_grid = Theta2D(128)
    stencil_A.setup_theta(theta_grid)
    # stencil LFA
    stencil_symbol = stencil_A.symbol()[0, :]
    print('symbol A: ', 'max: ', torch.max(stencil_symbol).item(), 'min: ', torch.min(stencil_symbol).item())
    # plot
    theta_grid.plot(stencil_symbol, title='Operator A', num_levels=20)
