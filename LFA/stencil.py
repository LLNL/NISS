import torch
# import numpy as np
import matplotlib.pyplot as plt
from LFA.theta import Theta2D


class StencilSymbl2D:
    def __init__(self, pattern, center, mat=None, centrosymmetric=False):
        """
        :param pattern: stencil pattern: boolean
        :param center[2]: stencil center in (dim0, dim1), i.e., (y, x)
        :param mat: stencil values
        :param centrosymmetric: if the stencil is symmetric (mat and pattern are centrosymmetric)
        """
        # size in x and y
        nx = pattern.size(1)
        ny = pattern.size(0)
        # stencil is symmetric if mat and pattern are centrosymmetric
        self.symmetric = centrosymmetric
        # k: number of stencil entries; ks: in the lower-symmetric part
        k = 0
        ks = 0
        for i in range(ny):
            for j in range(nx):
                if pattern[i, j]:
                    k += 1
                    if not self.symmetric or i * nx + j <= (nx * ny - 1) / 2:
                        ks += 1
        # store the sizes
        self.stencil_shape_len = k
        self.stencil_value_len = ks
        # stores stencils in arrays
        self.stencil_shape = torch.zeros(2, self.stencil_shape_len)
        if mat is None:
            self.stencil_value = None
        else:
            self.stencil_value = torch.zeros(1, self.stencil_value_len)
        # fill stencil data in the arrays
        k = 0
        ks = 0
        for i in range(ny):
            for j in range(nx):
                if pattern[i, j]:
                    self.stencil_shape[0, k] = i - center[0]
                    self.stencil_shape[1, k] = j - center[1]
                    k += 1
                    if not self.symmetric or i * nx + j <= (nx * ny - 1) / 2:
                        if mat is not None:
                            self.stencil_value[0, ks] = mat[i, j]
                        ks += 1
        # for symmetric stencil, construct a matrix that extends to full stencil
        if self.symmetric:
            self.sym_extend = torch.zeros(self.stencil_value_len, self.stencil_shape_len)
            for i in range(self.stencil_shape_len):
                if i < self.stencil_value_len:
                    self.sym_extend[i, i] = 1
                else:
                    self.sym_extend[2 * (self.stencil_value_len - 1) - i, i] = 1
        else:
            self.sym_extend = None
        # the inner product of lfa theta_quad and stencil kappa
        self.theta_stencil = None
        # cos and sin of theta_stencil
        self.cos_sin_theta_stencil = None

    def is_symmetric(self):
        return self.symmetric

    def setup_theta(self, theta):
        # (nx, ny, num_quad, stencil_shape_len)
        self.theta_stencil = torch.sum(theta.theta_quad.unsqueeze(-1) * self.stencil_shape[None, None, :, None, :],
                                       dim=2)
        if self.symmetric:
            # (1, nx, ny, n_quad, stencil_shape_len)
            self.cos_sin_theta_stencil = torch.cos(self.theta_stencil).unsqueeze(0)
        else:
            # (2, nx, ny, n_quad, stencil_shape_len)
            self.cos_sin_theta_stencil = torch.stack([torch.cos(self.theta_stencil), torch.sin(self.theta_stencil)])

    def symbol(self):
        if self.symmetric:
            value = torch.mm(self.stencil_value, self.sym_extend)
        else:
            value = self.stencil_value
        symbol = torch.sum(value * self.cos_sin_theta_stencil, dim=-1)

        return symbol


if __name__ == "__main__":
    # stencil for A
    A = -1 / 3 * torch.ones([3, 3])
    A[1, 1] = 8 / 3
    stencil_A = StencilSymbl2D(torch.ones(3, 3), torch.tensor([1, 1]), A, centrosymmetric=True)
    # (pi/4, pi/4) symbol
    theta_grid = Theta2D(1, start=torch.pi / 4, end=torch.pi / 4, quadrant=torch.tensor([0]))
    stencil_A.setup_theta(theta_grid)
    stencil_symbol = stencil_A.symbol()
    stencil_symbol_mod = torch.norm(stencil_symbol, dim=0)
    print("LFA of (%f, %f) is %f" % (theta_grid.theta[0], theta_grid.theta[0], stencil_symbol_mod.item()))

    # theta grid
    theta_grid = Theta2D(128)
    stencil_A.setup_theta(theta_grid)
    # stencil LFA
    stencil_symbol = stencil_A.symbol()
    stencil_symbol_mod = torch.norm(stencil_symbol, dim=0)
    print('symbol A: ', 'max: ', torch.max(stencil_symbol_mod).item(), 'min: ', torch.min(stencil_symbol_mod).item())
    # plot
    theta_grid.plot(stencil_symbol_mod, title='Operator A', num_levels=20)
    plt.show()
