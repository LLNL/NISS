import torch
import numpy as np
from torch import cos, sin
from LFA.lfa import LFA2D


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

    def symmetric(self):
        return self.symmetry

    def symbol(self, theta):
        symbol = torch.zeros(2, theta.size(0), theta.size(1), theta.size(3))
        x = torch.sum(theta * self.stencil[:, None, None, 0:2, None], dim=3)
        symbol[0, :] = torch.sum(self.stencil[:, 2, None, None, None] * cos(x), dim=0)
        if not self.symmetric():
            symbol[1, :] = torch.sum(self.stencil[:, 2, None, None, None] * sin(x), dim=0)

        return symbol


if __name__ == "__main__":
    A = -1 / 3 * torch.ones([3, 3])
    A[1, 1] = 8 / 3
    stencil_A = StencilSymbl2D(A, torch.tensor([1, 1]))
    symbol_A = stencil_A.symbol(torch.tensor([torch.pi / 4, torch.pi / 4])[None, None, :, None])
    print('(pi/4, pi/4): ', symbol_A[0].item())
    # lfa
    num_theta = 128
    lfa = LFA2D(num_theta)
    # smoother LFA
    stencil_symbol = lfa.lfa(stencil_A)[0, :]
    print('symbol A: ', 'max: ', torch.max(stencil_symbol).item(), 'min: ', torch.min(stencil_symbol).item())
    # plot
    lfa.plot(stencil_symbol, title='Operator A', num_levels=20)
