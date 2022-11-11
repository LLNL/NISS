import torch
from torch import cos, sin, sqrt
from LFA.lfa import LFA2D


class StencilSymbl2D:
    def __init__(self, mat, center, symmetric=True):
        """
        :param mat: stencil matrix
        :param center: stencil center: (dim0, dim1), i.e., (y, x)
        :param symmetric: if the stencil is symmetric
        It stores stencils in k-by-3 array
        """
        self.symmetric = symmetric
        self.stencil = torch.zeros(mat.size(0) * mat.size(1), 3)
        k = 0
        for i0 in range(0, mat.size(0)):
            for i1 in range(0, mat.size(1)):
                self.stencil[k, 0] = i0 - center[0]
                self.stencil[k, 1] = i1 - center[1]
                self.stencil[k, 2] = mat[i0, i1]
                k += 1

    def symbol(self, theta_grid):
        symbl_r = torch.zeros_like(theta_grid[:, :, 0])
        symbl_i = torch.zeros_like(theta_grid[:, :, 0])
        for k in range(0, self.stencil.size(0)):
            i0 = self.stencil[k, 0]
            i1 = self.stencil[k, 1]
            sv = self.stencil[k, 2]
            x = i0 * theta_grid[:, :, 0] + i1 * theta_grid[:, :, 1]
            symbl_r += sv * cos(x)
            symbl_i += sv * sin(x)

        symbl = sqrt(symbl_r * symbl_r + symbl_i * symbl_i)
        return symbl


if __name__ == "__main__":
    A = -1 / 3 * torch.ones([3, 3])
    A[1, 1] = 8 / 3
    stencil_A = StencilSymbl2D(A, torch.tensor([1, 1]))
    # symbol_A = stencil_A.symbol(torch.tensor(torch.pi / 4), torch.tensor(torch.pi / 4))
    # print('(pi/4, pi/4): ', symbol_A.item())
    # lfa
    num_theta = 128
    lfa = LFA2D(num_theta)
    # smoother LFA
    symbol_A = lfa.lfa(stencil_A)
    print('symbol A: ', 'max: ', torch.max(symbol_A).item(), 'min: ', torch.min(symbol_A).item())
    # plot
    lfa.plot(symbol_A, title='Operator A', num_levels=20)
