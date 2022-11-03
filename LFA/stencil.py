import torch
from numpy import pi
from torch import cos, sin, sqrt


class StencilSymbl2D:
    def __init__(self, mat, center):
        """
        :param mat: stencil matrix
        :param center: stencil center: (dim0, dim1), i.e., (y, x)
        """
        self.stencil = torch.zeros(mat.size(0) * mat.size(1), 3)
        k = 0
        for i0 in range(0, mat.size(0)):
            for i1 in range(0, mat.size(1)):
                self.stencil[k, 0] = i0 - center[0]
                self.stencil[k, 1] = i1 - center[1]
                self.stencil[k, 2] = mat[i0, i1]
                k += 1

    def symbol(self, theta0, theta1):
        symbl = torch.zeros(2)
        for k in range(0, self.stencil.size(0)):
            i0 = self.stencil[k, 0]
            i1 = self.stencil[k, 1]
            sv = self.stencil[k, 2]
            x = i0 * theta0 + i1 * theta1
            symbl[0] += sv * cos(x)
            symbl[1] += sv * sin(x)

        symbl = sqrt(symbl[0] * symbl[0] + symbl[1] * symbl[1])
        return symbl


if __name__ == "__main__":
    A = -1 / 3 * torch.ones([3, 3])
    A[1, 1] = 8 / 3
    stencil_A = StencilSymbl2D(A, torch.tensor([1, 1]))
    sym = stencil_A.symbol(pi/4, pi/4)
    print(sym)
