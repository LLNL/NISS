import sys
import numpy
import scipy.sparse as sparse
import torch
import matplotlib.pyplot as plt
from niss.config import NISSConfig
from niss.lfa.stencil import StencilSymbl2D


class StructMatrix:
    def __init__(self, stencil, grid_dim):
        stride = torch.cat((torch.tensor([1]), grid_dim.flip(0).cumprod(0)))
        n = stride[-1].item()
        stride = stride[:-1].flip(0)
        yy = numpy.arange(0, grid_dim[0])
        xx = numpy.arange(0, grid_dim[1])
        x_coo, y_coo = numpy.meshgrid(xx, yy, indexing='xy', sparse=True)
        x_coo = torch.from_numpy(x_coo)
        y_coo = torch.from_numpy(y_coo)
        row = (y_coo * grid_dim[1] + x_coo).unsqueeze(-1).repeat_interleave(stencil.stencil_shape_len, axis=2)
        y_coo = y_coo.unsqueeze(-1).repeat_interleave(stencil.stencil_shape_len, axis=2) + stencil.stencil_shape[0, :]
        x_coo = x_coo.unsqueeze(-1).repeat_interleave(stencil.stencil_shape_len, axis=2) + stencil.stencil_shape[1, :]
        y_coo[(y_coo < 0) | (y_coo >= grid_dim[0])] = torch.nan
        x_coo[(x_coo < 0) | (x_coo >= grid_dim[1])] = torch.nan
        col = y_coo * stride[0] + x_coo
        val = torch.mm(stencil.stencil_value.unsqueeze(0), stencil.sym_extend) + torch.zeros_like(row)
        row = row.flatten()
        col = col.flatten()
        val = val.flatten()
        flag = torch.isnan(col)
        row, col, val = row[~flag], col[~flag], val[~flag]
        # create coo
        coo = sparse.coo_matrix((val, (row, col)), shape=(n, n))
        self.csr = coo.tocsr()


def main(ny=2, nx=3):
    mat_a = torch.zeros([3, 3])
    mat_a[[0, 2],
          [1, 1]] = -1
    mat_a[[1, 1],
          [0, 2]] = -0.1
    mat_a[1, 1] = 2.2
    pat_a = torch.zeros([3, 3])
    pat_a[[0, 1, 1, 1, 2],
          [1, 0, 1, 2, 1]] = 1
    stencil_a = StencilSymbl2D(pattern=pat_a, center=torch.tensor([1, 1]), vmatrix=mat_a, centrosymmetric=True)
    csr_a = StructMatrix(stencil_a, torch.tensor([ny, nx]))

    return 0, csr_a


if __name__ == "__main__":
    NISSConfig.plotting = True
    err = main()[0]
    csr = main()[1]
    print(csr.csr.toarray())
    plt.show()
    sys.exit(err)
