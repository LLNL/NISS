import torch
import numpy as np


# multiply two complex tensors
# x and y should have the same shape
# the first dimension tells if they are real (dim = 1) or complex (dim = 2)
# x[0,:] and y[0,:] are the real parts and x[1, :] and y[1, :] are the imaginary parts
def complex_multiply(x, y):
    if x.size(0) == 1 and y.size(0) == 1:
        return x * y
    else:
        # real part
        z_r = x[0, :] * y[0, :]
        if x.size(0) > 1 and y.size(0) > 1:
            z_r -= x[1, :] * y[1, :]
        # imaginary part
        if x.size(0) > 1 and y.size(0) == 1:
            z_i = x[1, :] * y[0, :]
        elif x.size(0) == 1 and y.size(0) > 1:
            z_i = x[0, :] * y[1, :]
        else:
            z_i = x[1, :] * y[0, :] + x[0, :] * y[1, :]
        # complex
        return torch.stack([z_r, z_i])


def centrosymmetric_strict_upper_coord(n: int):
    if n <= 1:
        return [], []
    else:
        return np.unravel_index(range((n * n + 1) // 2, n * n), (n, n), 'C')


def centrosymmetric_strict_lower_coord(n: int):
    return np.unravel_index(range(0, (n * n) // 2), (n, n), 'C')


def centrosymmetric_upper_coord(n: int):
    return np.unravel_index(range((n * n) // 2, n * n), (n, n), 'C')


def centrosymmetric_lower_coord(n: int):
    return np.unravel_index(range(0, (n * n + 1) // 2), (n, n), 'C')


def centrosymmetrize_upper(mat):
    n = mat.size(0)
    assert(n == mat.size(1))
    lower_row, lower_col = centrosymmetric_strict_lower_coord(n)
    mat[centrosymmetric_strict_upper_coord(n)] = mat[(np.flip(lower_row).copy(), np.flip(lower_col).copy())]
