import torch
from numpy import pi
from LFA.stencil import StencilSymbl2D
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


class SmoothSymbl2D:
    def __init__(self, mat_a, center_a, mat_m, center_m, n_theta, nu_pre=1, nu_post=0):
        """
        :param mat_a:
        :param center_a:
        :param mat_m:
        :param center_m:
        :param n_theta: number of points in theta grid
        :param nu_pre:
        :param nu_post:
        """
        self.stencil_m = StencilSymbl2D(mat_m, center_m)
        self.stencil_a = StencilSymbl2D(mat_a, center_a)
        self.n_theta = n_theta
        h_theta = pi / n_theta
        self.theta = torch.linspace(-pi / 2 + h_theta / 2, pi / 2 - h_theta / 2, n_theta)
        self.nu_pre = nu_pre
        self.nu_post = nu_post

    def operator_symbol(self, theta0, theta1):
        symbl = self.stencil_a.symbol(theta0, theta1)
        return symbl

    def smoother_symbol(self, theta0, theta1):
        symbl = self.stencil_m.symbol(theta0, theta1)
        return symbl

    def smoothing_symbol(self, theta0, theta1):
        symbl_op = self.operator_symbol(theta0, theta1)
        symbl_sm = self.smoother_symbol(theta0, theta1)
        return abs(1 - symbl_op * symbl_sm)

    def lfa(self):
        """
        :return: the max (over theta) smoothing factor
        """
        smooth_sym = torch.zeros(self.n_theta, self.n_theta, 4)
        theta1 = self.theta.reshape(-1, 1).repeat(1, self.n_theta)
        theta2 = self.theta.reshape(1, -1).repeat(self.n_theta, 1)
        smooth_sym[:, :, 0] = self.smoothing_symbol(theta1, theta2)
        smooth_sym[:, :, 1] = self.smoothing_symbol(theta1 + pi, theta2)
        smooth_sym[:, :, 2] = self.smoothing_symbol(theta1, theta2 + pi)
        smooth_sym[:, :, 3] = self.smoothing_symbol(theta1 + pi, theta2 + pi)

        smooth_fac = torch.max(smooth_sym[:, :, 1:4])

        return smooth_fac, smooth_sym


if __name__ == "__main__":
    # A
    A = -1 / 3 * torch.ones([3, 3])
    A[1, 1] = 8 / 3
    # M
    M = -1 / 3 * torch.zeros([3, 3])
    M[1, 1] = 1 / 3
    #
    center = torch.tensor([1, 1])
    n_theta = 128
    smooth_operator = SmoothSymbl2D(A, center, M, center, n_theta)
    smooth_factor, smooth_symbol = smooth_operator.lfa()
    print('Smoothing factor is', smooth_factor.item())

    # plot
    fig, ax = plt.subplots(1, 1)
    theta_plot = np.zeros(2 * n_theta)
    theta_plot[0:n_theta] = smooth_operator.theta
    theta_plot[n_theta: 2 * n_theta] = smooth_operator.theta + pi
    smooth_symbol_plot = np.zeros((2 * n_theta, 2 * n_theta))
    smooth_symbol_plot[0:n_theta, 0:n_theta] = smooth_symbol[:, :, 0]
    smooth_symbol_plot[n_theta:2 * n_theta, 0:n_theta] = smooth_symbol[:, :, 1]
    smooth_symbol_plot[0:n_theta, n_theta:2 * n_theta] = smooth_symbol[:, :, 2]
    smooth_symbol_plot[n_theta:2 * n_theta, n_theta:2 * n_theta] = smooth_symbol[:, :, 3]
    cp = plt.contour(theta_plot, theta_plot, smooth_symbol_plot, levels=10)
    plt.clabel(cp, fontsize=8, colors='black')
    norm = colors.Normalize(vmin=cp.cvalues.min(), vmax=cp.cvalues.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cp.cmap)
    sm.set_array([])
    fig.colorbar(sm, ticks=cp.levels)
    plt.plot([np.min(theta_plot), np.max(theta_plot)], [0.5*pi, 0.5*pi], color='red')
    plt.plot([0.5 * pi, 0.5 * pi], [np.min(theta_plot), np.max(theta_plot)], color='red')
    ax.set_title(r'Smoothing factor', fontfamily='sans-serif', fontweight='bold', fontsize=14)
    ax.set_xlabel(r'$\theta_x\in(-0.5\pi, 1.5\pi)$', fontsize=14)
    ax.set_ylabel(r'$\theta_y\in(-0.5\pi, 1.5\pi)$', fontsize=14)
    plt.show()
