import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


class LFA:
    def __init__(self, num_theta):
        """
        :param num_theta: the number of points in theta grid
        """
        self.num_theta = num_theta
        h_theta = torch.pi / num_theta
        self.theta = torch.linspace(-torch.pi / 2 + h_theta / 2, torch.pi / 2 - h_theta / 2, num_theta)

    def lfa(self, operator, num_quadrant=4):
        """
        :return: all lfa symbols (over theta) of operator
        """
        all_sym = torch.zeros(self.num_theta, self.num_theta, 4)
        theta0 = self.theta.reshape(-1, 1).repeat(1, self.num_theta)
        theta1 = self.theta.reshape(1, -1).repeat(self.num_theta, 1)
        all_sym[:, :, 0] = operator.symbol(theta0, theta1)
        all_sym[:, :, 1] = operator.symbol(theta0 + torch.pi, theta1)
        all_sym[:, :, 2] = operator.symbol(theta0, theta1 + torch.pi)
        all_sym[:, :, 3] = operator.symbol(theta0 + torch.pi, theta1 + torch.pi)

        return all_sym

    def plot(self, all_symbol, title, num_levels=10):
        """
        :return: plot LFA
        """
        fig, ax = plt.subplots()
        theta_plot = np.zeros(2 * self.num_theta)
        theta_plot[0:self.num_theta] = self.theta
        theta_plot[self.num_theta: 2 * self.num_theta] = self.theta + torch.pi
        smooth_symbol_plot = np.zeros((2 * self.num_theta, 2 * self.num_theta))
        smooth_symbol_plot[0:self.num_theta, 0:self.num_theta] = all_symbol[:, :, 0]
        smooth_symbol_plot[self.num_theta:2 * self.num_theta, 0:self.num_theta] = all_symbol[:, :, 1]
        smooth_symbol_plot[0:self.num_theta, self.num_theta:2 * self.num_theta] = all_symbol[:, :, 2]
        smooth_symbol_plot[self.num_theta:2 * self.num_theta, self.num_theta:2 * self.num_theta] = all_symbol[:, :, 3]
        cp = plt.contour(theta_plot, theta_plot, smooth_symbol_plot, levels=num_levels)
        plt.clabel(cp, fontsize=8, colors='black')
        norm = colors.Normalize(vmin=cp.cvalues.min(), vmax=cp.cvalues.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cp.cmap)
        # sm.set_array()
        fig.colorbar(sm, ticks=cp.levels)
        plt.plot([np.min(theta_plot), np.max(theta_plot)], [0.5 * torch.pi, 0.5 * torch.pi], color='red')
        plt.plot([0.5 * torch.pi, 0.5 * torch.pi], [np.min(theta_plot), np.max(theta_plot)], color='red')
        ax.set_title(title, fontfamily='sans-serif', fontweight='bold', fontsize=14)
        ax.set_xlabel(r'$\theta_x\in(-0.5\pi, 1.5\pi)$', fontsize=14)
        ax.set_ylabel(r'$\theta_y\in(-0.5\pi, 1.5\pi)$', fontsize=14)
        plt.show()
