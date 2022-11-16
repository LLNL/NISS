import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


class Theta2D:
    def __init__(self, num_theta, start=-torch.pi / 2, end=torch.pi / 2, quadrant=torch.tensor([0, 1, 2, 3])):
        """
        :param num_theta: the number of points in theta grid
        :param start:
        :param end:
        :param quadrant:
        """
        self.num_theta = num_theta
        self.h_theta = (end - start) / num_theta
        self.theta = torch.linspace(start + self.h_theta / 2, end - self.h_theta / 2, num_theta)
        # theta grid
        self.theta_grid = torch.empty(num_theta, num_theta, 2)
        self.theta_grid[:, :, 0] = self.theta.reshape(-1, 1).repeat(1, num_theta)  # Y-dim
        self.theta_grid[:, :, 1] = self.theta.reshape(1, -1).repeat(num_theta, 1)  # X-dim
        # theta grids in all quadrants: [num_theta, num_theta, dim=2, num_quad]
        self.quadrant = quadrant
        # mesh quad    ^  [1, 3     -> x
        #            y |   0, 2]
        self.quad = torch.tensor([[0, torch.pi,        0, torch.pi],
                                  [0,        0, torch.pi, torch.pi]])[:, quadrant]
        self.theta_quad = self.theta_grid.unsqueeze(3) + self.quad[None, None, :, :]

    def plot(self, all_symbol, title, num_levels=10):
        """
        :return: plot LFA
        """
        fig, ax = plt.subplots()
        theta_plot = np.zeros(2 * self.num_theta)
        theta_plot[0:self.num_theta] = self.theta
        theta_plot[self.num_theta: 2 * self.num_theta] = self.theta + torch.pi
        smooth_symbol_plot = np.zeros((2 * self.num_theta, 2 * self.num_theta))
        for j in range(self.quadrant.numel()):
            i = self.quadrant[j].item()
            ix = i // 2
            iy = i - ix * 2
            smooth_symbol_plot[iy * self.num_theta:(iy + 1) * self.num_theta,
                               ix * self.num_theta:(ix + 1) * self.num_theta] = all_symbol[:, :, j]
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
