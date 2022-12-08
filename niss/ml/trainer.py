import numpy as np
import torch
import logging


class NNTrainer:
    def __init__(self, optimizer=torch.optim.Adam, learning_rate=0.0004, num_steps=8000, num_prints=20):
        self.optimizer_type = optimizer
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.num_prints = num_prints

    def train(self, net, criterion, nn_input):
        # Create an optimizer
        optimizer = self.optimizer_type(net.parameters(), lr=self.learning_rate)
        logging.info(f'Optimizer used: {type(optimizer).__name__}')
        # training loop
        print_interval = self.num_steps // self.num_prints
        if type(optimizer).__name__ == "LBFGS":
            for t in range(self.num_steps):
                def closure():
                    optimizer.zero_grad()
                    nn_output_ = net(nn_input)
                    loss_ = criterion(nn_output_)
                    loss_.backward()
                    return loss_
                loss = closure()
                if np.remainder(t, print_interval) == 0:
                    logging.info(f'Step {t:5} : Measured Smoothing factor = {loss.item():.4e}')
                optimizer.step(closure)
        else:
            for t in range(self.num_steps):
                optimizer.zero_grad()  # zero the gradient buffers
                nn_output = net(nn_input)
                loss = criterion(nn_output)
                if np.remainder(t, print_interval) == 0:
                    logging.info(f'Step {t:5} : Measured Smoothing factor = {loss.item():.4e}')
                # compute gradient
                loss.backward()
                # Does the update
                optimizer.step()
