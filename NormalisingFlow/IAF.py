from NormalisingFlow.Nets.MADE.MADE_IAF import MADE_IAF
import torch
import torch.nn as nn
from NormalisingFlow.base import BaseFlow


class IAF(nn.Module, BaseFlow):
    def __init__(self, x_dim, nodes_per_x=10, n_hidden_layers=2, reverse=True):
        super(IAF, self).__init__()
        hidden_layer_width = nodes_per_x*x_dim  # this lets us enter the layer width default argument dependent on x_dim
        self.AutoregressiveNN = MADE_IAF(x_dim=x_dim, hidden_layer_width=hidden_layer_width,
                                         n_hidden_layers=n_hidden_layers)
        self.reverse = reverse

    def forward(self, x):
        log_determinant = torch.zeros(x.shape[0])
        m, s = self.AutoregressiveNN(x)
        sigma = torch.sigmoid(s)
        x = sigma * x + (1 - sigma) * m
        log_determinant += torch.sum(torch.log(sigma), dim=1)
        if self.reverse:
            # reverse ordering, this let's each variable take a turn being dependent if we have multiple steps
            x = x.flip(dims=(-1,))
        return x, log_determinant

    def backward(self, x):
        raise Exception("not implemented yet, and IAF is slow for this")

if __name__ == '__main__':
    dim = 3
    x = torch.randn((10, dim))
    iaf = IAF(x_dim=3)
    x, log_determinant = iaf(x)
    print(x)
    print(x.shape, log_determinant.shape)




