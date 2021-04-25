from NormalisingFlow.Nets.MADE.MADE_IAF import MADE_IAF
import torch
import torch.nn as nn
from NormalisingFlow.base import BaseFlow


class IAF(nn.Module, BaseFlow):
    def __init__(self, x_dim, nodes_per_x=10, n_hidden_layers=2, reverse=True):
        super(IAF, self).__init__()
        self.x_dim = x_dim
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
        # found https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib/flows.py useful for this
        # expect my backwards and forwards are in opposite order to andrej (I go z -> x as forward)
        log_determinant = torch.zeros(x.shape[0])
        z = torch.zeros_like(x)
        if self.reverse:
            x = x.flip(dims=(-1,))
        for i in range(self.x_dim):
            m, s = self.AutoregressiveNN(z.clone())
            sigma = torch.sigmoid(s)
            z[:, i] = sigma[:, i] * x[:, i] + (1 - sigma)[:, i] * m[:, i]
            log_determinant -= torch.log(sigma[:, i])
        return z, log_determinant


if __name__ == '__main__':
    dim = 3
    z = torch.randn((10, dim))
    iaf = IAF(x_dim=3)
    x, log_determinant = iaf(z)
    print(x.shape, log_determinant.shape)
    z_backward, log_det_backward = iaf.backward(x)
    print(torch.sum(torch.abs(log_det_backward + log_determinant)))
    print(torch.sum(torch.abs(z_backward - z)))




