from NormalisingFlow.Nets.MADE.MADE_IAF import MADE_IAF
import torch
import torch.nn as nn
from NormalisingFlow.base import BaseFlow



class IAF(BaseFlow):
    def __init__(self, x_dim, nodes_per_x=3, n_hidden_layers=1, reversed=True, use_exp=True, init_zeros=True):
        super(IAF, self).__init__()
        self.use_exp = use_exp
        self.x_dim = x_dim
        hidden_layer_width = nodes_per_x*x_dim  # this lets us enter the layer width default argument dependent on x_dim
        self.AutoregressiveNN = MADE_IAF(x_dim=x_dim, hidden_layer_width=hidden_layer_width,
                                         n_hidden_layers=n_hidden_layers, init_zeros=init_zeros)
        self.reversed = reversed

    def inverse(self, x):
        log_determinant = torch.zeros(x.shape[0]).to(x.device)
        m, s = self.AutoregressiveNN(x)
        m, s = self.reparameterise(m, s)
        if self.use_exp:
            x = torch.exp(s) * x + m
            log_determinant += torch.sum(s, dim=1)
        else:
            sigma = torch.sigmoid(s)
            x = sigma*x + (1-sigma)*m
            log_determinant += torch.sum(torch.log(sigma), dim=1)
        if self.reversed:
            # reverse ordering, this let's each variable take a turn being dependent if we have multiple steps
            x = x.flip(dims=(-1,))
        return x, log_determinant

    def reparameterise(self, m, s):
        # make s start relatively close to 0, and exp(s) close to 1, and m close to 0
        # this helps the flow start of not overly funky!
        if self.use_exp:
            return m, s
        else:  # if sigmoid reparameterise s to be close to 1.5
            return m/10, s/10 + 1.5

    def forward(self, x):
        # found https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib/flows.py useful for this
        log_determinant = torch.zeros(x.shape[0]).to(x.device)
        z = torch.zeros_like(x)
        if self.reversed:
            x = x.flip(dims=(-1,))
        for i in range(self.x_dim):
            m, s = self.AutoregressiveNN(z.clone())
            m, s = self.reparameterise(m, s)
            if self.use_exp is True:
                z[:, i] = (x[:, i] - m[:, i])*torch.exp(-s[:, i])
                log_determinant -= s[:, i]
            else:
                sigma = torch.sigmoid(s)
                z[:, i] = (x[:, i] - (1-sigma[:, i])*m[:, i]) / sigma[:, i]
                log_determinant -= torch.log(sigma[:, i])
        return z, log_determinant


if __name__ == '__main__':
    dim = 3
    z = torch.randn((10, dim))
    iaf = IAF(x_dim=3, reversed=True)
    from copy import deepcopy
    iaf_copy2 = deepcopy(iaf.AutoregressiveNN.FirstLayer)
    iaf_copy1 = deepcopy(iaf.AutoregressiveNN)
    iaf_copy = deepcopy(iaf)
    print(2+2)
    """
    x, log_determinant = iaf(z)
    print(x.shape, log_determinant.shape)
    z_backward, log_det_backward = iaf.forward(x)
    print(torch.sum(torch.abs(log_det_backward + log_determinant)))
    print(torch.sum(torch.abs(z_backward - z)))
    """







