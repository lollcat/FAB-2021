from NormalisingFlow.Nets.MADE.MADE_IAF import MADE_IAF
import torch
import torch.nn as nn
from NormalisingFlow.base import BaseFlow



class IAF_mix(BaseFlow):
    def __init__(self, x_dim, nodes_per_x=3, n_hidden_layers=1, reversed=True, init_zeros=True):
        super(IAF_mix, self).__init__()
        self.x_dim = x_dim
        hidden_layer_width = int(nodes_per_x*x_dim)  # this lets us enter the layer width default argument dependent on x_dim
        # if this works we can rewrite AutoregressiveNN to allow any number of outputs
        self.AutoregressiveNN_1 = MADE_IAF(x_dim=x_dim, hidden_layer_width=hidden_layer_width,
                                         n_hidden_layers=n_hidden_layers, init_zeros=init_zeros)
        self.AutoregressiveNN_2 = MADE_IAF(x_dim=x_dim, hidden_layer_width=hidden_layer_width,
                                         n_hidden_layers=n_hidden_layers, init_zeros=init_zeros)
        self.AutoregressiveNN_mix_control = MADE_IAF(x_dim=x_dim, hidden_layer_width=hidden_layer_width,
                                           n_hidden_layers=n_hidden_layers, init_zeros=init_zeros)
        self.reversed = reversed

    def inverse(self, x):
        log_determinant = torch.zeros(x.shape[0]).to(x.device)
        m_1, s_1 = self.AutoregressiveNN_1(x)
        m_2, s_2 = self.AutoregressiveNN_2(x)
        control, _ = self.AutoregressiveNN_mix_control(x)
        selector = torch.sigmoid(control)
        sigma_1 = torch.sigmoid(s_1)
        sigma_2 = torch.sigmoid(s_2)
        x = (sigma_1*x + m_1) * selector + (sigma_2*x + m_2) * (1-selector)
        log_determinant += torch.sum(torch.log(sigma_1*selector + sigma_2*(1-selector)), dim=1)
        if self.reversed:
            # reverse ordering, this let's each variable take a turn being dependent if we have multiple steps
            x = x.flip(dims=(-1,))
        return x, log_determinant


    def forward(self, x):
        log_determinant = torch.zeros(x.shape[0]).to(x.device)
        z = torch.zeros_like(x)
        if self.reversed:
            x = x.flip(dims=(-1,))
        for i in range(self.x_dim):
            m_1, s_1 = self.AutoregressiveNN_1(z.clone())
            m_2, s_2 = self.AutoregressiveNN_2(z.clone())
            control, _ = self.AutoregressiveNN_mix_control(z.clone())
            selector = torch.sigmoid(control)
            sigma_1 = torch.sigmoid(s_1)
            sigma_2 = torch.sigmoid(s_2)
            z[:, i] = (x[:, i] - (selector[:, i]*m_1[:, i] + (1-selector[:, i])*m_2[:, i])) / \
                      (selector[:, i] * sigma_1[:, i] + (1-selector[:, i]) * sigma_2[:, i])
            log_determinant -= torch.log(sigma_1[:, i]*selector[:, i] + sigma_2[:, i] * (1-selector[:, i]))
        return z, log_determinant


class Reverse_IAF_MIX(IAF_mix):
    def __init__(self, *args, **kwargs):
        super(Reverse_IAF_MIX, self).__init__(*args, **kwargs)

    def forward(self, x):
        return super(Reverse_IAF_MIX, self).inverse(x)

    def inverse(self, x):
        return super(Reverse_IAF_MIX, self).forward(x)


if __name__ == '__main__':
    dim = 3
    z = torch.randn((10, dim))
    iaf = IAF_mix(x_dim=3, reversed=True)
    x, log_determinant = iaf.inverse(z)
    print(x.shape, log_determinant.shape)
    z_backward, log_det_backward = iaf.forward(x)
    print(torch.sum(torch.abs(log_det_backward + log_determinant)))
    print(torch.sum(torch.abs(z_backward - z)))






