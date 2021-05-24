import torch
import torch.nn as nn
import torch.nn.functional as F
from NormalisingFlow.Nets.MLP import MLP
from NormalisingFlow.base import BaseFlow

class RealNVP(nn.Module, BaseFlow):
    def __init__(self, x_dim, nodes_per_x=10, n_hidden_layers=3, reversed=False, use_exp=False):
        super(RealNVP, self).__init__()
        self.use_exp = use_exp
        self.d = x_dim // 2  # elements copied from input to output
        self.D_minus_d = x_dim - self.d  # dependent on d elements
        hidden_layer_width = nodes_per_x*x_dim  # this lets us enter the layer width default argument dependent on x_dim
        self.MLP = MLP(self.d, self.D_minus_d*2, hidden_layer_width, n_hidden_layers=n_hidden_layers)
        self.reversed = reversed

    def inverse(self, x):
        """return samples and log | dy / dx |"""
        if self.reversed:
            x = x.flip(dims=(-1,))
        x_1_to_d = x[:, 0:self.d]
        x_d_plus_1_to_D = x[:, self.d:]
        y_1_to_d = x_1_to_d
        st = self.MLP(x_1_to_d)
        s, t = st.split(self.D_minus_d, dim=-1)
        t, s = self.reparameterise_s(t, s)
        if self.use_exp:
            y_d_plus_1_to_D = x_d_plus_1_to_D * torch.exp(s) + t
            log_determinant = torch.sum(s, dim=-1)
        else:
            sigma = torch.sigmoid(s)
            y_d_plus_1_to_D = x_d_plus_1_to_D * sigma + t
            log_determinant = torch.sum(torch.log(sigma), dim=-1)

        y = torch.cat([y_1_to_d, y_d_plus_1_to_D], dim=-1)

        if self.reversed:
            y = y.flip(dims=(-1,))
        return y, log_determinant

    def forward(self, y):
        if self.reversed:
            y = y.flip(dims=(-1,))
        y_1_to_d = y[:, 0:self.d]
        y_d_plus_1_to_D = y[:, self.d:]
        x_1_to_d = y_1_to_d
        st = self.MLP(y_1_to_d)
        s, t = st.split(self.D_minus_d, dim=-1)
        t, s = self.reparameterise_s(t, s)
        if self.use_exp:
            x_d_plus_1_to_D = (y_d_plus_1_to_D - t) * torch.exp(-s)
            log_determinant = -torch.sum(s, dim=-1)
        else:
            sigma = torch.sigmoid(s)
            x_d_plus_1_to_D = (y_d_plus_1_to_D - t) / sigma
            log_determinant = -torch.sum(torch.log(sigma), dim=-1)
        x = torch.cat([x_1_to_d, x_d_plus_1_to_D], dim=-1)
        if self.reversed:
            x = x.flip(dims=(-1,))
        return x, log_determinant

    def reparameterise_s(self, t: torch.tensor, s: torch.tensor) -> torch.tensor:
        if self.use_exp:
            return t/10, s / 10
        else:  # if sigmoid reparameterise s to be close to 1.5
            return t/10, s / 10 + 1.5


if __name__ == '__main__':
    dim = 5
    x = torch.randn((11, dim))
    test = RealNVP(x_dim=dim, reversed=True)
    y, log_determinant_forward = test(x)
    x_out, log_determinant_backward = test.forward(y)
    print(log_determinant_forward, log_determinant_backward)
    print(y.shape, log_determinant_forward.shape)
    assert torch.max(torch.abs(x_out - x)) < 1e-6
    assert torch.max(torch.abs(log_determinant_forward + log_determinant_backward)) < 1e-6
    print(torch.sum(torch.abs(x_out - x)))