from NormalisingFlow.Nets.MADE.MADE_IAF import MADE_IAF
import torch
import torch.nn as nn
from NormalisingFlow.base import BaseFlow

class Monitor_NaN:
    # see  https://github.com/pytorch/pytorch/issues/15131
    def __init__(self):
        self.found_Nan = False

    def overwrite_NaN_grad(self, grad):
        if True in torch.isnan(grad):
            if self.found_Nan is False:
                print("found a NaN and overwrote it during flow gradient calculation")
                self.found_Nan = True
            grad[torch.isnan(grad)] = 0
        return grad

class IAF(nn.Module, BaseFlow):
    def __init__(self, x_dim, nodes_per_x=10, n_hidden_layers=2, reverse=True):
        super(IAF, self).__init__()
        self.x_dim = x_dim
        hidden_layer_width = nodes_per_x*x_dim  # this lets us enter the layer width default argument dependent on x_dim
        self.AutoregressiveNN = MADE_IAF(x_dim=x_dim, hidden_layer_width=hidden_layer_width,
                                         n_hidden_layers=n_hidden_layers)
        self.reverse = reverse
        self.Monitor_NaN = Monitor_NaN()

    def forward(self, x):
        log_determinant = torch.zeros(x.shape[0])
        m, s = self.AutoregressiveNN(x)
        m, s = self.reparameterise_s(m, s)
        if m.requires_grad: # to prevent grad problems
            m.register_hook(self.Monitor_NaN.overwrite_NaN_grad)
            s.register_hook(self.Monitor_NaN.overwrite_NaN_grad)
        x = torch.exp(s) * x + m
        log_determinant += torch.sum(s, dim=1)
        if self.reverse:
            # reverse ordering, this let's each variable take a turn being dependent if we have multiple steps
            x = x.flip(dims=(-1,))
        return x, log_determinant

    def forward_with_hooks(self, x):
        if x.requires_grad:
            #x.register_hook(lambda grad: print("\n\ngrad x before flow pre clamp max, min, nana", grad.max(), grad.min(),
            #                                  torch.sum(torch.isnan(grad))))
            #x.register_hook(lambda grad: print("\n\ngrad x before flow max, min, nana", grad))
            #x.register_hook(lambda grad: grad.clamp(-1e10, 1e10))
            #x.register_hook(lambda grad: print("\n\ngrad x before flow post clamp max, min, nana", grad.max(), grad.min(),
            #                                   torch.sum(torch.isnan(grad))))
            pass
        log_determinant = torch.zeros(x.shape[0])
        m, s = self.AutoregressiveNN(x)
        m, s = self.reparameterise_s(m, s)
        #m.register_hook(lambda grad: grad.clamp(-1e10, 1e10))
        #s.register_hook(lambda grad: grad.clamp(-1e10, 1e10))
        #m.register_hook(lambda grad: print("\n\ngrad m flow max, min, nana", grad.max(), grad.min(),
        #                                   torch.sum(torch.isnan(grad))))
        #s.register_hook(lambda grad: print("\n\ngrad s flow max, min, nana", grad.max(), grad.min(),
        #                                   torch.sum(torch.isnan(grad))))
        if m.requires_grad:
            m.register_hook(overwrite_NaN_grad)
            s.register_hook(overwrite_NaN_grad)
        x = torch.exp(s) * x + m
        log_determinant += torch.sum(s, dim=1)
        if self.reverse:
            # reverse ordering, this let's each variable take a turn being dependent if we have multiple steps
            x = x.flip(dims=(-1,))
        #x.register_hook(lambda grad: grad.clamp(-1e1, 1e1))
        x.register_hook(lambda grad: print("\n\ngrad x after flow max, min, nana", grad.max(), grad.min(),
                                           torch.sum(torch.isnan(grad))))
        #log_determinant.register_hook(lambda grad: print("\n\ngrad log_det flow max min nan", grad.max(), grad.min(),
        #                                          torch.sum(torch.isnan(grad))))
        return x, log_determinant

    def reparameterise_s(self, m, s):
        # make s start relatively close to 0, and exp(s) close to 1, and m close to 0
        # this helps the flow start of not overly funky!
        return m/10, s/10


    def backward(self, x):
        # found https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib/flows.py useful for this
        # expect my backwards and forwards are in opposite order to andrej (I go z -> x as forward)
        log_determinant = torch.zeros(x.shape[0])
        z = torch.zeros_like(x)
        if self.reverse:
            x = x.flip(dims=(-1,))
        for i in range(self.x_dim):
            m, s = self.AutoregressiveNN(z.clone())
            m, s = self.reparameterise_s(m, s)
            z[:, i] = (x[:, i] - m[:, i])*torch.exp(-s[:, i])
            log_determinant -= s[:, i]
        return z, log_determinant


if __name__ == '__main__':
    dim = 3
    z = torch.randn((10, dim))
    iaf = IAF(x_dim=3, reverse=True)
    x, log_determinant = iaf(z)
    print(x.shape, log_determinant.shape)
    z_backward, log_det_backward = iaf.backward(x)
    print(torch.sum(torch.abs(log_det_backward + log_determinant)))
    print(torch.sum(torch.abs(z_backward - z)))





