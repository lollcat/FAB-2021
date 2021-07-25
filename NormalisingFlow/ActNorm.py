import torch
import torch.nn as nn
import torch.nn.functional as F
from NormalisingFlow.Nets.MLP import MLP
from NormalisingFlow.base import BaseFlow


class ActNorm(BaseFlow):
    """
    https: // arxiv.org / pdf / 1807.03039.pdf
    """
    def __init__(self, x_dim, use_exp = False):
        super(ActNorm, self).__init__()
        self.loc = nn.Parameter(torch.ones(x_dim))
        self.scale = nn.Parameter(torch.ones(x_dim))
        self.register_buffer("initialised", torch.tensor(False))
        self.use_exp = use_exp

    def inverse(self, z: torch.tensor) -> (torch.tensor, torch.tensor):
        if self.initialised == False:
            self.loc.data = torch.mean(z, dim=0)
            if self.use_exp:
                self.scale.data = torch.log(torch.std(z, dim=0))
            else:
                self.scale.data = torch.log(torch.exp(torch.std(z, dim=0)) - 1.0)
            self.initialised.data = torch.tensor(True).to(self.loc.device)
        if self.use_exp:
            out = (z - self.loc) / torch.exp(self.scale)
            log_det = -torch.sum(self.scale)
        else:
            s = torch.nn.functional.softplus(self.scale)
            out = (z - self.loc) / s
            log_det = -torch.sum(torch.log(s))
        return out, log_det


    def forward(self, x: torch.tensor) -> (torch.tensor, torch.tensor):
        if self.use_exp:
            out = x*torch.exp(self.scale) + self.loc
            log_det = torch.sum(self.scale)
        else:
            s = torch.nn.functional.softplus(self.scale)
            out = x*s + self.loc
            log_det = torch.sum(torch.log(s))
        return out, log_det



if __name__ == '__main__':
    z = torch.randn(10, 2)
    actnorm = ActNorm(2)
    x, log_det = actnorm.inverse(z)
    print(torch.std(x, dim=0), torch.mean(x, dim=0))
    assert actnorm.initialised == True
    x, log_det = actnorm.inverse(z)




