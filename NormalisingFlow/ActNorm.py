import torch
import torch.nn as nn
import torch.nn.functional as F
from NormalisingFlow.Nets.MLP import MLP
from NormalisingFlow.base import BaseFlow


class ActNorm(BaseFlow):
    """
    https: // arxiv.org / pdf / 1807.03039.pdf
    """
    def __init__(self, x_dim):
        super(ActNorm, self).__init__()
        self.loc = nn.Parameter(torch.ones(x_dim))
        self.scale = nn.Parameter(torch.ones(x_dim))
        self.register_buffer("initialised", torch.tensor(False))

    def inverse(self, z: torch.tensor) -> (torch.tensor, torch.tensor):
        if self.initialised == False:
            self.loc.data = torch.mean(z, dim=0)
            self.scale.data = torch.log(torch.std(z, dim=0))
            self.initialised.data = torch.tensor(True)
        return (z - self.loc) / torch.exp(self.scale), -torch.sum(self.scale)


    def forward(self, x: torch.tensor) -> (torch.tensor, torch.tensor):
        return x*torch.exp(self.scale) + self.loc, torch.sum(self.scale)


if __name__ == '__main__':
    z = torch.randn(10, 2)
    actnorm = ActNorm(2)
    x, log_det = actnorm.inverse(z)
    print(torch.std(x, dim=0), torch.mean(x, dim=0))
    assert actnorm.initialised == True
    x, log_det = actnorm.inverse(z)




