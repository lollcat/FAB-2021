import abc
import torch
import torch.nn as nn


class BaseFlow(nn.Module):
    def __init__(self):
        super(BaseFlow, self).__init__()


    def inverse(self, z: torch.tensor) -> (torch.tensor, torch.tensor):
        """
        Inverse flow, from z to x, i.e. if combined with prior, is used for sampling
        return x and log det(dx/dz)
        """
        raise NotImplementedError


    def forward(self, x: torch.tensor) -> (torch.tensor, torch.tensor):
        """
        Computes z, and log det(dz/dx) useful for density evaluation
        """
        raise NotImplementedError





