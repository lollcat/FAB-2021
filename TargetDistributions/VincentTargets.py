"""
Taken from
https://github.com/VincentStimper/normalizing-flows
to compare code
"""


import torch
import torch.nn as nn
import numpy as np


class PriorDistribution(nn.Module):
    def __init__(self):
        super(PriorDistribution, self).__init__()

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        raise NotImplementedError





class TwoModes(PriorDistribution):
    def __init__(self, loc, scale):
        super(TwoModes, self).__init__()
        """
        Distribution 2d with two modes at z[0] = -loc and z[0] = loc
        :param loc: distance of modes from the origin
        :param scale: scale of modes
        """
        self.register_buffer("loc", torch.tensor([loc]))
        self.register_buffer("scale", torch.tensor([scale]))

    def log_prob(self, z):
        """
        log(p) = 1/2 * ((norm(z) - loc) / (2 * scale)) ** 2
                - log(exp(-1/2 * ((z[0] - loc) / (3 * scale)) ** 2) + exp(-1/2 * ((z[0] + loc) / (3 * scale)) ** 2))
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        a = torch.abs(z[:, 0])
        eps = torch.abs(self.loc)

        log_prob = - 0.5 * ((torch.norm(z, dim=1) - self.loc) / (2 * self.scale)) ** 2 \
                   - 0.5 * ((a - eps) / (3 * self.scale)) ** 2 \
                   + torch.log(1 + torch.exp(-2 * (a * eps) / (3 * self.scale) ** 2))

        return log_prob

