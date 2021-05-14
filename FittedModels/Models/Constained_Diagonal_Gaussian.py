import torch
import torch.nn as nn
import torch.nn.functional as F
from FittedModels.Models.base import BaseLearntDistribution


class DiagonalGaussian(nn.Module, BaseLearntDistribution):
    # diagonal guassian distribution constained to have bounded covariance
    def __init__(self, dim=5, pre_sigmoid_std_initial_scaling=-1, std_min=-1, std_max=10):
        super(DiagonalGaussian, self).__init__()
        self.means = torch.nn.Parameter(torch.zeros(dim))
        self.std_min = std_min
        self.std_max = std_max
        self.pre_sigmoid_std = torch.nn.Parameter(torch.ones(dim) * pre_sigmoid_std_initial_scaling)

    def forward(self, batch_size=1):
        distribution = self.distribution
        sample = distribution.rsample((batch_size, ))
        log_prob = distribution.log_prob(sample)
        return sample, log_prob

    def sample(self, sample_shape=(1,)):
        # This function is for sampling after training, where we don't need gradients
        with torch.no_grad():
            return self.distribution.sample(sample_shape=sample_shape)

    def log_prob(self, x):
        return self.distribution.log_prob(x)

    @property
    def covariance(self):
        return torch.diag(self.std_min + torch.sigmoid(self.pre_sigmoid_std) * (self.std_max - self.std_min))

    @property
    def distribution(self):
        return torch.distributions.MultivariateNormal(self.means, self.covariance)


if __name__ == '__main__':
    dist = DiagonalGaussian()
    sample, log_prob = dist(2)
    print(sample.shape, log_prob.shape)
    print(dist.covariance)


