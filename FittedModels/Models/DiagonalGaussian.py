import torch
import torch.nn as nn
import torch.nn.functional as F
from FittedModels.Models.base import BaseLearntDistribution


class DiagonalGaussian(nn.Module, BaseLearntDistribution):
    # diagonal guassian distribution
    def __init__(self, dim=5, log_std_initial_scaling=1):
        super(DiagonalGaussian, self).__init__()
        self.dim=dim
        self.class_args = (dim, log_std_initial_scaling)
        self.class_kwargs = {}
        self.means = torch.nn.Parameter(torch.zeros(dim))
        self.log_std = torch.nn.Parameter(torch.ones(dim)*log_std_initial_scaling)
        self.distribution = self.get_distribution

    def to(self, device):
        super(DiagonalGaussian, self).to(device)
        self.distribution = self.get_distribution

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
        return torch.diag(torch.exp(self.log_std))

    @property
    def get_distribution(self):
        return torch.distributions.MultivariateNormal(self.means, self.covariance)


if __name__ == '__main__':
    dist = DiagonalGaussian()
    sample, log_prob = dist(2)
    print(sample.shape, log_prob.shape)
    dist.to("cuda")
    print(dist.sample((2,)).device)


