import torch
from TargetDistributions.base import BaseTargetDistribution
import torch.nn as nn

class Guassian_FullCov(BaseTargetDistribution):
    def __init__(self, dim=5, scale_covariance=1):
        super(Guassian_FullCov, self).__init__()
        # scale_covariance just multiplies covariance by a constant, giving us the ability to make the distribution
        # more or less wide
        loc = torch.randn(size=(dim,))
        scale_tril = torch.abs(torch.tril(torch.randn((dim, dim)), -1)) + 0.1  # non-diagonal elements, floor at 0.1
        # ensure diagonal elements are bigger than non diagonal elements
        # this is pretty hacky and can probably find a better way
        scale_tril += torch.diag(torch.abs(torch.randn(size=(dim,)))) \
                      + dim * torch.maximum(torch.diag(torch.max(scale_tril, dim=1).values),
                                            torch.diag(torch.max(scale_tril, dim=0).values))
        scale_tril *= scale_covariance
        scale_tril = torch.tril(scale_tril)
        self.register_buffer("scale_tril", scale_tril)
        self.register_buffer("loc", loc)

    @property
    def get_distribution(self):
        return torch.distributions.multivariate_normal.MultivariateNormal(self.loc, scale_tril=self.scale_tril)

    def log_prob(self, x):
        return self.get_distribution.log_prob(x)

    def sample(self, shape=(1,)):
        return self.get_distribution.sample(shape)


if __name__ == '__main__':
    size = 5
    test_unnormalised_dist = Guassian_FullCov(size)
    samples = torch.ones((size,))
    print(test_unnormalised_dist.log_prob(samples))