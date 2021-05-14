import torch
from TargetDistributions.base import BaseTargetDistribution

class Guassian_FullCov(torch.distributions.multivariate_normal.MultivariateNormal, BaseTargetDistribution):
    def __init__(self, dim=5, scale_covariance=1):
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
        super(Guassian_FullCov, self).__init__(loc, scale_tril)


if __name__ == '__main__':
    size = 5
    test_unnormalised_dist = Guassian_FullCov(size)
    samples = torch.ones((size,))
    print(test_unnormalised_dist.log_prob(samples))