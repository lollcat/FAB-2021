import torch
import torch.nn as nn
from TargetDistributions.base import BaseTargetDistribution

class MoG(torch.distributions.MixtureSameFamily, BaseTargetDistribution):
    def __init__(self, dim=2, n_mixes=5, min_cov=0.5, loc_scaling=3.0):
        self.dim = dim
        self.n_mixes = n_mixes
        self.distributions = []
        locs = []
        covs = []
        for i in range(n_mixes):
            loc = torch.randn(dim)*loc_scaling
            covariance = torch.diag(torch.rand(dim) + min_cov)
            locs.append(loc[None, :])
            covs.append(covariance[None, :, :])

        locs = torch.cat(locs)
        covs = torch.cat(covs)
        mix = torch.distributions.Categorical(torch.rand(n_mixes))
        com = torch.distributions.MultivariateNormal(locs, covs)
        super(MoG, self).__init__(mixture_distribution = mix, component_distribution = com)





if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Utils import plot_distribution
    size = 2
    dist = MoG(size)
    samples = dist.sample((10,)) #torch.randn((10,size))
    log_probs = dist.log_prob(samples)
    print(log_probs)
    print(log_probs.shape)
    plot_distribution(dist, n_points=300)
    plt.show()
    print(dist.sample((10,)))
