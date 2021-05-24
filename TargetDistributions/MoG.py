import torch
import torch.nn as nn
from TargetDistributions.base import BaseTargetDistribution

class custom_MoG(torch.distributions.MixtureSameFamily, BaseTargetDistribution, nn.Module):
    # Mog with hard coded mean and cov
    def __init__(self, dim=2, loc_scaling=1, cov_scaling=1, locs_=(-1, 1)):
        self.dim = dim
        distributions = []
        locs = []
        covs = []
        for i in range(2):
            loc = torch.ones(dim)*locs_[i]*loc_scaling
            covariance = torch.eye(dim)*cov_scaling
            locs.append(loc[None, :])
            covs.append(covariance[None, :, :])
        locs = torch.cat(locs)
        covs = torch.cat(covs)
        mix = torch.distributions.Categorical(torch.tensor([0.6, 0.4]))
        com = torch.distributions.MultivariateNormal(locs, covs)
        super(custom_MoG, self).__init__(mixture_distribution=mix, component_distribution = com)

class MoG(torch.distributions.MixtureSameFamily, BaseTargetDistribution, nn.Module):
    # mog with random mean and var
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
