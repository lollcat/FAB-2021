import torch
from TargetDistributions.base import BaseTargetDistribution
from TargetDistributions.MoG import MoG
import time

class MoG_old(BaseTargetDistribution):
    # mog with random mean and var
    def __init__(self, dim : int =2, n_mixes: int =5,
                 min_cov: float=0.5, loc_scaling: float=3.0):
        super(MoG_old, self).__init__()
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

        self.register_buffer("cat_probs", torch.rand(n_mixes))
        self.register_buffer("locs", locs)
        self.register_buffer("covs", covs)

    @property
    def get_distribution(self):
        mix = torch.distributions.Categorical(self.cat_probs)
        com = torch.distributions.MultivariateNormal(self.locs, self.covs)
        return torch.distributions.MixtureSameFamily(mixture_distribution=mix, component_distribution=com)

    def log_prob(self, x: torch.Tensor):
        return self.get_distribution.log_prob(x)

    def sample(self, shape=(1,)):
        return self.get_distribution.sample(shape)

if __name__ == '__main__':
    Mog = MoG(2)
    Mog_old = MoG_old(2)

    for i in range(5):
        start_MoG = time.time()
        Mog.log_prob(torch.zeros(100,2))
        print("Mog", time.time() - start_MoG)

        start_MoG_old = time.time()
        Mog_old.log_prob(torch.zeros(100,2))
        print("Mog old", time.time() - start_MoG_old)
