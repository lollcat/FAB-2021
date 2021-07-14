import torch
import torch.nn as nn
from TargetDistributions.base import BaseTargetDistribution

class custom_MoG(BaseTargetDistribution):
    # Mog with hard coded mean and cov
    def __init__(self, dim=2, loc_scaling=1, cov_scaling=1, locs_=(-1, 1)):
        super(custom_MoG, self).__init__()
        self.dim = dim
        locs = []
        covs = []
        for loc_ in locs_:
            loc = torch.ones(dim)*loc_*loc_scaling
            covariance = torch.eye(dim)*cov_scaling
            locs.append(loc[None, :])
            covs.append(covariance[None, :, :])
        locs = torch.cat(locs)
        covs = torch.cat(covs)
        self.register_buffer("locs", locs)
        self.register_buffer("covs", covs)
        self.register_buffer("cat_probs", torch.ones(len(locs_))/len(locs_))
        self.distribution = self.get_distribution

    def to(self, device):
        super(custom_MoG, self).to(device)
        self.distribution = self.get_distribution

    @property
    def get_distribution(self):
        mix = torch.distributions.Categorical(self.cat_probs)
        com = torch.distributions.MultivariateNormal(self.locs, self.covs)
        return torch.distributions.MixtureSameFamily(mixture_distribution=mix, component_distribution=com)

    def log_prob(self, x):
        return self.distribution.log_prob(x)

    def sample(self, shape=(1,)):
        return self.distribution.sample(shape)



class MoG(BaseTargetDistribution):
    # mog with random mean and var
    def __init__(self, dim : int =2, n_mixes: int =5,
                 min_cov: float=0.5, loc_scaling: float=3.0):
        super(MoG, self).__init__()
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
        self.distribution = self.get_distribution

    def to(self, device):
        super(MoG, self).to(device)
        self.distribution = self.get_distribution

    @property
    def get_distribution(self):
        mix = torch.distributions.Categorical(self.cat_probs)
        com = torch.distributions.MultivariateNormal(self.locs, self.covs)
        return torch.distributions.MixtureSameFamily(mixture_distribution=mix, component_distribution=com)

    def log_prob(self, x: torch.Tensor):
        return self.distribution.log_prob(x)

    def sample(self, shape=(1,)):
        return self.distribution.sample(shape)

class Triangle_MoG(BaseTargetDistribution):
    # Mog with hard coded mean and cov to form a triangle
    def __init__(self, loc_scaling=5, cov_scaling=1):
        super(Triangle_MoG, self).__init__()
        dim = 2
        locs = torch.stack([torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([-1.0, 0.0])])*loc_scaling
        covs = torch.stack([torch.eye(dim)*cov_scaling]*3)
        self.register_buffer("locs", locs)
        self.register_buffer("covs", covs)
        self.register_buffer("cat_probs", torch.tensor([0.2, 0.6, 0.2]))
        self.distribution = self.get_distribution

    def to(self, device):
        super(Triangle_MoG, self).to(device)
        self.distribution = self.get_distribution


    @property
    def get_distribution(self):
        mix = torch.distributions.Categorical(self.cat_probs)
        com = torch.distributions.MultivariateNormal(self.locs, self.covs)
        return torch.distributions.MixtureSameFamily(mixture_distribution=mix, component_distribution=com)

    def log_prob(self, x):
        return self.distribution.log_prob(x)

    def sample(self, shape=(1,)):
        return self.distribution.sample(shape)

class Difficult_MoG(BaseTargetDistribution):
    def __init__(self, loc_scaling=5, cov_scaling=1):
        super(Difficult_MoG, self).__init__()
        dim = 2
        locs = torch.stack([torch.tensor([2.0, 0.0]),
                            torch.tensor([0.0, 2.0]),
                            torch.tensor([0.0, -2.0]),
                            torch.tensor([1, 1.5])]
                           )*loc_scaling
        covs = torch.stack([torch.eye(dim)*cov_scaling*0.05,  # biggest hump is quite steep
                            torch.eye(dim)*cov_scaling,
                            torch.eye(dim)*cov_scaling,
                            torch.eye(dim)*cov_scaling*0.1]
                           )
        self.register_buffer("locs", locs)
        self.register_buffer("covs", covs)
        self.register_buffer("cat_probs", torch.tensor([0.5, 0.1, 0.2, 0.2]))
        self.distribution = self.get_distribution

    def to(self, device):
        super(Difficult_MoG, self).to(device)
        self.distribution = self.get_distribution

    @property
    def get_distribution(self):
        mix = torch.distributions.Categorical(self.cat_probs)
        com = torch.distributions.MultivariateNormal(self.locs, self.covs)
        return torch.distributions.MixtureSameFamily(mixture_distribution=mix, component_distribution=com)

    def log_prob(self, x):
        return self.distribution.log_prob(x)

    def sample(self, shape=(1,)):
        return self.distribution.sample(shape)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Utils.plotting_utils import plot_distribution
    size = 2
    dist = MoG(size)
    samples = dist.sample((10,)) #torch.randn((10,size))
    log_probs = dist.log_prob(samples)
    print(log_probs)
    print(log_probs.shape)
    plot_distribution(dist, n_points=300)
    plt.show()
    print(dist.sample((10,)))
