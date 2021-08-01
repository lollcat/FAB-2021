import torch
import torch.nn as nn
from TargetDistributions.base import BaseTargetDistribution
from Utils.numerical_utils import MC_estimate_true_expectation
from Utils.numerical_utils import quadratic_function
import numpy as np


class MoG(BaseTargetDistribution):
    # mog with random mean and var
    def __init__(self, dim : int =2, n_mixes: int =5,
                 min_cov: float=0.5, loc_scaling: float=3.0, diagonal_covariance=True,
                 cov_scaling=1.0, uniform_component_probs = False):
        super(MoG, self).__init__()
        self.dim = dim
        self.n_mixes = n_mixes
        self.distributions = []
        locs = []
        scale_trils = []
        for i in range(n_mixes):
            loc = torch.randn(dim)*loc_scaling
            if diagonal_covariance:
                scale_tril = torch.diag(torch.rand(dim)*cov_scaling + min_cov)
            else:
                # https://stackoverflow.com/questions/58176501/how-do-you-generate-positive-definite-matrix-in-pytorch
                Sigma_k = torch.rand(dim, dim) * cov_scaling + min_cov
                Sigma_k = torch.mm(Sigma_k, Sigma_k.t())
                Sigma_k.add_(torch.eye(dim))
                scale_tril = torch.tril(Sigma_k)
            locs.append(loc[None, :])
            scale_trils.append(scale_tril[None, :, :])

        locs = torch.cat(locs)
        scale_trils = torch.cat(scale_trils)
        if uniform_component_probs:
             self.register_buffer("cat_probs", torch.ones(n_mixes))
        else:
            self.register_buffer("cat_probs", torch.rand(n_mixes))
        self.register_buffer("locs", locs)
        self.register_buffer("scale_trils", scale_trils)
        self.distribution = self.get_distribution
        self.expectation_function = quadratic_function
        self.true_expectation = MC_estimate_true_expectation(self, self.expectation_function, int(1e6)).item()

    def to(self, device):
        super(MoG, self).to(device)
        self.distribution = self.get_distribution

    @property
    def get_distribution(self):
        mix = torch.distributions.Categorical(self.cat_probs)
        com = torch.distributions.MultivariateNormal(self.locs, scale_tril=self.scale_trils)
        return torch.distributions.MixtureSameFamily(mixture_distribution=mix, component_distribution=com)

    def log_prob(self, x: torch.Tensor):
        return self.distribution.log_prob(x)

    def sample(self, shape=(1,)):
        return self.distribution.sample(shape)

    @torch.no_grad()
    def performance_metrics(self, train_class, x_samples, log_w,
                            n_batches_stat_aggregation=10):
        samples_per_batch = x_samples.shape[0] // n_batches_stat_aggregation
        expectations = []
        for i, batch_number in enumerate(range(n_batches_stat_aggregation)):
            if i != n_batches_stat_aggregation - 1:
                log_w_batch = log_w[batch_number * samples_per_batch:(batch_number + 1) * samples_per_batch]
                x_samples_batch = x_samples[batch_number * samples_per_batch:(batch_number + 1) * samples_per_batch]
            else:
                log_w_batch = log_w[batch_number * samples_per_batch:]
                x_samples_batch = x_samples[batch_number * samples_per_batch:]
            expectation = train_class.AIS_train.\
                estimate_expectation_given_samples_and_log_w(self.expectation_function,
                                                             x_samples_batch, log_w_batch).item()
            expectations.append(expectation)
        bias_normed = np.abs(np.mean(expectations) - self.true_expectation) / self.true_expectation
        std_normed = np.std(expectations) / self.true_expectation
        summary_dict = {"bias_normed": bias_normed, "std_normed": std_normed}
        long_dict = {}
        return summary_dict, long_dict


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
    size = 20
    dist = MoG(size, diagonal_covariance=False, cov_scaling=0.1, min_cov=0.0, loc_scaling=8.0, n_mixes=size,
               uniform_component_probs=True)
    samples = dist.sample((10,)) #torch.randn((10,size))
    log_probs = dist.log_prob(samples)
    print(log_probs)
    print(log_probs.shape)
    if size == 2:
        from Utils.plotting_utils import plot_distribution
        plot_distribution(dist, n_points=300)
    else:
        from Utils.plotting_utils import plot_marginals
        plot_marginals(distribution=dist, n_samples=1000, clamp_samples=30)
        plt.savefig("Mog_tst.png")
    plt.show()
    print(dist.sample((10,)))
