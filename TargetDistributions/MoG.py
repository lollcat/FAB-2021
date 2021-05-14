import torch
import torch.nn as nn
from TargetDistributions.base import BaseTargetDistribution

class MoG(BaseTargetDistribution):
    def __init__(self, dim=2, n_mixes=5, min_cov=0.5):
        super(MoG, self).__init__()
        self.dim = dim
        self.n_mixes = n_mixes
        self.distributions = []
        for i in range(n_mixes):
            loc = torch.randn(dim)*3
            covariance = torch.diag(torch.rand(dim) + min_cov)
            self.distributions.append(torch.distributions.multivariate_normal.MultivariateNormal(loc,
                                                                                        covariance_matrix=covariance))

    def log_prob(self, x: torch.tensor) -> torch.tensor:
        # log p(x, m) = log p(x | m ) + log p(m) - assume uniform prior so p(m) is 0.5
        log_prob_concat = torch.stack([distribution.log_prob(x) for distribution in self.distributions]) \
                          + torch.log(torch.tensor(1/self.n_mixes))
        log_prob_concat = torch.clamp_min(log_prob_concat, -1e1)  # prevent -inf
        log_prob_concat[torch.isnan(log_prob_concat)] = -1e1    # prevent nan
        # log p(x) = log( sum_M p(x, m) )
        return torch.logsumexp(log_prob_concat, dim=0)

    @torch.no_grad()
    def sample(self, sample_shape=(1,)) -> torch.tensor:
        # for simplicity just assume sample_shape is always of form (n,)
        # rand=1 corresponds to sampling from distribution for that point
        # n_mixes by n_samples
        mixture_samples = torch.zeros(self.n_mixes, sample_shape[0])
        random_selection_indices = torch.randint(self.n_mixes, size=sample_shape)
        mixture_samples[random_selection_indices, torch.arange(sample_shape[0])] = 1
        # n_mixes by n_samples by sample dim
        samples_all_dist = torch.stack([distribution.sample(sample_shape) for distribution in self.distributions])
        samples = torch.einsum("ij,ijk->jk", mixture_samples, samples_all_dist)
        self.check(mixture_samples, samples, samples_all_dist)
        return samples

    def check(self, mixture_samples, samples, samples_all_dist):
        for point_n in [0,1,2]:
            chosen_dist = 1 == mixture_samples[:, point_n]
            sample_n_from_chosen_dist = samples_all_dist[chosen_dist, point_n, :]
            assert torch.sum(sample_n_from_chosen_dist != samples[point_n, :]) == 0



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Utils import plot_distribution
    size = 2
    dist = MoG(size)
    samples = dist.sample((10,)) #torch.randn((10,size))
    log_probs = dist.log_prob(samples)
    print(log_probs)
    print(log_probs.shape)
    plot_distribution(dist, n_points=300, range=15)
    plt.show()
    print(dist.sample((10,)))
