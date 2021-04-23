import torch
from TargetDistributions.base import BaseTargetDistribution

class MoG(BaseTargetDistribution):
    def __init__(self, dim=5):
        super(MoG, self).__init__()
        self.dim=dim
        self.loc1 = torch.ones(dim) * -2
        self.loc2 = torch.ones(dim) * 2
        self.covariance = torch.eye(dim)

        self.distribution1 = torch.distributions.multivariate_normal.MultivariateNormal(self.loc1,
                                                                                        covariance_matrix=self.covariance)
        self.distribution2 = torch.distributions.multivariate_normal.MultivariateNormal(self.loc2,
                                                                                        covariance_matrix=self.covariance)

    def log_prob(self, x: torch.tensor) -> torch.tensor:
        # log p(x, m) = log p(x | m ) + log p(m) - assume uniform prior so p(m) is 0.5
        log_prob_concat = torch.stack([self.distribution1.log_prob(x) + torch.log(torch.tensor([0.5])),
                                       self.distribution2.log_prob(x) + torch.log(torch.tensor([0.5]))])
        # log p(x) = log( sum_M p(x, m) )
        return torch.logsumexp(log_prob_concat, dim=0)

    def sample(self, sample_shape=(1,)):
        # for simplicity just assume sample_shape is always of form (n,)
        # rand=1 corresponds to sampling from distributino 1 for that point
        rand = torch.randint(1+1, size=sample_shape)
        rand = rand[:, None].repeat([1, self.dim])
        return self.distribution1.sample(sample_shape)*rand + self.distribution2.sample(sample_shape)*(1-rand)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Utils import plot_distribution
    size = 2
    dist = MoG(size)
    samples = torch.randn((10,size))
    log_probs = dist.log_prob(samples)
    print(log_probs)
    print(log_probs.shape)
    plot_distribution(dist, n_points=100, range=10)
    plt.show()
    print(dist.sample((10,)))