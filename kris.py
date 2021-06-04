"""
Given z, optimize prior variances of auxilliary variables such that their kl is equal to 5
"""
import torch
import torch.nn as nn

class Kris(nn.Module):
    def __init__(self, dim=5, b_k=torch.zeros(5), previous_sigmas=torch.zeros(3), z=torch.zeros(5), total_variance=1,
                 target_kl=5, sigma_k_max=1):
        super(Kris, self).__init__()
        self.dim = dim
        assert b_k.shape[-1] == dim; assert z.shape[0] == dim
        self.register_buffer("prior_loc", torch.zeros(dim))
        self.register_buffer("prior_covariance", torch.eye(dim))
        self.untransformed_sigma_k = nn.Parameter(torch.tensor(-2.0))
        self.register_buffer("sigma_k_max", torch.tensor(sigma_k_max))
        self.register_buffer("posterior_mean_vector_part", z - b_k)
        self.register_buffer("posterior_s_k_minus_1", total_variance - torch.sum(previous_sigmas))
        self.register_buffer("target_kl", torch.tensor(target_kl))


    @property
    def prior(self):
        return torch.distributions.multivariate_normal.MultivariateNormal(self.prior_loc,
                                                                          self.prior_covariance*self.sigma_k)

    @property
    def sigma_k(self):
        return torch.sigmoid(self.untransformed_sigma_k)*self.sigma_k_max

    @property
    def posterior(self):
        s_k = self.posterior_s_k_minus_1 - self.sigma_k
        loc = self.posterior_mean_vector_part*(self.sigma_k/self.posterior_s_k_minus_1)
        cov = s_k*self.sigma_k/self.posterior_s_k_minus_1*torch.eye(self.dim)
        return torch.distributions.multivariate_normal.MultivariateNormal(loc,
                                                                          cov)

    def loss(self):
        return (self.kl() - self.target_kl)**2

    def kl(self):
        return torch.mean(torch.distributions.kl.kl_divergence(self.posterior, self.prior))





