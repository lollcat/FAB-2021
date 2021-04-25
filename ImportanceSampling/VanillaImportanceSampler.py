import torch
import torch.nn.functional as F
from ImportanceSampling.base import BaseImportanceSampler
from FittedModels.Models.base import BaseLearntDistribution

class VanillaImportanceSampling(BaseImportanceSampler):
    def __init__(self, sampling_distribution, target_distribution):
        super(VanillaImportanceSampling, self).__init__()
        self.sampling_distribution: BaseLearntDistribution
        self.sampling_distribution = sampling_distribution
        self.target_distribution: torch.distributions.Distribution
        self.target_distribution = target_distribution

    @torch.no_grad()
    def calculate_expectation(self, n_samples:int=1000, expectation_function=lambda x: torch.sum(x, dim=-1))\
            -> (torch.tensor, dict):
        x_samples, normalised_sampling_weights = self.generate_samples_and_weights(n_samples=n_samples)
        expectation_func_x = expectation_function(x_samples)
        expectation = torch.sum(normalised_sampling_weights*expectation_func_x)
        effective_sample_size = self.effective_sample_size(normalised_sampling_weights)
        info_dict = {"effective_sample_size": effective_sample_size,
                     "normalised_sampling_weights": normalised_sampling_weights.detach()}
        return expectation, info_dict

    @torch.no_grad()
    def generate_samples_and_weights(self, n_samples:int=1000):
        x_samples, log_q_x = self.sampling_distribution(n_samples)
        log_p_x = self.target_distribution.log_prob(x_samples)
        if True in torch.isnan(log_p_x) or True in torch.isinf(log_p_x):
            print("Nan encountered in importance weights")
            log_p_x[torch.isnan(log_p_x)] = -1e6
            log_p_x[torch.isinf(log_p_x)] = -1e6
        normalised_sampling_weights = F.softmax(log_p_x - log_q_x, dim=-1)
        return x_samples, normalised_sampling_weights

if __name__ == '__main__':
    from TargetDistributions.Guassian_FullCov import Guassian_FullCov
    from FittedModels.Models.DiagonalGaussian import DiagonalGaussian
    size = 5
    n_samples = 10000
    target_dist = Guassian_FullCov(size)
    sampling_dist = DiagonalGaussian(size)
    importance_sampler = VanillaImportanceSampling(sampling_dist, target_dist)
    expectation, info = importance_sampler.calculate_expectation(n_samples)
    true_expectation = torch.sum(target_dist.mean)
    print(f"calculated expectation is {expectation} \n true expectation is {true_expectation}")

