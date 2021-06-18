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
    def calculate_expectation(self, n_samples: int, expectation_function)\
            -> (torch.tensor, dict):
        x_samples, normalised_sampling_weights = self.generate_samples_and_weights(n_samples=n_samples)
        expectation_func_x = expectation_function(x_samples)
        expectation = torch.sum(normalised_sampling_weights*expectation_func_x)
        effective_sample_size = self.effective_sample_size(normalised_sampling_weights)
        info_dict = {"effective_sample_size": effective_sample_size,
                     "normalised_sampling_weights": normalised_sampling_weights.detach()}
        return expectation, info_dict

    @torch.no_grad()
    def generate_samples_and_weights(self, n_samples:int=1000, drop_nan_and_infs=True):
        x_samples, log_q_x = self.sampling_distribution(n_samples)
        if drop_nan_and_infs:
            contains_neg_infs = torch.isinf(log_q_x) | torch.isnan(log_q_x)
            n_valid_samples = torch.sum(~contains_neg_infs)
            log_q_x = torch.masked_select(log_q_x, ~contains_neg_infs)
            x_flat = torch.masked_select(x_samples, ~contains_neg_infs[:, None].repeat(1, x_samples.shape[-1]))
            x_samples = x_flat.unflatten(dim=0, sizes=(n_valid_samples, x_samples.shape[-1]))
        log_p_x = self.target_distribution.log_prob(x_samples)
        log_w = log_p_x - log_q_x
        if drop_nan_and_infs:
            contains_neg_infs = (torch.isinf(log_w) | torch.isnan(log_w)) & (log_w < torch.tensor(0.0))
            n_valid_samples = torch.sum(~contains_neg_infs)
            log_w = torch.masked_select(log_w, ~contains_neg_infs)
            x_flat = torch.masked_select(x_samples, ~contains_neg_infs[:, None].repeat(1, x_samples.shape[-1]))
            x_samples = x_flat.unflatten(dim=0, sizes=(n_valid_samples, x_samples.shape[-1]))
        normalised_sampling_weights = F.softmax(log_w, dim=-1)
        return x_samples, normalised_sampling_weights

if __name__ == '__main__':
    from TargetDistributions.Guassian_FullCov import Guassian_FullCov
    #from FittedModels.Models.DiagonalGaussian import DiagonalGaussian
    # sampling_dist = DiagonalGaussian(size)
    from FittedModels.Models.FlowModel import FlowModel
    size = 5
    n_samples = 10000
    target_dist = Guassian_FullCov(size)
    sampling_dist = FlowModel(size, scaling_factor=0.01)

    importance_sampler = VanillaImportanceSampling(sampling_dist, target_dist)
    expectation, info = importance_sampler.calculate_expectation(n_samples, expectation_function=lambda x: torch.sum(x))
    print(f"calculated expectation is {expectation}")

    x = target_dist.sample((10,))
    sampling_dist.log_prob(x)

