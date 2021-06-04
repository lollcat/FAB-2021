import torch
import torch.nn.functional as F
from ImportanceSampling.base import BaseImportanceSampler
from collections.abc import Callable
import numpy as np


class AnnealedImportanceSampler(BaseImportanceSampler):
    """
    Sample from p_0 (sampling distribution) through a chain of intermediate distributions,
    to the target distribution p_N
    f_n(x) = p_0 ^ (1 - bt) + p_N ^ bt
    where 0 = b_0 < b_1 ... < b_d = 1
    """
    def __init__(self, sampling_distribution, target_distribution,
                 n_distributions=200, n_updates_Metropolis=10, save_for_visualisation=True, save_spacing=20,
                 distribution_spacing="geometric", noise_scaling=1.0):
        self.noise_scaling = noise_scaling
        self.sampling_distribution = sampling_distribution
        self.target_distribution = target_distribution
        self.n_distributions = n_distributions
        self.n_updates_Metropolis = n_updates_Metropolis
        n_linspace_points = int(n_distributions/5) # rough heuristic, copying ratio used in example in AIS paper
        n_geomspace_points = n_distributions - n_linspace_points
        if distribution_spacing == "geometric":
            self.B_space = torch.tensor(list(np.linspace(0, 0.01, n_linspace_points)) +
                                        list(np.geomspace(0.01, 1, n_geomspace_points)))
        elif distribution_spacing == "linear":
            self.B_space = torch.linspace(0, 1, n_distributions)
        else:
            raise Exception(f"distribution spacing incorrectly specified: '{distribution_spacing}',"
                            f"options are 'geometric' or 'linear'")

        self.save_for_visualisation = save_for_visualisation
        if self.save_for_visualisation:
            self.save_spacing = save_spacing
            self.log_w_history = []
            self.samples_history = []

    @property
    def device(self):
        return next(self.sampling_distribution.parameters()).device

    def run(self, n_runs):
        log_w = torch.zeros(n_runs).to(self.device)  # log importance weight
        x_new, log_prob_p0 = self.sampling_distribution(n_runs)
        log_w += self.intermediate_unnormalised_log_prob(x_new, 1) - log_prob_p0
        for j in range(1, self.n_distributions-1):
            x_new = self.Metropolis_transition(x_new, j)
            log_w += self.intermediate_unnormalised_log_prob(x_new, j+1) - \
                     self.intermediate_unnormalised_log_prob(x_new, j)
            if self.save_for_visualisation:
                if (j+1) % self.save_spacing == 0:
                    self.log_w_history.append(log_w)
                    self.samples_history.append(x_new)
        return x_new, log_w


    def Metropolis_transition(self, x, j):
        for n in range(self.n_updates_Metropolis):
            x_proposed = x + torch.randn(x.shape).to(x.device) * self.noise_scaling
            x_proposed_log_prob = self.intermediate_unnormalised_log_prob(x_proposed, j)
            x_prev_log_prob = self.intermediate_unnormalised_log_prob(x, j)
            acceptance_probability = torch.exp(x_proposed_log_prob - x_prev_log_prob)
            # not that sometimes this will be greater than one, corresonding to 100% probability of acceptance
            accept = (acceptance_probability > torch.rand(acceptance_probability.shape).to(x.device)).int()
            accept = accept[:, None].repeat(1, x.shape[-1])
            x = accept*x_proposed + (1-accept)*x
        return x


    def intermediate_unnormalised_log_prob(self, x, j):
        # j is the step of the algorithm, and corresponds which intermediate distribution that we are sampling from
        # j = 0 is the sampling distribution, j=N is the target distribution
        beta = self.B_space[j]
        return (1-beta)*self.sampling_distribution.log_prob(x) + beta*self.target_distribution.log_prob(x)


    def calculate_expectation(self, n_samples: int, expectation_function)\
            -> (torch.tensor, dict):
        samples, log_w = self.run(n_runs=n_samples)
        normalised_importance_weights = F.softmax(log_w, dim=-1)
        function_values = expectation_function(samples)
        expectation = normalised_importance_weights.T @ function_values
        effective_sample_size = self.effective_sample_size(normalised_importance_weights)
        info_dict = {"effective_sample_size": effective_sample_size,
                     "normalised_sampling_weights": normalised_importance_weights.detach(),
                     "samples": samples.detach()}
        return expectation, info_dict



if __name__ == '__main__':
    from TargetDistributions.Guassian_FullCov import Guassian_FullCov
    from FittedModels.Models.DiagonalGaussian import DiagonalGaussian
    from Utils.numerical_utils import expectation_function, MC_estimate_true_expectation
    dim = 5
    target = Guassian_FullCov(dim=dim)
    learnt_sampler = DiagonalGaussian(dim=dim)
    test = AnnealedImportanceSampler(sampling_distribution=learnt_sampler, target_distribution=target)
    true_expectation = MC_estimate_true_expectation(target, expectation_function, int(1e4))
    expectation, info_dict = test.calculate_expectation(5000, expectation_function=expectation_function)
    print(true_expectation, expectation)
