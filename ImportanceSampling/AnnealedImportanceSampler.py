import torch
import torch.nn.functional as F
from ImportanceSampling.base import BaseImportanceSampler

class AnnealedImportanceSampler(BaseImportanceSampler):
    """
    Sample from p_0 (sampling distribution) through a chain of intermediate distributions,
    to the target distribution p_N
    f_n(x) = p_0 ^ (1 - bt) + p_N ^ bt
    where 0 = b_0 < b_1 ... < b_d = 1
    """
    def __init__(self, sampling_distribution, target_distribution,
                 n_distributions=200, save_for_visualisation=True, save_spacing=20):
        self.sampling_distribution = sampling_distribution
        self.target_distribution = target_distribution
        self.n_distributions = n_distributions
        self.B_space = torch.linspace(0, 1, n_distributions)  # TODO update to geometric spacing

        self.save_for_visualisation = save_for_visualisation
        if self.save_for_visualisation:
            self.save_spacing = save_spacing
            self.log_w_history = []
            self.samples_history = []

    @torch.no_grad()
    def run(self, n_runs):
        log_w = torch.zeros(n_runs)  # log importance weight
        x_new = self.sampling_distribution.sample((n_runs, ))
        log_w += self.intermediate_unnormalised_log_prob(x_new, 1) - \
                     self.intermediate_unnormalised_log_prob(x_new, 0)
        for j in range(1, self.n_distributions-1):
            x_new = self.Metropolis_transition(x_new, j)
            log_w += self.intermediate_unnormalised_log_prob(x_new, j+1) - \
                     self.intermediate_unnormalised_log_prob(x_new, j)
            if self.save_for_visualisation:
                if (j+1) % self.save_spacing == 0:
                    self.log_w_history.append(log_w)
                    self.samples_history.append(x_new)
        return x_new, log_w


    def Metropolis_transition(self, x, j, n_updates=10):
        for n in range(n_updates):
            x_proposed = x + torch.randn(x.shape)
            x_proposed_log_prob = self.intermediate_unnormalised_log_prob(x_proposed, j)
            x_prev_log_prob = self.intermediate_unnormalised_log_prob(x, j)
            acceptance_probability = torch.exp(x_proposed_log_prob - x_prev_log_prob)
            # not that sometimes this will be greater than one, corresonding to 100% probability of acceptance
            accept = (acceptance_probability > torch.rand(acceptance_probability.shape)).int()
            accept = accept[:, None].repeat(1, x.shape[-1])
            x = accept*x_proposed + (1-accept)*x
        return x


    def intermediate_unnormalised_log_prob(self, x, j):
        # j is the step of the algorithm, and corresponds which intermediate distribution that we are sampling from
        # j = 0 is the sampling distribution, j=N is the target distribution
        beta = self.B_space[j]
        return (1-beta)*self.sampling_distribution.log_prob(x) + beta*self.target_distribution.log_prob(x)


    def calculate_expectation(self, n_samples: int = 1000, expectation_function=lambda x: torch.sum(x, dim=-1)) \
            -> (torch.tensor, dict):
        samples, log_w = self.run(n_runs=n_samples)
        normalised_importance_weights = F.softmax(log_w, dim=-1)
        function_values = expectation_function(samples)
        expectation = normalised_importance_weights.T @ function_values
        effective_sample_size = self.effective_sample_size(normalised_importance_weights)
        info_dict = {"effective_sample_size": effective_sample_size,
                     "normalised_sampling_weights": normalised_importance_weights,
                     "samples": samples}
        return expectation, info_dict



if __name__ == '__main__':
    from TargetDistributions.Guassian_FullCov import Guassian_FullCov
    from FittedModels.Models.DiagonalGaussian import DiagonalGaussian
    dim = 5
    target = Guassian_FullCov(dim=dim)
    learnt_sampler = DiagonalGaussian(dim=dim)
    test = AnnealedImportanceSampler(sampling_distribution=learnt_sampler, target_distribution=target)
    true_expectation = torch.sum(test.target_distribution.mean)
    expectation, info_dict = test.calculate_expectation(5000)
    print(true_expectation, expectation)
