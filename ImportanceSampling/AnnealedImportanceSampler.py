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
                 n_distributions=200, n_steps_transition_operator=10, save_for_visualisation=True, save_spacing=20,
                 distribution_spacing="geometric",
                 step_size=1.0, transition_operator="Metropolis", HMC_inner_steps=5):
        # this changes meaning depending on algorithm, for Metropolis it scales noise, for HMC it is step size
        self.step_size = torch.tensor([step_size])
        self.sampling_distribution = sampling_distribution
        self.target_distribution = target_distribution
        self.n_distributions = n_distributions
        self.n_steps_transition_operator = n_steps_transition_operator
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
        if transition_operator == "Metropolis":
            from ImportanceSampling.SamplingAlgorithms.Metropolis import Metropolis
            self.transition_operator_class = Metropolis(n_updates=self.n_steps_transition_operator,
                                      step_size=self.step_size)
        elif transition_operator == "HMC":
            from ImportanceSampling.SamplingAlgorithms.HamiltonianMonteCarlo import HMC
            # TODO
            raise NotImplementedError(f"Sampling method {transition_operator} not implemented")
            self.transition_operator = lambda x, j: \
                HMC(log_q_x=lambda x_new: self.intermediate_unnormalised_log_prob(x_new, j),
                    epsilon=self.step_size, n_outer=n_steps_transition_operator, L=HMC_inner_steps,
                    current_q=x, grad_log_q_x=None)
        elif transition_operator == "NUTS":
            from ImportanceSampling.SamplingAlgorithms.NUTS import NUTS
            raise NotImplementedError(f"Sampling method {transition_operator} not implemented")
        else:
            raise NotImplementedError(f"Sampling method {transition_operator} not implemented")


        self.transition_operator = lambda x, j: \
            self.transition_operator_class.run(
                x=x, log_p_x_func=lambda x_new: self.intermediate_unnormalised_log_prob(x_new, j))

    @property
    def device(self):
        return next(self.sampling_distribution.parameters()).device

    def run(self, n_runs):
        log_w = torch.zeros(n_runs).to(self.device)  # log importance weight
        x_new, log_prob_p0 = self.sampling_distribution(n_runs)
        log_w += self.intermediate_unnormalised_log_prob(x_new, 1) - log_prob_p0
        for j in range(1, self.n_distributions-1):
            x_new = self.transition_operator(x_new, j)
            log_w += self.intermediate_unnormalised_log_prob(x_new, j+1) - \
                     self.intermediate_unnormalised_log_prob(x_new, j)
            if self.save_for_visualisation:
                if (j+1) % self.save_spacing == 0:
                    self.log_w_history.append(log_w)
                    self.samples_history.append(x_new)
        return x_new, log_w


    def intermediate_unnormalised_log_prob(self, x, j):
        # j is the step of the algorithm, and corresponds which intermediate distribution that we are sampling from
        # j = 0 is the sampling distribution, j=N is the target distribution
        beta = self.B_space[j]
        return (1-beta)*self.sampling_distribution.log_prob(x) + beta*self.target_distribution.log_prob(x)


    def calculate_expectation(self, n_samples: int, expectation_function, batch_size=None)\
            -> (torch.tensor, dict):
        if batch_size is None:
            samples, log_w = self.run(n_runs=n_samples)
        else:
            assert n_samples % batch_size == 0.0
            n_batches = int(n_samples / batch_size)
            samples = []
            log_w = []
            for i in range(n_batches):
                sample_batch, log_w_batch = self.run(n_runs=batch_size)
                samples.append(sample_batch.detach())
                log_w.append(log_w_batch.detach())
            samples = torch.cat(samples, dim=0)
            log_w = torch.cat(log_w, dim=0)
        with torch.no_grad():
            normalised_importance_weights = F.softmax(log_w, dim=-1)
            function_values = expectation_function(samples)
            expectation = normalised_importance_weights.T @ function_values
            effective_sample_size = self.effective_sample_size(normalised_importance_weights)
        info_dict = {"effective_sample_size": effective_sample_size.cpu().detach(),
                     "normalised_sampling_weights": normalised_importance_weights.cpu().detach(),
                     "samples": samples.cpu().detach()}
        return expectation, info_dict



if __name__ == '__main__':
    from TargetDistributions.MoG import MoG
    from FittedModels.Models.DiagonalGaussian import DiagonalGaussian
    from Utils.numerical_utils import MC_estimate_true_expectation
    from Utils.numerical_utils import quadratic_function as expectation_function
    import matplotlib.pyplot as plt

    dim = 2
    n_samples = 1000
    target = MoG(dim=dim)
    learnt_sampler = DiagonalGaussian(dim=dim)
    test = AnnealedImportanceSampler(sampling_distribution=learnt_sampler, target_distribution=target,
                                     transition_operator="HMC", n_steps_transition_operator=4,
                                     n_distributions=20)
    true_expectation = MC_estimate_true_expectation(target, expectation_function, int(1e4))
    expectation, info_dict = test.calculate_expectation(n_samples , expectation_function=expectation_function,
                                                        batch_size=10)
    print(true_expectation, expectation)

    sampler_samples = learnt_sampler.sample((n_samples,)).cpu().detach()
    plt.plot(sampler_samples[:, 0], sampler_samples[:, 1], "o")
    plt.title("sampler samples")
    plt.show()
    plt.plot(info_dict["samples"][:, 0], info_dict["samples"][:, 1], "o")
    plt.title("annealed samples")
    plt.show()
    true_samples = target.sample((n_samples, )).cpu().detach()
    plt.plot(true_samples[:, 0], true_samples[:, 1], "o")
    plt.title("true samples")
    plt.show()
