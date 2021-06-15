import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
from ImportanceSampling.AnnealedImportanceSampler import AnnealedImportanceSampler as BaseAIS
import numpy as np


class AnnealedImportanceSampler(BaseAIS):
    """
    Sample from p_0 (sampling distribution) through a chain of intermediate distributions,
    to the target distribution p_N
    f_n(x) = p_0 ^ (1 - bt) + p_N ^ bt
    where 0 = b_0 < b_1 ... < b_d = 1
    """
    def __init__(self, loss_type, train_parameters, sampling_distribution, target_distribution,
                 n_distributions=200, n_steps_transition_operator=10, save_for_visualisation=True, save_spacing=20,
                 distribution_spacing="geometric",
                 step_size=1.0, transition_operator="Metropolis",
                 HMC_inner_steps = 5):
        # this changes meaning depending on algorithm, for Metropolis it scales noise, for HMC it is step size
        self.loss_type = loss_type
        self.sampling_distribution = sampling_distribution
        self.target_distribution = target_distribution
        self.n_distributions = n_distributions
        self.n_steps_transition_operator = n_steps_transition_operator
        n_linspace_points = int(n_distributions / 5)  # rough heuristic, copying ratio used in example in AIS paper
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

        self.train_parameters = train_parameters
        if train_parameters is True:
            self.log_step_size = nn.parameter.Parameter(torch.log(torch.tensor([step_size])))
        else:
            self.log_step_size = torch.log(torch.tensor([step_size]))

        # Metropolis_transition(x, n_updates, p_x_func, noise_scaling)
        if transition_operator == "Metropolis":
            from ImportanceSampling.SamplingAlgorithms.Metropolis import Metropolis_transition
            self.transition_operator = lambda x, j: \
                Metropolis_transition(x=x,
                                      log_p_x_func=lambda x_new: self.intermediate_unnormalised_log_prob(x_new, j),
                                      n_updates=self.n_steps_transition_operator,
                                      noise_scalings=self.step_size)
        elif transition_operator == "HMC":
            from ImportanceSampling.SamplingAlgorithms.HamiltonianMonteCarlo import HMC
            self.transition_operator = lambda x, j: \
                HMC(log_q_x=lambda x_new: self.intermediate_unnormalised_log_prob(x_new, j),
                    epsilon=self.step_size, n_outer=n_steps_transition_operator, L=HMC_inner_steps,
                    current_q=x, grad_log_q_x=None)
        else:
            raise NotImplementedError(f"Sampling method {transition_operator} not implemented")


    def run(self, n_runs):
        log_w = torch.zeros(n_runs).to(self.device)  # log importance weight
        x_new, log_prob_p0 = self.sampling_distribution(n_runs)
        if "DReG" in self.loss_type:
            self.sampling_distribution.set_requires_grad(False)
            log_prob_p0 = self.sampling_distribution.log_prob(x_new)
        log_w += self.intermediate_unnormalised_log_prob(x_new, 1) - log_prob_p0
        for j in range(1, self.n_distributions-1):
            x_new = self.transition_operator(x_new, j)
            log_w += self.intermediate_unnormalised_log_prob(x_new, j+1) - \
                     self.intermediate_unnormalised_log_prob(x_new, j)
            if self.save_for_visualisation:
                if (j+1) % self.save_spacing == 0:
                    self.log_w_history.append(log_w)
                    self.samples_history.append(x_new)
        if "DReG" in self.loss_type:
            self.sampling_distribution.set_requires_grad(True)
        return x_new, log_w

    @property
    def step_size(self):
        return torch.exp(self.log_step_size)



if __name__ == '__main__':
    from FittedModels.Models.FlowModel import FlowModel
    from TargetDistributions.MoG import MoG
    torch.manual_seed(2)
    epochs = 1000
    dim = 2
    n_samples_estimation = int(1e4)
    target = MoG(dim=dim, n_mixes=2, min_cov=1, loc_scaling=3)
    learnt_sampler = FlowModel(x_dim=dim, scaling_factor=3.0)  # , flow_type="RealNVP")
    test = AnnealedImportanceSampler(train_parameters=True, sampling_distribution=learnt_sampler, target_distribution=target)


