import torch
torch.set_default_dtype(torch.float64)
import torch.nn.functional as F
import torch.nn as nn
from ImportanceSampling.AnnealedImportanceSampler import AnnealedImportanceSampler as BaseAIS
import numpy as np


class AnnealedImportanceSampler(BaseAIS):
    """
    Sample from p_0 (sampling distribution) through a chain of intermediate distributions,
    to the target distribution p_N
    f_n(x) = p_0 ^ (1 - bt) + p_N ^ bt
    where 0 = b_0 < b_1 ... < b_d = 1
    """
    def __init__(self, loss_type, sampling_distribution, target_distribution,
                 n_distributions=20, distribution_spacing="linear", transition_operator="HMC",
                 transition_operator_kwargs=None):
        # this changes meaning depending on algorithm, for Metropolis it scales noise, for HMC it is step size
        self.dim = sampling_distribution.dim
        self.loss_type = loss_type
        self.sampling_distribution = sampling_distribution
        self.target_distribution = target_distribution
        self.setup_n_distributions(n_distributions=n_distributions, distribution_spacing=distribution_spacing)
        if transition_operator == "HMC":
            from ImportanceSampling.SamplingAlgorithms.HamiltonianMonteCarlo import HMC
            self.transition_operator_class = HMC(dim=self.dim, n_distributions=n_distributions,
                                                 **transition_operator_kwargs)
        else:
            raise NotImplementedError  # "We currently just focus on using HMC, but we do have a version of Metropolis
            # and NUTS that should be easy to introduce as options here

    def to(self, device):
        self.transition_operator_class.to(device)

    def run(self, n_runs):
        log_w = torch.zeros(n_runs).to(self.device)  # log importance weight
        self.sampling_distribution.set_requires_grad(False)
        with torch.no_grad():
            x_new, log_prob_p0 = self.sampling_distribution(n_runs)
            nan_indices = (torch.sum(torch.isnan(x_new) | torch.isinf(x_new), dim=-1) |
                          torch.isinf(log_prob_p0) | torch.isnan(log_prob_p0)).bool()
            n_nan_indices = torch.sum(nan_indices)
            if n_nan_indices != 0:
                print(f"{n_nan_indices} nan encountered in sampling from flow")
                # replace with legit samples
                x_new[nan_indices] = x_new[~nan_indices][:n_nan_indices]
                log_prob_p0[nan_indices] = log_prob_p0[~nan_indices][:n_nan_indices]
            log_w += self.intermediate_unnormalised_log_prob(x_new, 1) - log_prob_p0
        for j in range(1, self.n_distributions-1):
            x_new, log_w = self.perform_transition(x_new, log_w, j)
        self.sampling_distribution.set_requires_grad(True)
        return x_new, log_w

    def perform_transition(self, x_new, log_w, j):
        target_p_x = lambda x: self.intermediate_unnormalised_log_prob(x, j)
        x_new = self.transition_operator_class.run(x_new, target_p_x, j-1)
        log_w = log_w + self.intermediate_unnormalised_log_prob(x_new, j + 1) - \
                 self.intermediate_unnormalised_log_prob(x_new, j)
        return x_new, log_w

    def intermediate_unnormalised_log_prob(self, x, j):
        # j is the step of the algorithm, and corresponds which intermediate distribution that we are sampling from
        # j = 0 is the sampling distribution, j=N is the target distribution
        beta = self.B_space[j]
        return (1-beta)*self.sampling_distribution.log_prob(x) + beta*self.target_distribution.log_prob(x)

    def setup_n_distributions(self, n_distributions, distribution_spacing="geometric"):
        self.n_distributions = n_distributions
        assert self.n_distributions > 1
        if distribution_spacing == "geometric":
            if self.n_distributions == 2:
                print("running without any intermediate distributions")
                self.B_space = np.linspace(0.0, 1.0, 2)
                return
            elif self.n_distributions == 3:
                print("using linear spacing as there is only 1 intermediate distribution")
                self.B_space = np.array([0.0, 0.5, 1.0])
                return
            n_linspace_points = max(int(n_distributions / 5),
                                    2)  # rough heuristic, copying ratio used in example in AIS paper
            n_geomspace_points = n_distributions - n_linspace_points
            self.B_space = torch.tensor(list(np.linspace(0, 0.1-0.1/n_linspace_points, n_linspace_points)) +
                                        list(np.geomspace(0.1, 1, n_geomspace_points)))
        elif distribution_spacing == "linear":
            self.B_space = torch.linspace(0.0, 1.0, n_distributions)
        else:
            raise Exception(f"distribution spacing incorrectly specified: '{distribution_spacing}',"
                            f"options are 'geometric' or 'linear'")


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    from FittedModels.Models.FlowModel import FlowModel
    from TargetDistributions.MoG import MoG
    import matplotlib.pyplot as plt
    torch.manual_seed(2)
    epochs = 1000
    dim = 2
    n_samples_estimation = int(1e4)
    target = MoG(dim=dim, n_mixes=2, min_cov=1, loc_scaling=3)
    learnt_sampler = FlowModel(x_dim=dim, scaling_factor=3.0)
    HMC_transition_operator_args = {}
    test = AnnealedImportanceSampler(loss_type="kl",
                                     sampling_distribution=learnt_sampler,
                                     target_distribution=target, n_distributions=5,
                                     transition_operator="HMC",
                                     transition_operator_kwargs=HMC_transition_operator_args)
    x_new, log_w = test.run(1000)
    x_new = x_new.cpu().detach()
    plt.plot(x_new[:, 0], (x_new[:, 1]), "o")
    plt.show()

    true_samples = target.sample((1000,)).detach().cpu()
    plt.plot(true_samples[:, 0], (true_samples[:, 1]), "o")
    plt.show()
    print(test.transition_operator_class.interesting_info())
