import torch
from FittedModels.utils import plot_samples, plot_sampling_info, plot_divergences
torch.manual_seed(5)
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
from FittedModels.train import LearntDistributionManager
from Utils.numerical_utils import MC_estimate_true_expectation, expectation_function
from FittedModels.Models.FlowModel import FlowModel
from FittedModels.utils import plot_history
import matplotlib.pyplot as plt
from TargetDistributions.MoG import MoG

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    dim = 10
    n_samples_estimation = int(1e4)
    target = MoG(dim=dim, n_mixes=10, min_cov=0, loc_scaling=3)
    true_expectation = MC_estimate_true_expectation(target, expectation_function, int(1e6))
    torch.manual_seed(1)
    learnt_sampler = FlowModel(x_dim=dim, n_flow_steps=5, scaling_factor=10.0, flow_type="RealNVP")
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type="DReG", lr=1e-3)
    expectation_before, info_before = tester.estimate_expectation(n_samples_estimation, expectation_function)

    samples_fig_before = plot_samples(tester)  # this just looks at 2 dimensions
    plt.show()
    history = tester.train(1000, batch_size=int(1e3), clip_grad=True, max_grad_norm=1)
    samples_fig_AFTER = plot_samples(tester)  # this just looks at 2 dimensions
    plt.show()
    plot_history(history)
    plt.show()
    plot_divergences(history)
    plt.show()
    plot_sampling_info(history)
    plt.show()



