import torch
from FittedModels.Utils.plotting_utils import plot_samples

torch.manual_seed(5)
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
from FittedModels.train import LearntDistributionManager
from FittedModels.Utils.plotting_utils import plot_history
from Utils.plotting_utils import plot_distribution
from Utils.numerical_utils import MC_estimate_true_expectation
from Utils.numerical_utils import quadratic_function as expectation_function
from FittedModels.Models.FlowModel import FlowModel
import matplotlib.pyplot as plt
from TargetDistributions.MoG import MoG


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)  # 0 breaks it within 1000 epochs
    dim = 2
    epoch = 1000
    batch_size = int(1e3)
    n_samples_estimation = int(1e4)
    target = MoG(dim=dim, n_mixes=10, min_cov=0, loc_scaling=3)
    fig = plot_distribution(target, bounds=[[-5, 5], [-5, 5]])
    true_expectation = MC_estimate_true_expectation(target, expectation_function, int(1e6))
    torch.manual_seed(1)
    learnt_sampler = FlowModel(x_dim=dim, n_flow_steps=5, scaling_factor=4.0)
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type="DReG",
                                       lr=1e-3, weight_decay=1e-5)
    expectation_before, info_before = tester.estimate_expectation(n_samples_estimation, expectation_function)
    samples_fig_before = plot_samples(tester)
    plt.show()
    history = tester.train(epoch, batch_size=batch_size, clip_grad_norm=True, max_grad_norm=1)
    expectation, info = tester.estimate_expectation(n_samples_estimation, expectation_function)
    samples_fig_after = plot_samples(tester)
    plt.show()
    print(f"True expectation estimate is {true_expectation} \n"
          f"estimate before training is {expectation_before} \n"
          f"estimate after training is {expectation} \n"
          f"effective sample size before is {info_before['effective_sample_size']} out of {n_samples_estimation}\n"
          f"effective sample size after train is {info['effective_sample_size']}  out of {n_samples_estimation}\n"
          f"variance in weights is {torch.var(info['normalised_sampling_weights'])}")

    plot_history(history)
    plt.show()
