from TargetDistributions.DoubleWell import DoubleWellEnergy

import torch

torch.manual_seed(5)
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
from FittedModels.utils import plot_distributions
from FittedModels.train import LearntDistributionManager
from Utils.plotting_utils import plot_distribution, expectation_function
from FittedModels.Models.FlowModel import FlowModel
from FittedModels.utils import plot_history
import matplotlib.pyplot as plt

if __name__ == '__main__':
    target = DoubleWellEnergy(2, a=-0.5, b=-6)

    dist = plot_distribution(target, bounds=[[-3, 3], [-3, 3]], n_points=100)
    plt.show()
    dim = 2
    n_samples_estimation = int(1e3)
    torch.manual_seed(0)  # 0
    learnt_sampler = FlowModel(x_dim=dim, n_flow_steps=5, scaling_factor=2)  # , flow_type="RealNVP"
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type="DReG")
    expectation_before, info_before = tester.estimate_expectation(n_samples_estimation, expectation_function)
    fig_before_train = plot_distributions(tester, bounds=[[-3, 3], [-3, 3]], n_points=100)
    plt.show()
    history = tester.train(10000, batch_size=100)  # epochs 1000

    expectation, info = tester.estimate_expectation(n_samples_estimation, expectation_function)

    print(f"estimate before training is {expectation_before} \n"
          f"estimate after training is {expectation} \n"
          f"effective sample size before is {info_before['effective_sample_size']} out of {n_samples_estimation}\n"
          f"effective sample size after train is {info['effective_sample_size']}  out of {n_samples_estimation}\n"
          f"variance in weights is {torch.var(info['normalised_sampling_weights'])}")
    plot_history(history)
    plt.show()

    fig_after_train = plot_distributions(tester, bounds=[[-3, 3], [-3, 3]], n_points=100)
    plt.show()


