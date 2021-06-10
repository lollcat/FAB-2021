import torch
from FittedModels.utils.plotting_utils import plot_samples, plot_sampling_info, plot_divergences, plot_history
torch.manual_seed(5)
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
from FittedModels.utils.plotting_utils import plot_distributions
from FittedModels.train import LearntDistributionManager
from Utils.plotting_utils import plot_distribution
from Utils.numerical_utils import MC_estimate_true_expectation
from Utils.numerical_utils import quadratic_function as expectation_function
from FittedModels.Models.FlowModel import FlowModel
import matplotlib.pyplot as plt
from TargetDistributions.MoG import Difficult_MoG

if __name__ == '__main__':
    torch.manual_seed(0)  # 0 breaks it within 1000 epochs
    target = Difficult_MoG(loc_scaling=15, cov_scaling=1)
    width = 8
    plot_distribution(target, bounds=[[-width, width], [-width, width]])
    plt.show()
    torch.manual_seed(1)
    true_expectation = MC_estimate_true_expectation(target, expectation_function, int(1e3))  # int(1e6)

    epochs = 3000
    batch_size = int(1e2)
    dim = 2
    n_samples_estimation = int(1e6)
    KPI_n_samples = int(1e4)


    learnt_sampler = FlowModel(x_dim=dim, n_flow_steps=3, scaling_factor=15.0)  # , flow_type="RealNVP")
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type="DReG", lr=1e-4)
    plot_samples(tester)
    plt.show()
    expectation_before, info_before = tester.estimate_expectation(n_samples_estimation, expectation_function)

    history = tester.train(epochs, batch_size=batch_size, KPI_batch_size=KPI_n_samples,
                           clip_grad_norm=True, max_grad_norm=1)

    expectation, info = tester.estimate_expectation(n_samples_estimation, expectation_function)
    print(f"True expectation estimate is {true_expectation} \n"
          f"estimate before training is {expectation_before} \n"
          f"estimate after training is {expectation} \n"
          f"effective sample size before is {info_before['effective_sample_size'] / n_samples_estimation}\n"
          f"effective sample size after train is {info['effective_sample_size'] / n_samples_estimation}\n"
          f"variance in weights is {torch.var(info['normalised_sampling_weights'])}")

    plot_samples(tester)
    plt.show()
    plot_sampling_info(history)
    plt.show()
    plot_history(history)
    plt.show()
    width = 8
    fig_after_train = plot_distributions(tester, bounds=[[-width, width], [-width, width]])
    plot_divergences(history)
    plt.show()




