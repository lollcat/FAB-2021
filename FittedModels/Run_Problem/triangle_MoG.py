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
from TargetDistributions.MoG import Triangle_MoG
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':
    torch.manual_seed(0)  # 0 breaks it within 1000 epochs
    target = Triangle_MoG(loc_scaling=5, cov_scaling=1)
    true_expectation = MC_estimate_true_expectation(target, expectation_function, int(1e6))
    print(true_expectation)
    print(MC_estimate_true_expectation(target, expectation_function,
                                       int(1e6)))  # print twice to make sure estimates are resonably close
    width = 8
    fig = plot_distribution(target, bounds=[[-width, width], [-width, width]])

    epochs = int(1e4)
    lr = 5e-4
    weight_decay = 1e-3
    batch_size = int(1e2)
    dim = 2
    n_flow_steps = 64
    n_samples_estimation = int(1e6)
    KPI_n_samples = int(1e4)
    flow_type = "RealNVP"
    loss_type = "DReG" # "kl"
    def plotter(*args, **kwargs):
        # wrap plotting function like this so it displays during training
        plot_samples(*args, **kwargs)
        plt.show()

    torch.manual_seed(1)
    learnt_sampler = FlowModel(x_dim=dim, n_flow_steps=n_flow_steps, scaling_factor=4.0, flow_type=flow_type)
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type=loss_type,
                                       lr=lr, weight_decay=weight_decay)
    # unstable with lr=1e-3
    expectation_before, info_before = tester.estimate_expectation(n_samples_estimation, expectation_function)

    width = 8
    fig_before_train = plot_distributions(tester, bounds=[[-width, width], [-width, width]])

    samples_fig_before = plot_samples(tester)

    history = tester.train(epochs, batch_size=batch_size, KPI_batch_size=KPI_n_samples, intermediate_plots=True,
                           plotting_func=plotter)
                           #clip_grad_norm=True, max_grad_norm=1, intermediate_plots=True)

    tester.learnt_sampling_dist.check_forward_backward_consistency()

    expectation, info = tester.estimate_expectation(n_samples_estimation, expectation_function)
    print(f"True expectation estimate is {true_expectation} \n"
          f"estimate before training is {expectation_before} \n"
          f"estimate after training is {expectation} \n"
          f"effective sample size before is {info_before['effective_sample_size'] / n_samples_estimation}\n"
          f"effective sample size after train is {info['effective_sample_size'] / n_samples_estimation}\n"
          f"variance in weights is {torch.var(info['normalised_sampling_weights'])}")

    plot_samples(tester)
    plt.show()
    plot_divergences(history)
    plt.show()
    plot_sampling_info(history)
    plt.show()
    plot_history(history)
    plt.show()

    width = 8
    fig_after_train = plot_distributions(tester, bounds=[[-width, width], [-width, width]])




