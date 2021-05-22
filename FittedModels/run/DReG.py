from TargetDistributions.MoG import custom_MoG
import matplotlib.pyplot as plt
from FittedModels.utils import plot_distributions, plot_samples
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
from TargetDistributions.MoG import MoG
from TargetDistributions.Guassian_FullCov import Guassian_FullCov
from FittedModels.Models.DiagonalGaussian import DiagonalGaussian
from FittedModels.utils import plot_distributions
from FittedModels.train import LearntDistributionManager
from Utils import plot_func2D, MC_estimate_true_expectation, plot_distribution, expectation_function
from FittedModels.Models.FlowModel import FlowModel
from FittedModels.experimental.train_AIS import AIS_trainer
from FittedModels.utils import plot_history
import matplotlib.pyplot as plt
import torch


if __name__ == '__main__':
    torch.manual_seed(6)
    epochs = 500
    dim = 2
    n_samples_estimation = int(1e4)
    target = custom_MoG(dim=dim, cov_scaling=0.3)
    true_expectation = MC_estimate_true_expectation(target, expectation_function, int(1e3))
    fig = plot_distribution(target, bounds=[[-5, 5], [-5, 5]])
    plt.show()

    torch.manual_seed(0)
    learnt_sampler = FlowModel(x_dim=dim, n_flow_steps=3)  # , flow_type="RealNVP")
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type="DReG", k=None)
    expectation_before, info_before = tester.estimate_expectation(n_samples_estimation, expectation_function)
    samples_before = plot_samples(tester)
    plt.show()
    history = tester.train(epochs, batch_size=int(1e2))
    samples_fig_after = plot_samples(tester)
    plt.show()
    expectation, info = tester.estimate_expectation(n_samples_estimation, expectation_function)

    print(f"True expectation estimate is {true_expectation} \n"
          f"estimate before training is {expectation_before} \n"
          f"estimate after training is {expectation} \n"
          f"effective sample size before is {info_before['effective_sample_size']} \n"
          f"effective sample size after train is {info['effective_sample_size']} \n"
          f"variance in weights is {torch.var(info['normalised_sampling_weights'])}")
    plot_history(history)
    plt.show()

    grads = tester.get_gradients(n_batches=100, batch_size=100)
