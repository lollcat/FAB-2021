import torch
from FittedModels.utils import plot_samples
torch.manual_seed(5)
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
from TargetDistributions.MoG import custom_MoG
from FittedModels.utils import plot_distributions
from FittedModels.train import LearntDistributionManager
from Utils.plotting_utils import plot_distribution, expectation_function
from FittedModels.Models.FlowModel import FlowModel
from FittedModels.utils import plot_history
import matplotlib.pyplot as plt
import torch


if __name__ == '__main__':
    torch.manual_seed(2)
    torch.set_default_dtype(torch.float64)
    epochs = 2000
    # batch size makes a difference
    batch_size = int(1e3)
    dim = 2
    n_samples_estimation = int(1e4)
    #target = MoG(dim=dim, n_mixes=2, min_cov=1, loc_scaling=10)
    target = custom_MoG(locs_=[-10, 10])
    fig = plot_distribution(target, bounds=[[-20, 20], [-20, 20]])
    torch.manual_seed(0)
    learnt_sampler = FlowModel(x_dim=dim, n_flow_steps=3, scaling_factor=10) #, flow_type="RealNVP")
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type="DReG", lr=1e-3)
    expectation_before, sampling_weights_before = tester.estimate_expectation(n_samples_estimation, expectation_function)
    samples_before = plot_samples(tester, n_samples=batch_size)
    plt.show()
    fig_before_train = plot_distributions(tester, bounds=[[-20, 20], [-20, 20]])
    plt.show()
    history = tester.train(epochs, batch_size=batch_size, clip_grad_norm=True)
    samples_after = plot_samples(tester)
    plt.show()
    plot_history(history)
    plt.show()
    #fig_after_train = plot_distributions(tester, bounds=[[-20, 20], [-20, 20]])
    #plt.show()
