import torch
from FittedModels.utils import plot_samples

torch.manual_seed(5)
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
from FittedModels.train import LearntDistributionManager
from Utils.plotting_utils import plot_distribution
from Utils.numerical_utils import MC_estimate_true_expectation
from Utils.numerical_utils import quadratic_function as expectation_function
from FittedModels.Models.FlowModel import FlowModel
import matplotlib.pyplot as plt
from TargetDistributions.MoG import custom_MoG

if __name__ == '__main__':
    torch.manual_seed(6)
    epochs = 1000
    dim = 2
    n_samples_estimation = int(1e4)
    #target = MoG(dim=dim, n_mixes=2, min_cov=0.01, loc_scaling=0.8)
    target = custom_MoG(dim=dim, cov_scaling=0.3)
    fig = plot_distribution(target, bounds=[[-5, 5], [-5, 5]])
    true_expectation = MC_estimate_true_expectation(target, expectation_function, int(1e6))

    torch.manual_seed(0)
    learnt_sampler = FlowModel(x_dim=dim , n_flow_steps=3) #, flow_type="RealNVP")
    # kl with annealing
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type="kl",
                                       annealing=True)
    # DReG_kl
    #tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type="DReG", k=None)
    expectation_before, info_before = tester.estimate_expectation(n_samples_estimation, expectation_function)
    samples_before = plot_samples(tester, n_samples=int(1e4))
    plt.show()
    history = tester.train(epochs, batch_size=int(1e3), clip_grad_max=False) #True)
    samples_fig_after = plot_samples(tester)
    plt.show()

    tester.setup_loss("DReG", alpha=2, k=None, new_lr=1e-4)
    history2 = tester.train(epochs, batch_size=int(1e3), clip_grad_max=False)  # True)


