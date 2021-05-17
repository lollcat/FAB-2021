import torch
import matplotlib.pyplot as plt
from FittedModels.utils import plot_distributions, plot_samples
torch.manual_seed(5)
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
from TargetDistributions.MoG import MoG
from TargetDistributions.Guassian_FullCov import Guassian_FullCov
from FittedModels.Models.DiagonalGaussian import DiagonalGaussian
from FittedModels.utils import plot_distributions
from FittedModels.train import LearntDistributionManager
from Utils import plot_func2D, MC_estimate_true_expectation, plot_distribution
from FittedModels.Models.FlowModel import FlowModel
from FittedModels.experimental.train_AIS import AIS_trainer
from FittedModels.utils import plot_history
import matplotlib.pyplot as plt
import torch

# setup expectation function
def expectation_function(x):
    A = torch.ones((x.shape[-1], x.shape[-1]))
    return torch.einsum("bi,ij,bj->b", x, A, x)

if __name__ == '__main__':
    torch.manual_seed(2)
    torch.set_default_dtype(torch.float64)
    epochs = 10000
    batch_size = 100
    dim = 2
    n_samples_estimation = int(1e4)
    target = MoG(dim=dim, n_mixes=2, min_cov=1, loc_scaling=10)
    fig = plot_distribution(target, bounds=[[-20, 20], [-20, 20]])
    torch.manual_seed(0)
    learnt_sampler = FlowModel(x_dim=dim , n_flow_steps=5) # , flow_type="RealNVP"
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type="kl")
    expectation_before, sampling_weights_before = tester.estimate_expectation(n_samples_estimation, expectation_function)
    samples_before = plot_samples(tester)
    plt.show()
    fig_before_train = plot_distributions(tester, bounds=[[-20, 20], [-20, 20]])
    plt.show()
    history = tester.train(epochs, batch_size=batch_size)
    samples_after = plot_samples(tester)
    plt.show()
    plot_history(history)
    plt.show()
    fig_after_train = plot_distributions(tester, bounds=[[-20, 20], [-20, 20]])
    plt.show()
