import torch
from FittedModels.utils.plotting_utils import plot_samples
torch.manual_seed(5)
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
from TargetDistributions.Guassian_FullCov import Guassian_FullCov
from FittedModels.train import LearntDistributionManager
from Utils.plotting_utils import plot_distribution
from FittedModels.Models.FlowModel import FlowModel
from FittedModels.utils.plotting_utils import plot_history
import matplotlib.pyplot as plt
import torch


# setup expectation function
def expectation_function(x):
    A = torch.ones((x.shape[-1], x.shape[-1]))
    return torch.einsum("bi,ij,bj->b", x, A, x)

if __name__ == '__main__':
    """
    Hypotheses:
    1. Caused my extreme points making gradient unstable?
     - seems like this was actually just a result of incorrectly fixing the last bug
    """
    torch.manual_seed(0)
    epochs = 130 # 130
    batch_size = 100
    dim = 2
    n_samples_estimation = int(1e4)
    target = Guassian_FullCov(dim=dim, scale_covariance=1)
    fig = plot_distribution(target, bounds=[[-20, 20], [-20, 20]])

    torch.manual_seed(0)
    learnt_sampler = FlowModel(x_dim=dim, n_flow_steps=3)
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type="kl")
    expectation_before, sampling_weights_before = tester.estimate_expectation(n_samples_estimation, expectation_function)

    samples_fig_before = plot_samples(tester)
    history = tester.train(epochs, batch_size=batch_size)
    samples_fig_after = plot_samples(tester)
    plt.show()
    plot_history(history)
    plt.show()



