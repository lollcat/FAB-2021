import torch

torch.manual_seed(5)
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
from TargetDistributions.MoG import MoG
from FittedModels.utils.plotting_utils import plot_distributions
from FittedModels.train import LearntDistributionManager
from Utils.plotting_utils import plot_func2D
from FittedModels.Models.FlowModel import FlowModel
from FittedModels.utils.plotting_utils import plot_history
import matplotlib.pyplot as plt
import torch

# setup expectation function
def expectation_function(x):
    A = torch.ones((x.shape[-1], x.shape[-1]))
    return torch.einsum("bi,ij,bj->b", x, A, x)
expectation_func_fig = plot_func2D(expectation_function, n_points=200, range=15)

if __name__ == '__main__':
    torch.manual_seed(2)
    epochs = 1000
    dim = 2
    n_samples_estimation = int(1e4)
    target = MoG(dim=dim, n_mixes=2, min_cov=2)

    torch.manual_seed(0)
    learnt_sampler = FlowModel(x_dim=dim, n_flow_steps=3, flow_type="RealNVP") #
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type="kl") #"DReG"
    fig_before = plot_distributions(tester, range=15)
    # expectation_before, sampling_weights_before = tester.estimate_expectation(n_samples_estimation, expectation_function)

    history = tester.train(epochs, batch_size=200)
    plot_history(history)
    fig_after_train = plot_distributions(tester, range=15)
    plt.show()

