import torch

torch.manual_seed(5)
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
from FittedModels.utils.plotting_utils import plot_distributions
from FittedModels.train import LearntDistributionManager
from Utils.numerical_utils import quadratic_function as expectation_function
from FittedModels.Models.FlowModel import FlowModel
from TargetDistributions.MoG import Triangle_MoG

if __name__ == '__main__':
    epochs = 1000
    batch_size = int(1e2)
    dim = 2
    n_samples_estimation = int(1e6)
    KPI_n_samples = int(1e6)

    torch.manual_seed(0) # 0 breaks it within 1000 epochs
    target = Triangle_MoG(loc_scaling=5, cov_scaling=1)
    torch.manual_seed(1)
    learnt_sampler = FlowModel(x_dim=dim, n_flow_steps=3, scaling_factor=5.0) #, flow_type="RealNVP")
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type="kl", lr=1e-3, annealing=True)
    expectation_before, info_before = tester.estimate_expectation(n_samples_estimation, expectation_function)
    width = 8
    fig_before_train = plot_distributions(tester, bounds=[[-width, width], [-width, width]])