import torch
import matplotlib.pyplot as plt
from FittedModels.utils import plot_distributions
torch.manual_seed(5)
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
from TargetDistributions.Guassian_FullCov import Guassian_FullCov
from FittedModels.Models.DiagonalGaussian import DiagonalGaussian
from FittedModels.utils import plot_distributions, plot_history
from FittedModels.train import LearntDistributionManager
from Utils import expectation_function
from FittedModels.Models.FlowModel import FlowModel
from TargetDistributions.MoG import MoG


def run_experiment(k, dim=2, seed=0, n_samples=int(1e3), epochs=500, batch_size=int(1e4)):
    torch.manual_seed(seed)
    #target = Guassian_FullCov(dim=dim)
    #learnt_sampler = DiagonalGaussian(dim=dim)
    #target = MoG(dim=dim, n_mixes=2, min_cov=1, loc_scaling=3)
    # learnt_sampler = FlowModel(x_dim=dim, n_flow_steps=3, flow_type="RealNVP")
    target = MoG(dim=dim, n_mixes=10, min_cov=0, loc_scaling=1.5)
    learnt_sampler = FlowModel(x_dim=dim, n_flow_steps=3, scaling_factor=2.0)
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type="DReG", k=k)
    try:
        history = tester.train(epochs, batch_size=batch_size)
        expectation, info = tester.estimate_expectation(n_samples)
        return info['effective_sample_size']
    except:
        print(f"didn't run for k={k}")
        return 0

if __name__ == '__main__':
    ESS_list = []
    k_options = [20, 100, 1000, 10000] #, 200, 500, 1000]
    for k in k_options:
        ESS = run_experiment(k)
        ESS_list.append(ESS)

    plt.plot(k_options, ESS_list)
    plt.show()




    """
    k = 1000
    epochs = 1000
    dim = 2
    batch_size = 1000
    n_samples_estimation = int(1e4)
    target = Guassian_FullCov(dim=dim)
    learnt_sampler = DiagonalGaussian(dim=dim)
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type="DReG", k=k)
    fig_before = fig_before_train = plot_distributions(tester, bounds=[[-5, 5], [-5, 5]])
    plt.show()
    expectation_before, info_before = tester.estimate_expectation(n_samples=n_samples_estimation,
                                                                  expectation_function=expectation_function)

    history = tester.train(epochs, batch_size)
    expectation, info = tester.estimate_expectation(n_samples=n_samples_estimation,
                                                                  expectation_function=expectation_function)
    true_expectation = torch.sum(tester.target_dist.mean)

    print(f"true expectation is {true_expectation} \n"
          f"estimate before training is {expectation_before} \n"
          f"estimate after training is {expectation} \n" 
           f"effective sample size before is {info_before['effective_sample_size']} \n"
         f"effective sample size is {info['effective_sample_size']} \n"
         f"variance in weights is {torch.var(info['normalised_sampling_weights'])}")
    fig_after_train = plot_distributions(tester, bounds=[[-5, 5], [-5, 5]])
    plt.show()
    """
