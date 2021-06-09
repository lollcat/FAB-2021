import torch
from FittedModels.Utils.plotting_utils import plot_samples

torch.manual_seed(5)
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
from FittedModels.train import LearntDistributionManager
from Utils.numerical_utils import MC_estimate_true_expectation, expectation_function
from FittedModels.Models.FlowModel import FlowModel
from TargetDistributions.MoG import MoG

if __name__ == '__main__':
    """
    Problem: unable to calculate alpha divergence over p(x)
    Hypotheses: make flow wider ??
    """

    torch.set_default_dtype(torch.float64)
    epochs = 500
    batch_size = int(1e2)
    dim = 3
    n_samples_estimation = int(1e6)
    KPI_n_samples = int(1e3)
    torch.manual_seed(0)  # 0 breaks it within 1000 epochs
    target = MoG(dim=dim, n_mixes=10, min_cov=0, loc_scaling=3)
    true_expectation = MC_estimate_true_expectation(target, expectation_function, int(1e6))
    print(true_expectation)
    print(MC_estimate_true_expectation(target, expectation_function,
                                       int(1e6)))  # print twice to make sure estimates are resonably close
    torch.manual_seed(1)
    learnt_sampler = FlowModel(x_dim=dim, n_flow_steps=3, scaling_factor=5.0)  # , flow_type="RealNVP")
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type="DReG", lr=1e-3)
    expectation_before, info_before = tester.estimate_expectation(n_samples_estimation, expectation_function)

    samples_fig_before = plot_samples(tester)  # this just looks at 2 dimensions

    history = tester.train(epochs, batch_size=batch_size, KPI_batch_size=KPI_n_samples,
                           clip_grad_norm=True, max_grad_norm=1)

    expectation, info = tester.estimate_expectation(n_samples_estimation, expectation_function)
    print(f"True expectation estimate is {true_expectation} \n"
          f"estimate before training is {expectation_before} \n"
          f"estimate after training is {expectation} \n"
          f"effective sample size before is {info_before['effective_sample_size'] / n_samples_estimation}\n"
          f"effective sample size after train is {info['effective_sample_size'] / n_samples_estimation}\n"
          f"variance in weights is {torch.var(info['normalised_sampling_weights'])}")


