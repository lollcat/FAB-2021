import torch
import matplotlib.pyplot as plt

torch.manual_seed(5)
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
from TargetDistributions.BayesianNN import PosteriorBNN
from FittedModels.Models.FlowModel import FlowModel
from FittedModels.utils.plotting_utils import plot_history, plot_sampling_info, plot_divergences
from FittedModels.train import LearntDistributionManager
from Utils.numerical_utils import expectation_function

if __name__ == '__main__':
    # setup target distribution
    torch.manual_seed(1)
    target = PosteriorBNN(n_datapoints=10, x_dim=2, y_dim=2, n_hidden_layers=1, layer_width=2)
    epochs = 2  # 1000
    batch_size = 10  # int(1e5)
    n_samples_estimation = int(1e2)
    dim = target.n_parameters
    print(dim)

    # setup learnt sampler with flow
    learnt_sampler = FlowModel(x_dim=dim, n_flow_steps=3, scaling_factor=5)
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type="DReG")
    expectation_before, info_before = tester.estimate_expectation(n_samples_estimation, expectation_function)

    # train
    history = tester.train(epochs, batch_size=batch_size, clip_grad_norm=True, max_grad_norm=2)
    expectation, info = tester.estimate_expectation(n_samples_estimation, expectation_function)

    hist_plot = plot_history(history)
    plt.show()

    plot_divergences(history)
    plt.show()

    plot_sampling_info(history)
    plt.show()

    expectation, info = tester.estimate_expectation(n_samples_estimation, expectation_function)
    print(f"estimate before training is {expectation_before} \n"
          f"estimate after training is {expectation} \n"
          f"effective sample size before is {info_before['effective_sample_size'] / n_samples_estimation}\n"
          f"effective sample size after train is {info['effective_sample_size'] / n_samples_estimation}\n"
          f"variance in weights is {torch.var(info['normalised_sampling_weights'])}")




