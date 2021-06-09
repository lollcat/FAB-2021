import torch
from FittedModels.Utils.plotting_utils import plot_sampling_info, plot_divergences
torch.manual_seed(5)
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
from FittedModels.train import LearntDistributionManager
from Utils.numerical_utils import quadratic_function as expectation_function
from FittedModels.Models.FlowModel import FlowModel
from FittedModels.Utils.plotting_utils import plot_history
import matplotlib.pyplot as plt
from TargetDistributions.VincentTargets import TwoModes
from FittedModels.Utils.plotting_utils import plot_samples_vs_contours
from FittedModels.Utils.plotting_utils import plot_distributions

if __name__ == '__main__':
    torch.manual_seed(1)
    torch.set_default_dtype(torch.float64)
    # ******************* Parameters *******************
    # using the same as Vincent's code so we have a fair comparison
    dim = 2
    epochs = int(5e3)
    n_samples_estimation = int(1e5)
    batch_size = int(1e2)  # 20
    lr = 4e-4
    train_prior = False
    weight_decay = 1e-6
    optimizer = "Adamax"
    flow_type = "RealNVP"  # "IAF"
    loss_type = "kl"  # "DReG" # "kl"  #    #
    initial_flow_scaling = 1.0
    n_flow_steps = 64
    annealing = True
    #*******************************************************
    target = TwoModes(2.0, 0.1)

    torch.manual_seed(1)  # 0
    learnt_sampler = FlowModel(x_dim=dim, n_flow_steps=n_flow_steps,
                               scaling_factor=initial_flow_scaling, flow_type=flow_type)
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type=loss_type,
                                       lr=lr, optimizer=optimizer, annealing=annealing, weight_decay=weight_decay)

    plot_samples_vs_contours(tester)
    plt.show()
    expectation_before, info_before = tester.estimate_expectation(n_samples_estimation, expectation_function)

    if train_prior:
        history_prior = tester.train_prior(epochs=200, batch_size=batch_size, lr=0.01)
        plot_history(history_prior)
        plt.show()
        plot_samples_vs_contours(tester)
        plt.show()
        expectation_prior_trained, info_prior = tester.estimate_expectation(n_samples_estimation, expectation_function)




    history = tester.train(epochs, batch_size=batch_size, clip_grad_norm=True, max_grad_norm=1,
                           intermediate_plots=True, plotting_func=plot_samples_vs_contours)
    plot_history(history)
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

    if train_prior:
        print(f"estimate after prior training is {expectation_prior_trained} \n"
            f"effective sample size trained prior is {info_prior['effective_sample_size'] / n_samples_estimation}\n")

    plot_samples_vs_contours(tester, n_samples=1000)
    plt.show()

    plot_distributions(tester, bounds=[[-3, 3], [-3, 3]], n_points=100)
    plt.show()




