from TargetDistributions.DoubleWell import ManyWellEnergy
import torch
from FittedModels.utils.plotting_utils import plot_sampling_info, plot_divergences
torch.manual_seed(5)
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
from FittedModels.train import LearntDistributionManager
from Utils.numerical_utils import quadratic_function as expectation_function
from FittedModels.Models.FlowModel import FlowModel
from FittedModels.utils.plotting_utils import plot_history
import matplotlib.pyplot as plt
from FittedModels.utils.plotting_utils import plot_samples_vs_contours_many_well



if __name__ == '__main__':
    torch.manual_seed(1)
    torch.set_default_dtype(torch.float64)
    # ******************* Parameters *******************
    dim = 6
    epochs = int(1e3)
    n_samples_estimation = int(1e5)
    batch_size = int(512)
    lr = 1e-3
    train_prior = False
    weight_decay = 1e-6
    clip_grad_norm = False
    optimizer = "Adamax"
    flow_type = "RealNVP"  # "IAF"
    loss_type = "DReG" # "kl"  #    #
    initial_flow_scaling = 1.5
    n_flow_steps = 64
    annealing = True

    def plotter(*args, **kwargs):
        # wrap plotting function like this so it displays during training
        plot_samples_vs_contours_many_well(*args, **kwargs)
        plt.show()
    n_plots = 20
    #*******************************************************

    target = ManyWellEnergy(dim=dim, a=-0.5, b=-6)
    torch.manual_seed(0)  # 0
    learnt_sampler = FlowModel(x_dim=dim, n_flow_steps=n_flow_steps,
                               scaling_factor=initial_flow_scaling, flow_type=flow_type)
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type=loss_type,
                                       lr=lr, optimizer=optimizer, annealing=annealing, weight_decay=weight_decay)

    plotter(tester)
    plt.show()
    expectation_before, info_before = tester.estimate_expectation(n_samples_estimation, expectation_function)

    if train_prior:
        history_prior = tester.train_prior(epochs=200, batch_size=batch_size, lr=5e-3)
        plot_history(history_prior)
        plt.show()
        plotter(tester)
        plt.show()
        expectation_prior_trained, info_prior = tester.estimate_expectation(n_samples_estimation, expectation_function)


    history = tester.train(epochs, batch_size=batch_size, clip_grad_norm=clip_grad_norm, max_grad_norm=1,
                           intermediate_plots=True, plotting_func=plotter, n_plots=n_plots)
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

    plotter(tester, n_samples=1000)
    plt.show()







