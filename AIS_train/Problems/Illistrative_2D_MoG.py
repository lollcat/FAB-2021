from FittedModels.Models.FlowModel import FlowModel
from FittedModels.utils.plotting_utils import plot_history
import matplotlib.pyplot as plt
from TargetDistributions.MoG import MoG
from Utils.plotting_utils import plot_distribution
from Utils.numerical_utils import MC_estimate_true_expectation
from Utils.numerical_utils import quadratic_function as expectation_function
from FittedModels.utils.plotting_utils import plot_samples
import torch
from AIS_train.train_AIS import AIS_trainer
from FittedModels.utils.plotting_utils import plot_marginals
import pathlib

if __name__ == '__main__':
    save = True
    torch.manual_seed(2)
    train_AIS_params = True
    print(f"train AIS parameters = {train_AIS_params}")
    epochs = 600
    n_plots = 100
    step_size = 1.0
    batch_size = int(1e3)
    dim = 2
    n_samples_estimation = int(1e4)
    n_distributions = 2 + 2
    flow_type = "RealNVP"  # "ReverseIAF"  #"ReverseIAF_MIX" #"ReverseIAF" #IAF"  #
    n_flow_steps = 30
    target = MoG(dim=dim, n_mixes=5, min_cov=1, loc_scaling=10)
    samples_target = target.sample((batch_size,)).detach().cpu()
    clamp_at = round(float(torch.max(torch.abs(samples_target))) + 0.5)
    save_path = "MoG_2D_illustration"
    save_path = pathlib.Path(save_path)
    def plotter(*args, **kwargs):
        plot_marginals(*args, **kwargs, clamp_samples=clamp_at)
    true_expectation = MC_estimate_true_expectation(target, expectation_function, int(1e5))
    fig = plot_distribution(target, bounds=[[-30, 20], [-20, 20]])
    plt.show()
    plotter(None, n_samples=batch_size,
                  title=f"true samples",
                  samples_q=samples_target, dim=2)
    plt.show()
    learnt_sampler = FlowModel(x_dim=dim, scaling_factor=1.0, flow_type=flow_type, n_flow_steps=n_flow_steps)
    tester = AIS_trainer(target, learnt_sampler, n_distributions=n_distributions, n_steps_transition_operator=2,
                         step_size=step_size, train_AIS_params=train_AIS_params, loss_type=False,  # "DReG",
                         transition_operator="HMC", learnt_dist_kwargs={"lr": 1e-3, "optimizer": "AdamW"},
                         loss_type_2="alpha_2")
    plotter(tester)
    plt.show()


    expectation, info_dict = tester.AIS_train.calculate_expectation(n_samples_estimation,
                                                                    expectation_function=expectation_function)
    print(f"true expectation is {true_expectation}, estimated expectation is {expectation}")
    print(
        f"ESS is {info_dict['effective_sample_size'] / n_samples_estimation}, "
        f"var is {torch.var(info_dict['normalised_sampling_weights'])}")


    history = tester.train(epochs, batch_size=batch_size, intermediate_plots=True,
                           plotting_func=plotter, n_plots=n_plots, save_path=save_path, save=save)
    plot_history(history)
    plt.show()
    plot_samples(tester)
    plt.show()

    expectation, info_dict = tester.AIS_train.calculate_expectation(n_samples_estimation,
                                                                    expectation_function=expectation_function)
    print(f"AFTER TRAINING: \n true expectation is {true_expectation}, estimated expectation is {expectation}")
    print(
        f"ESS is {info_dict['effective_sample_size'] / n_samples_estimation}, "
        f"var is {torch.var(info_dict['normalised_sampling_weights'])}")