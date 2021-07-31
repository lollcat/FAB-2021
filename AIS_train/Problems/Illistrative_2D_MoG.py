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
    lr = 1e-3

    epochs = 10
    n_plots = 10
    step_size = 1.0
    batch_size = int(1e3)
    dim = 2
    n_samples_estimation = int(1e3)
    n_samples_expectation = int(1e3)
    n_distributions = 2 + 5
    flow_type = "RealNVP"  # "ReverseIAF"  #"ReverseIAF_MIX" #"ReverseIAF" #IAF"  #
    n_flow_steps = 30
    HMC_transition_operator_args = {"step_tuning_method": "p_accept", "L": 5}  #  p_accept  "No-U"
    print(HMC_transition_operator_args)
    target = MoG(dim=dim, n_mixes=5, min_cov=1, loc_scaling=10)
    samples_target = target.sample((batch_size,)).detach().cpu()
    clamp_at = round(float(torch.max(torch.abs(samples_target))) + 0.5)
    summary_results = str(HMC_transition_operator_args) + str(n_distributions)
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
    tester = AIS_trainer(target, learnt_sampler, n_distributions=n_distributions,
                         transition_operator="HMC", lr=lr,
                         tranistion_operator_kwargs=HMC_transition_operator_args)
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
    if save:
        plt.savefig(str(save_path / "histories.pdf"))
    plt.show()
    torch.manual_seed(2)
    expectation, info_dict = tester.AIS_train.calculate_expectation(n_samples_expectation,
                                                                    expectation_function=expectation_function,
                                                                    batch_size=batch_size,
                                                                    drop_nan_and_infs=True)
    summary_results += f"ESS for samples from AIS  " \
                       f"{info_dict['effective_sample_size'].item() / n_samples_expectation}" \
                       f" calculated using {n_samples_expectation} samples \n"
    torch.manual_seed(3)
    expectation, info_dict = tester.AIS_train.calculate_expectation(n_samples_expectation,
                                                                    expectation_function=expectation_function,
                                                                    batch_size=batch_size,
                                                                    drop_nan_and_infs=True)
    summary_results += f"ESS for samples from AIS of repeat calc " \
                       f"{info_dict['effective_sample_size'].item() / n_samples_expectation}" \
                       f" calculated using {n_samples_expectation} samples \n"
    plotter(tester, n_samples=None, title=f"Samples from AIS after training",
            samples_q=info_dict["samples"], alpha=0.01)
    if save:
        plt.savefig(str(save_path / "plots_AIS_samples_final.pdf"))
    plt.show()
    torch.manual_seed(5)
    expectation_flo, info_dict_flo = tester.AIS_train.calculate_expectation_over_flow(n_samples_expectation,
                                                                                      expectation_function=expectation_function,
                                                                                      batch_size=batch_size,
                                                                                      drop_nan_and_infs=True)
    plotter(tester, n_samples=None, title=f"Samples from flow after trainin",
            samples_q=info_dict_flo["samples"], alpha=0.01)
    if save:
        plt.savefig(str(save_path / "plots_flo_samples_final.pdf"))
    plt.show()
    summary_results += f"ESS of flow model after training is " \
                       f"{info_dict_flo['effective_sample_size'].item() / n_samples_expectation}" \
                       f" calculated using {n_samples_expectation} samples"

    print(summary_results)
    if save:
        summary_results_path = str(save_path / "results.txt")
        with open(summary_results_path, "w") as g:
            g.write(summary_results)