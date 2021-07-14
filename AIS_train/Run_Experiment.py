import pathlib
import os
import sys

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
import torch
from Utils.plotting_utils import multipage
from TargetDistributions.DoubleWell import ManyWellEnergy
from FittedModels.utils.plotting_utils import plot_samples_vs_contours_many_well

from FittedModels.Models.FlowModel import FlowModel
from AIS_train.train_AIS import AIS_trainer
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
from FittedModels.utils.plotting_utils import plot_history, plot_distributions, plot_samples

import matplotlib.pyplot as plt
import torch
from Utils.plotting_utils import plot_func2D, plot_distribution
from Utils.numerical_utils import MC_estimate_true_expectation
from Utils.numerical_utils import quadratic_function as expectation_function
from Utils.DebuggingUtils import print_memory_stats

def plotter(*args, **kwargs):
    plot_samples_vs_contours_many_well(*args, **kwargs)

def run_experiment(dim, save_path, epochs, n_flow_steps, n_distributions,
                   flow_type="ReverseIAF", batch_size=int(1e3), seed=0,
                   n_samples_expectation=int(1e5), save=True, n_plots=5, train_AIS_params=True):
    local_var_dict = locals().copy()
    summary_results = "*********     Parameters      *******************\n\n"  # for writing to file
    for key in local_var_dict:
        summary_results += f"{key} {local_var_dict[key]}\n"
    if save:
        save_path = pathlib.Path(save_path)
        save_path.mkdir(parents=True, exist_ok=False)

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    target = ManyWellEnergy(dim=dim, a=-0.5, b=-6)

    learnt_sampler = FlowModel(x_dim=dim, scaling_factor=2.0, flow_type=flow_type,
                               n_flow_steps=n_flow_steps)
    tester = AIS_trainer(target, learnt_sampler, loss_type=False, n_distributions=n_distributions
                         , n_steps_transition_operator=1,
                         step_size=1.0, transition_operator="HMC", learnt_dist_kwargs={"lr": 1e-4},
                         loss_type_2="alpha_2", train_AIS_params=train_AIS_params, inner_loop_steps=5)
    summary_results += "\n\n *******************************    Results ********************* \n\n"
    expectation_before, info_dict_before = tester.AIS_train.calculate_expectation(n_samples_expectation,
                                                                                  expectation_function=expectation_function,
                                                                                  batch_size=int(1e3))
    summary_results += f"ESS of AIS before training is " \
       f"{info_dict_before['effective_sample_size'].item() / info_dict_before['normalised_sampling_weights'].shape[0]}" \
       f" calculated using {info_dict_before['normalised_sampling_weights'].shape[0]} samples \n"

    history = tester.train(epochs, batch_size=batch_size, intermediate_plots=True, n_plots=n_plots, plotting_func=plotter)

    if save:
        multipage(str(save_path / "plots_during_training.pdf"))
        plt.close("all")
    else:
        plt.show()
    plot_history(history)
    if save:
        multipage(str(save_path / "histories.pdf"))
        plt.close("all")
    else:
        plt.show()
    expectation, info_dict = tester.AIS_train.calculate_expectation(n_samples_expectation,
                                                                    expectation_function=expectation_function,
                                                                    batch_size=int(1e3),
                                                                    drop_nan_and_infs=True)
    summary_results += f"ESS for samples from AIS  "\
        f"{info_dict['effective_sample_size'].item()/ info_dict['normalised_sampling_weights'].shape[0]}" \
        f" calculated using {info_dict['normalised_sampling_weights'].shape[0]} samples \n"
    expectation, info_dict = tester.AIS_train.calculate_expectation(n_samples_expectation,
                                                                    expectation_function=expectation_function,
                                                                    batch_size=int(1e3),
                                                                    drop_nan_and_infs=True)
    summary_results += f"ESS for samples from AIS of repeat calc " \
                       f"{info_dict['effective_sample_size'].item() / info_dict['normalised_sampling_weights'].shape[0]}" \
                       f" calculated using {info_dict['normalised_sampling_weights'].shape[0]} samples \n"
    plot_samples_vs_contours_many_well(tester, n_samples=1000,
                                       title=f"training epoch, samples from flow")
    plot_samples_vs_contours_many_well(tester, n_samples=None,
                                       title=f"training epoch, samples from AIS",
                                       samples_q=info_dict["samples"])
    if save:
        multipage(str(save_path / "plots.pdf"))
        plt.close("all")
    else:
        plt.show()

    expectation_flo, info_dict_flo = tester.AIS_train.calculate_expectation_over_flow(n_samples_expectation,
                                                                  expectation_function=expectation_function,
                                                                  batch_size=int(1e3),
                                                                  drop_nan_and_infs=True)
    summary_results += f"ESS of flow model after training is " \
           f"{info_dict_flo['effective_sample_size'].item()/ info_dict_flo['normalised_sampling_weights'].shape[0]}" \
           f" calculated using {info_dict_flo['normalised_sampling_weights'].shape[0]} samples"

    print(summary_results)
    if save:
        summary_results_path = str(save_path / "results.txt")
        with open(summary_results_path, "w") as g:
            g.write(summary_results)

if __name__ == '__main__':
    from datetime import datetime
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    dim = 2
    epochs = 1000
    n_flow_steps = 5
    n_distributions = 2
    experiment_name = None #"testing4"
    flow_type = "RealNVP"  # "ReverseIAF"
    save_path = f"{experiment_name}__" \
                f"{dim}dim_{flow_type}_epochs{epochs}_flowsteps{n_flow_steps}_dist{n_distributions}__{current_time}"
    print(f"running experiment {save_path} \n\n")
    run_experiment(dim, save_path, epochs, n_flow_steps, n_distributions,
                   flow_type, n_samples_expectation=int(1e3), save=False, train_AIS_params=True)
    print(f"\n\nfinished running experiment {save_path}")







