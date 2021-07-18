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
                   n_samples_expectation=int(1e5), save=True, n_plots=5, train_AIS_params=False,
                   step_size=1.0, learnt_dist_kwargs={"lr": 1e-4}):
    local_var_dict = locals().copy()
    summary_results = "*********     Parameters      *******************\n\n"  # for writing to file
    for key in local_var_dict:
        summary_results += f"{key} {local_var_dict[key]}\n"
    if save:
        save_path = pathlib.Path(save_path)
        save_path.mkdir(parents=True, exist_ok=False)
    else:
        save_path = None

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    target = ManyWellEnergy(dim=dim, a=-0.5, b=-6)

    learnt_sampler = FlowModel(x_dim=dim, scaling_factor=2.0, flow_type=flow_type,
                               n_flow_steps=n_flow_steps)
    tester = AIS_trainer(target, learnt_sampler, loss_type=False, n_distributions=n_distributions
                         , n_steps_transition_operator=1,
                         step_size=step_size, transition_operator="HMC", learnt_dist_kwargs=learnt_dist_kwargs,
                         loss_type_2="alpha_2", train_AIS_params=train_AIS_params, inner_loop_steps=5)
    summary_results += "\n\n *******************************    Results ********************* \n\n"
    expectation_before, info_dict_before = tester.AIS_train.calculate_expectation(n_samples_expectation,
                                                                                  expectation_function=expectation_function,
                                                                                  batch_size=batch_size)
    summary_results += f"ESS of AIS before training is " \
       f"{info_dict_before['effective_sample_size'].item() / n_samples_expectation}" \
       f" calculated using {n_samples_expectation} samples \n"

    history = tester.train(epochs, batch_size=batch_size, intermediate_plots=True, n_plots=n_plots,
                           plotting_func=plotter, save_path=save_path, save=save)

    plot_history(history)
    if save:
        plt.savefig(str(save_path / "histories.png"))
    plt.show()
    torch.manual_seed(2)
    expectation, info_dict = tester.AIS_train.calculate_expectation(n_samples_expectation,
                                                                    expectation_function=expectation_function,
                                                                    batch_size=batch_size,
                                                                    drop_nan_and_infs=True)
    summary_results += f"ESS for samples from AIS  "\
        f"{info_dict['effective_sample_size'].item()/ n_samples_expectation}" \
        f" calculated using {n_samples_expectation} samples \n"
    torch.manual_seed(3)
    expectation, info_dict = tester.AIS_train.calculate_expectation(n_samples_expectation,
                                                                    expectation_function=expectation_function,
                                                                    batch_size=batch_size,
                                                                    drop_nan_and_infs=True)
    summary_results += f"ESS for samples from AIS of repeat calc " \
                       f"{info_dict['effective_sample_size'].item() / n_samples_expectation}" \
                       f" calculated using {n_samples_expectation} samples \n"
    plot_samples_vs_contours_many_well(tester, n_samples=None,
                                       title=f"training epoch, samples from AIS",
                                       samples_q=info_dict["samples"], alpha=0.01)
    if save:
        plt.savefig(str(save_path / "plots_AIS_samples_final.png"))
    plt.show()
    torch.manual_seed(5)
    expectation_flo, info_dict_flo = tester.AIS_train.calculate_expectation_over_flow(n_samples_expectation,
                                                                  expectation_function=expectation_function,
                                                                  batch_size=batch_size,
                                                                  drop_nan_and_infs=True)
    plot_samples_vs_contours_many_well(tester, n_samples=None,
                                       title=f"training epoch, samples from flow",
                                       samples_q=info_dict_flo["samples"], alpha=0.01)
    if save:
        plt.savefig(str(save_path / "plots_flo_samples_final.png"))
    plt.show()
    summary_results += f"ESS of flow model after training is " \
           f"{info_dict_flo['effective_sample_size'].item()/ n_samples_expectation}" \
           f" calculated using {n_samples_expectation} samples"

    print(summary_results)
    if save:
        summary_results_path = str(save_path / "results.txt")
        with open(summary_results_path, "w") as g:
            g.write(summary_results)
    return tester


if __name__ == '__main__':
    testing_local = False
    if not testing_local:
        from datetime import datetime
        current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        dim = 64
        epochs = int(1e5)
        n_flow_steps = 20
        n_distributions = 2 + 4
        batch_size = int(3e3)
        n_samples_expectation = int(batch_size * 100)
        experiment_name = "mogdog_AdamW_lower_lr_bigger_batch"
        n_plots = 20
        learnt_dist_kwargs = {"lr": 1e-4, "optimizer": "AdamW"}
        flow_type = "ReverseIAF" # "RealNVP"
        save_path = f"Results/{experiment_name}__" \
                    f"{dim}dim_{flow_type}_epochs{epochs}_flowsteps{n_flow_steps}_dist{n_distributions}__{current_time}"
        print(f"running experiment {save_path} \n\n")
        run_experiment(dim, save_path, epochs, n_flow_steps, n_distributions,
                       flow_type, learnt_dist_kwargs=learnt_dist_kwargs, train_AIS_params=False, n_plots=n_plots,
                       batch_size=batch_size, n_samples_expectation=n_samples_expectation)
        print(f"\n\nfinished running experiment {save_path}")

    else:
        from datetime import datetime
        current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        dim = 2
        epochs = 5
        n_flow_steps = 2
        n_distributions = 3
        experiment_name = "testing5"
        flow_type = "ReverseIAF" # "RealNVP"
        learnt_dist_kwargs = {"lr": 1e-3, "optimizer": "AdamW"}
        save_path = f"Results/{experiment_name}__" \
                    f"{dim}dim_{flow_type}_epochs{epochs}_flowsteps{n_flow_steps}_dist{n_distributions}__{current_time}"
        print(f"running experiment {save_path} \n\n")
        run_experiment(dim, save_path, epochs, n_flow_steps, n_distributions,
                       flow_type, save=True, n_samples_expectation=int(1e3), train_AIS_params=False,
                       learnt_dist_kwargs=learnt_dist_kwargs)
        print(f"\n\nfinished running experiment {save_path}")
