import pathlib
import os
import sys

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
import torch
from Utils.plotting_utils import multipage


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



def run_experiment(dim, save_path, epochs, n_flow_steps, n_distributions,
                   flow_type="ReverseIAF", batch_size=int(1e3), seed=0,
                   n_samples_expectation=int(1e5), save=True, n_plots=5, HMC_transition_args={}, learnt_dist_kwargs={"lr": 1e-4}, problem="ManyWell",
                   non_default_flow_width = None):
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
    if problem == "ManyWell":
        from TargetDistributions.DoubleWell import ManyWellEnergy
        from FittedModels.utils.plotting_utils import plot_samples_vs_contours_many_well
        target = ManyWellEnergy(dim=dim, a=-0.5, b=-6)
        def plotter(*args, **kwargs):
            plot_samples_vs_contours_many_well(*args, **kwargs)

        if non_default_flow_width is None:
            scaling_factor_flow = 2.0
        else:
            scaling_factor_flow = non_default_flow_width
    elif problem == "MoG":
        from TargetDistributions.MoG import MoG
        from FittedModels.utils.plotting_utils import plot_marginals
        target = MoG(dim, diagonal_covariance=False, cov_scaling=0.1, min_cov=0.0, loc_scaling=8.0, n_mixes=dim,
                     uniform_component_probs=True)
        if non_default_flow_width is None:
            scaling_factor_flow = 10.0
        else:
            scaling_factor_flow = non_default_flow_width
        samples_target = target.sample((batch_size,)).detach().cpu()
        clamp_at = round(float(torch.max(torch.abs(samples_target)) + 0.5))
        plot_marginals(None, n_samples=None, title=f"samples from target",
                samples_q=samples_target, dim=dim, clamp_samples=float(torch.max(torch.abs(samples_target))))
        if save:
            plt.savefig(str(save_path / "target_samples.png"))
        plt.show()
        def plotter(*args, **kwargs):
            plot_marginals(*args, **kwargs, clamp_samples=clamp_at)

    else:
        raise Exception

    learnt_sampler = FlowModel(x_dim=dim, scaling_factor=scaling_factor_flow, flow_type=flow_type,
                               n_flow_steps=n_flow_steps)
    tester = AIS_trainer(target, learnt_sampler, n_distributions=n_distributions
                         , tranistion_operator_kwargs=HMC_transition_args, transition_operator="HMC",
                         **learnt_dist_kwargs)
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
    plotter(tester, n_samples=None, title=f"Samples from AIS after training",
            samples_q=info_dict["samples"], alpha=0.01)
    if save:
        plt.savefig(str(save_path / "plots_AIS_samples_final.png"))
    plt.show()
    torch.manual_seed(5)
    expectation_flo, info_dict_flo = tester.AIS_train.calculate_expectation_over_flow(n_samples_expectation,
                                                                  expectation_function=expectation_function,
                                                                  batch_size=batch_size,
                                                                  drop_nan_and_infs=True)
    plotter(tester, n_samples=None, title=f"Samples from flow after trainin",
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
    return tester, history


if __name__ == '__main__':
    testing_local = True
    if not testing_local:
        from datetime import datetime
        current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        problem = "ManyWell"
        dim = 32
        epochs = int(1e5)
        n_flow_steps = 20
        n_distributions = 2 + 4
        batch_size = int(1e3)
        n_samples_expectation = int(batch_size*100)
        experiment_name = "glenbeach"
        n_plots = 10
        learnt_dist_kwargs = {"lr": 1e-4, "optimizer": "AdamW"}
        flow_type = "ReverseIAF" # "RealNVP"
        # "Expected_target_prob", "No-U", "p_accept", "No-U-unscaled"
        HMC_transition_args = {"step_tuning_method": "No-U"}
        save_path = f"Results/{experiment_name}__{problem}" \
                    f"{dim}dim_{flow_type}_epochs{epochs}_flowsteps{n_flow_steps}_dist{n_distributions}" \
                    f"__{current_time}" \
                    f"HMC{HMC_transition_args['step_tuning_method']}"
        print(f"running experiment {save_path} \n\n")
        assert n_samples_expectation % batch_size == 0
        tester, history = run_experiment(dim, save_path, epochs, n_flow_steps, n_distributions,
                       flow_type, learnt_dist_kwargs=learnt_dist_kwargs, n_plots=n_plots,
                       batch_size=batch_size, n_samples_expectation=n_samples_expectation, problem=problem,
                       HMC_transition_args=HMC_transition_args)
        print(f"\n\nfinished running experiment {save_path}")

    else:
        from datetime import datetime
        current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        problem = "ManyWell" # "MoG" #
        dim = 8
        epochs = 500
        n_flow_steps = 5
        n_plots = 2
        n_distributions = 2 + 2
        experiment_name = "testing5"
        flow_type = "RealNVP" # "ReverseIAF" #
        # "Expected_target_prob", "No-U", "p_accept", "No-U-unscaled"
        HMC_transition_args = {"step_tuning_method": "No-U"} # "Expected_target_prob","No-U" ,"p_accept"
        learnt_dist_kwargs = {"lr": 2e-4, "optimizer": "AdamW"}
        save_path = f"Results/{experiment_name}__{problem}" \
                    f"{dim}dim_{flow_type}_epochs{epochs}_flowsteps{n_flow_steps}_dist{n_distributions}" \
                    f"__{current_time}" \
                    f"HMC{HMC_transition_args['step_tuning_method']}"
        print(f"running experiment {save_path} \n\n")
        tester, history = run_experiment(dim, save_path, epochs, n_flow_steps, n_distributions,
                       flow_type, save=False, n_samples_expectation=int(1e3),
                       learnt_dist_kwargs=learnt_dist_kwargs, problem=problem, n_plots=n_plots,
                       HMC_transition_args=HMC_transition_args)
        print(f"\n\n finished running experiment {save_path}")
