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
import pickle



def run_experiment(dim, save_path, epochs, n_flow_steps, n_distributions,
                   flow_type="ReverseIAF", batch_size=int(1e3), seed=0,
                   n_samples_expectation=int(1e5), save=True, n_plots=5, HMC_transition_args={}, learnt_dist_kwargs={"lr": 1e-4}, problem="ManyWell",
                   non_default_flow_width=None, KPI_batch_size=int(1e4)):
    local_var_dict = locals().copy()
    summary_results = "*********     Parameters      *******************\n\n"  # for writing to file
    for key in local_var_dict:
        summary_results += f"{key} {local_var_dict[key]}; \n"
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
            plt.savefig(str(save_path / "target_samples.pdf"))
        plt.show()
        def plotter(*args, **kwargs):
            plot_marginals(*args, **kwargs, clamp_samples=clamp_at)

    elif problem == "MoG_2D_illustration":
        assert dim == 2
        from TargetDistributions.MoG import MoG
        from FittedModels.utils.plotting_utils import plot_marginals
        target = MoG(dim=dim, n_mixes=5, min_cov=1, loc_scaling=10)
        if non_default_flow_width is None:
            scaling_factor_flow = 1.0
        else:
            scaling_factor_flow = non_default_flow_width
        samples_target = target.sample((batch_size,)).detach().cpu()
        clamp_at = round(float(torch.max(torch.abs(samples_target)) + 0.5))
        plot_marginals(None, n_samples=None, title=f"samples from target",
                       samples_q=samples_target, dim=dim, clamp_samples=float(torch.max(torch.abs(samples_target))))
        if save:
            plt.savefig(str(save_path / "target_samples.pdf"))
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
    summary_dict_before_AIS, long_dict_before_AIS = tester.get_performance_metrics_AIS(n_samples_expectation,
                                                                                  batch_size=batch_size)
    summary_results += f"\nBEFORE TRAINING calculated using {n_samples_expectation} samples \n"
    for key in summary_dict_before_AIS:
        summary_results += f"{key}_AIS = {round(summary_dict_before_AIS[key], 6)}; "

    summary_dict_before_flow, long_dict_before_flow = tester.get_performance_metrics_flow(n_samples_expectation,
                                                                                       batch_size=batch_size)
    for key in summary_dict_before_flow:
        summary_results += f"{key}_flow = {round(summary_dict_before_flow[key], 6)}; "

    if save:
        with open(str(save_path / "long_performance_metrics_AIS_before_train.pkl"), "wb") as f:
            pickle.dump(long_dict_before_AIS, f)
            with open(str(save_path / "long_performance_metrics_flow_before_train.pkl"), "wb") as f:
                pickle.dump(long_dict_before_flow, f)

    history = tester.train(epochs, batch_size=batch_size, intermediate_plots=True, n_plots=n_plots,
                           plotting_func=plotter, save_path=save_path, save=save,
                           KPI_batch_size=KPI_batch_size)

    plot_history(history)
    if save:
        plt.savefig(str(save_path / "histories.pdf"))
    plt.show()
    torch.manual_seed(3)
    summary_dict_after_AIS, long_dict_after_AIS, x_AIS = tester.get_performance_metrics_AIS(n_samples_expectation,
                                                                                       batch_size=batch_size,
                                                                                            return_samples=True)
    summary_results += f"\nAFTER TRAINING calculated using {n_samples_expectation} samples: \n"
    for key in summary_dict_after_AIS:
        summary_results += f"{key}_AIS = {round(summary_dict_after_AIS[key], 6)}; "

    summary_dict_after_flow, long_dict_after_flow, x_flow = tester.get_performance_metrics_flow(n_samples_expectation,
                                                                                          batch_size=batch_size,
                                                                                            return_samples=True)
    for key in summary_dict_after_flow:
        summary_results += f"{key}_flow = {round(summary_dict_after_flow[key], 6)}; "

    if save:
        with open(str(save_path / "long_performance_metrics_AIS_after_train.pkl"), "wb") as f:
            pickle.dump(long_dict_after_AIS, f)
            with open(str(save_path / "long_performance_metrics_flow_after_train.pkl"), "wb") as f:
                pickle.dump(long_dict_after_flow, f)
    plotter(tester, n_samples=None, title=f"Samples from AIS after training",
            samples_q=x_AIS, alpha=0.01)
    if save:
        plt.savefig(str(save_path / "plots_AIS_samples_final.pdf"))
    plt.show()
    plotter(tester, n_samples=None, title=f"Samples from flow after trainin",
            samples_q=x_flow, alpha=0.01)
    if save:
        plt.savefig(str(save_path / "plots_flo_samples_final.pdf"))
    plt.show()

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
        dim = 64
        use_memory = False
        epochs = int(2e3)
        n_flow_steps = 20
        n_distributions = 2 + 50
        batch_size = int(1e3)
        KPI_batch_size = batch_size * 10
        n_samples_expectation = int(batch_size*100)
        experiment_name = "not-remember"
        n_plots = 10
        flow_type = "ReverseIAF"  # "RealNVP"
        # "Expected_target_prob", "No-U", "p_accept", "No-U-unscaled"
        HMC_transition_args = {"step_tuning_method": "p_accept"}
        learnt_dist_kwargs = {"lr": 1e-4, "optimizer": "AdamW",
                              "use_memory_buffer": use_memory,
                              "memory_n_batches": 100}
        save_path = f"Results/{experiment_name}__{problem}" \
                    f"{dim}dim_{flow_type}_epochs{epochs}_flowsteps{n_flow_steps}_dist{n_distributions}" \
                    f"__{current_time}" \
                    f"HMC{HMC_transition_args['step_tuning_method']}__use_memory{use_memory}"
        print(f"running experiment {save_path} \n\n")
        assert n_samples_expectation % batch_size == 0
        tester, history = run_experiment(dim, save_path, epochs, n_flow_steps, n_distributions,
                       flow_type, learnt_dist_kwargs=learnt_dist_kwargs, n_plots=n_plots,
                       batch_size=batch_size, n_samples_expectation=n_samples_expectation, problem=problem,
                       HMC_transition_args=HMC_transition_args, KPI_batch_size=KPI_batch_size)
        print(f"\n\nfinished running experiment {save_path}")

    else:
        from datetime import datetime
        current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        problem = "MoG_2D_illustration" # "ManyWell" # "MoG" #
        dim = 2
        use_memory = False
        epochs = 500
        batch_size = int(1e2)
        n_samples_expectation = batch_size*10
        KPI_batch_size = batch_size*10
        n_flow_steps = 10
        n_plots = 5
        n_distributions = 2 + 2
        experiment_name = "local"
        flow_type = "RealNVP" # "ReverseIAF" #
        HMC_tune_options = [ "No-U", "p_accept", "No-U-unscaled" ]
        HMC_transition_args = {"step_tuning_method": HMC_tune_options[2]} # "Expected_target_prob","No-U" ,"p_accept"
        learnt_dist_kwargs = {"lr": 5e-4, "optimizer": "AdamW",
                              "use_memory_buffer": use_memory,
                              "memory_n_batches":10,
                              "alpha": 0.1}  #  , "loss_type": "kl_q"# "alpha_2_q"
        save_path = f"Results/{experiment_name}__{problem}" \
                    f"{dim}dim_{flow_type}_epochs{epochs}_flowsteps{n_flow_steps}_dist{n_distributions}" \
                    f"__{current_time}" \
                    f"HMC{HMC_transition_args['step_tuning_method']}__use_memory{use_memory}"
        print(f"running experiment {save_path} \n\n")
        tester, history = run_experiment(dim, save_path, epochs, n_flow_steps, n_distributions,
                       flow_type, save=False, n_samples_expectation=n_samples_expectation,
                       learnt_dist_kwargs=learnt_dist_kwargs, problem=problem, n_plots=n_plots,
                       HMC_transition_args=HMC_transition_args, seed=2, KPI_batch_size=KPI_batch_size)
        print(f"\n\n finished running experiment {save_path}")
