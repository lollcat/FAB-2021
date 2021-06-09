import torch
from FittedModels.Utils.plotting_utils import plot_samples, plot_sampling_info, plot_divergences
torch.manual_seed(5)
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
from FittedModels.train import LearntDistributionManager
from Utils.numerical_utils import MC_estimate_true_expectation
from Utils.numerical_utils import quadratic_function as expectation_function
from FittedModels.Models.FlowModel import FlowModel
from FittedModels.Utils.plotting_utils import plot_history
import matplotlib.pyplot as plt
from TargetDistributions.MoG import MoG
import pathlib
from datetime import datetime

def run(loss_type, lr, optimizer="Adamax", seed=0, epochs = int(5e3), dim = 3, save_path = None,
        save=False):
    if save is True:
        save_path = pathlib.Path(save_path)
        save_path = save_path / f"{loss_type}_loss_type" / f"lr_{lr}" / f"{seed}_seed"
        save_path.mkdir(parents=True, exist_ok=True)

    # ******************* Parameters *******************
    n_samples_estimation = int(1e7)
    batch_size = int(1e2)
    initial_flow_scaling = 10.0
    n_flow_steps = 3

    summary_results = f"loss type {loss_type} \n" \
                      f"batch size {batch_size} \n" + f"epochs {epochs} \n" + \
                      f"optimizer: {optimizer}, lr: {lr} \n" \
                      f"n_samples_estimation {n_samples_estimation} \n" \
                      f"initial_flow_scaling {initial_flow_scaling}, n_flow_steps {n_flow_steps} \n"
    print(summary_results)
    # ***********************************************

    # **************** Let's go   ************************
    torch.manual_seed(seed)
    target = MoG(dim=dim, n_mixes=10, min_cov=0, loc_scaling=3)
    true_expectation = MC_estimate_true_expectation(target, expectation_function, n_samples_estimation)
    torch.manual_seed(1)
    learnt_sampler = FlowModel(x_dim=dim, n_flow_steps=n_flow_steps, scaling_factor=initial_flow_scaling) # , flow_type="RealNVP", use_exp=True
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type=loss_type,
                                       lr=lr, optimizer=optimizer)
    expectation_before, info_before = tester.estimate_expectation(n_samples_estimation, expectation_function)

    if save is True:
        samples_fig_before = plot_samples(tester)  # this just looks at 2 dimensions
        plt.savefig(str(save_path / "samples_before.png"))

    history = tester.train(epochs, batch_size=batch_size, clip_grad_norm=True, max_grad_norm=1)

    if save is True:
        samples_fig_AFTER = plot_samples(tester)  # this just looks at 2 dimensions
        plt.savefig(str(save_path / "samples_after.png"))
        plot_history(history)
        plt.savefig(str(save_path / "history.png"))
        plot_divergences(history)
        plt.savefig(str(save_path / "divergences_history.png"))
        plot_sampling_info(history)
        plt.savefig(str(save_path / "sampling_history.png"))

    expectation, info = tester.estimate_expectation(n_samples_estimation, expectation_function)

    results_print = f"True expectation estimate is {true_expectation} \n" + \
          f"estimate before training is {expectation_before} \n" + \
          f"estimate after training is {expectation} \n" + \
          f"effective sample size before is {info_before['effective_sample_size'] / n_samples_estimation}\n" + \
          f"effective sample size after train is {info['effective_sample_size'] / n_samples_estimation}\n" + \
          f"variance in weights is {torch.var(info['normalised_sampling_weights'])}"
    print(results_print)

    if save is True:
        summary_results += "\n\n******************** Results ************************* \n" + results_print

        summary_results_path = str(save_path / "results.txt")
        with open(summary_results_path, "w") as g:
            g.write(summary_results)
    plt.close('all') # close all plots
    return info['effective_sample_size'] / n_samples_estimation

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    import numpy as np
    import time
    loss_types = ["DReG_kl", "kl"] # DReG
    lrs = [1e-2, 1e-3, 1e-4]
    seeds = [0, 1, 2, 3]
    ESS_history = np.zeros((len(loss_types), len(lrs), len(seeds)))
    start_time = time.time()
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    save_path = "Experiment_Results/loss_type_experiment" + f"/{current_time}"
    for j, lr in enumerate(lrs):
        for i, loss_type in enumerate(loss_types):
            for k, seed in enumerate(seeds):
                try:
                    ESS = run(loss_type, lr, seed=seed, save=True,
                              save_path=save_path)
                    ESS_history[i, j, k] = ESS
                except:
                    print(f"failure with loss_type={loss_type}, lr={lr}")
    run_time = time.time() - start_time
    print(f"runtime was {run_time/60} min")

    plt.figure()
    plt.scatter(np.array([lrs]).repeat(len(seeds), axis=0), ESS_history[0, :, :], label=loss_types[0])
    plt.scatter(np.array([lrs]).repeat(len(seeds), axis=0), ESS_history[1, :, :], label=loss_types[1], marker="x")
    plt.xscale("log")
    plt.legend()
    plt.savefig(save_path + f"/grid_search.png")
    plt.show()

