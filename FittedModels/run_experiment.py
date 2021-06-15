import pathlib

from TargetDistributions.DoubleWell import ManyWellEnergy
import torch
from FittedModels.utils.plotting_utils import plot_sampling_info, plot_divergences
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
from FittedModels.train import LearntDistributionManager
from Utils.numerical_utils import quadratic_function as expectation_function
from FittedModels.Models.FlowModel import FlowModel
from FittedModels.utils.plotting_utils import plot_history
import matplotlib.pyplot as plt

def run_experiment(save_path,
                   seed, target_type,
                   flow_type, n_flow_steps, initial_flow_scaling,
                   loss_type, epochs, batch_size, optimizer, lr, weight_decay, clip_grad_norm, annealing,
                   train_prior, train_prior_epoch, train_prior_lr,
                   n_plots, n_samples_estimation):
    """
    Function for running experiments
    """
    # save arguments
    local_var_dict = locals().copy()
    summary_results = "*********     Parameters      *******************\n\n"  # for writing to file
    for key in local_var_dict:
        summary_results += f"{key} {local_var_dict[key]}\n"

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    # setup target problem
    if target_type == "TwoModes":
        from FittedModels.utils.plotting_utils import plot_samples_vs_contours
        from TargetDistributions.VincentTargets import TwoModes
        dim = 2
        plotter = plot_samples_vs_contours
        target = TwoModes(2.0, 0.1)

    elif target_type == "DoubleWell":
        from TargetDistributions.DoubleWell import DoubleWellEnergy
        from FittedModels.utils.plotting_utils import plot_samples_vs_contours
        dim = 2
        target = DoubleWellEnergy(2, a=-0.5, b=-6)
        plotter = plot_samples_vs_contours

    elif target_type == "QuadrupleWell":
        from FittedModels.utils.plotting_utils import plot_samples_vs_contours_quadruple_well
        dim = 4
        target = ManyWellEnergy(a=-0.5, b=-6)
        plotter = plot_samples_vs_contours_quadruple_well

    elif target_type[0:3] == "MoG":  # e.g. MoG_3D
        from TargetDistributions.MoG import MoG
        from FittedModels.utils.plotting_utils import plot_samples
        dim = int(target_type[4]) # Grab dimension
        target = MoG(dim=dim, n_mixes=10, min_cov=0, loc_scaling=3)
        plotter = plot_samples
    else:
        raise NotImplementedError(f"target type: {target_type} not implemented")


    plt.rcParams.update({'figure.max_open_warning': n_plots+10})
    save_path = pathlib.Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    try:
        # let's go
        learnt_sampler = FlowModel(x_dim=dim, n_flow_steps=n_flow_steps,
                                   scaling_factor=initial_flow_scaling, flow_type=flow_type)
        tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type=loss_type,
                                           lr=lr, optimizer=optimizer, annealing=annealing, weight_decay=weight_decay)

        plotter(tester, title="before training")
        expectation_before, info_before = tester.estimate_expectation(n_samples_estimation, expectation_function)

        if train_prior:
            history_prior = tester.train_prior(epochs=train_prior_epoch, batch_size=batch_size, lr=train_prior_lr)
            plot_history(history_prior)
            plotter(tester, title="after prior training")
            expectation_prior_trained, info_prior = tester.estimate_expectation(n_samples_estimation, expectation_function)

        history = tester.train(epochs, batch_size=batch_size, clip_grad_norm=clip_grad_norm, max_grad_norm=1,
                               intermediate_plots=True, plotting_func=plotter, n_plots=n_plots)
        plot_history(history)
        plot_divergences(history)
        plot_sampling_info(history)

        expectation, info = tester.estimate_expectation(n_samples_estimation, expectation_function)

        summary_results += "\n\n *******************************    Results ********************* \n\n"
        summary_results += f"estimate before training is {expectation_before} \n" \
              f"estimate after training is {expectation} \n" \
              f"effective sample size before is {info_before['effective_sample_size'] / n_samples_estimation}\n" \
              f"effective sample size after train is {info['effective_sample_size'] / n_samples_estimation}\n" \
              f"variance in weights is {torch.var(info['normalised_sampling_weights'])} \n" \


        if train_prior:
            summary_results += f"estimate after prior training is {expectation_prior_trained} \n" \
                f"effective sample size trained prior is {info_prior['effective_sample_size'] / n_samples_estimation}\n"

        plotter(tester, n_samples=1000, title="after training")
        from Utils.plotting_utils import multipage
        multipage(str(save_path / "plots.pdf")) # save plots to pdf
        summary_results_path = str(save_path / "results.txt")
        print(summary_results)
        with open(summary_results_path, "w") as g:
            g.write(summary_results)
        plt.close("all")
    except:
        plt.close("all")
        summary_results_path = str(save_path / "results.txt")
        print(summary_results)
        with open(summary_results_path, "w") as g:
            g.write("fail")
        print("failure")

if __name__ == '__main__':
    run_experiment(save_path="Experiment_results",
                   seed=0, target_type="DoubleWell", #"TwoModes",
                   flow_type="RealNVP", n_flow_steps=64, initial_flow_scaling=1.0,
                   loss_type="kl", epochs=25, batch_size=64, optimizer="Adam", lr=1e-3, weight_decay=1e-6,
                   clip_grad_norm=False, annealing=False,
                   train_prior=True, train_prior_epoch=5, train_prior_lr=1e-2,
                   n_plots=10, n_samples_estimation=int(1e5))








