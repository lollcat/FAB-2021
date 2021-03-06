if __name__ == '__main__':
    epochs = 12000
    n_flow_steps = 30
    n_distributions = 20
    batch_size = int(1e3)
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

    torch.set_default_dtype(torch.float64)


    def plotter(*args, **kwargs):
        plot_samples_vs_contours_many_well(*args, **kwargs)


    save_path = pathlib.Path(f"16_dim_ReverseIAF_epochs{epochs}_flowsteps{n_flow_steps}_dist{n_distributions}")
    save_path.mkdir(parents=True, exist_ok=False)
    dim = 16
    target = ManyWellEnergy(dim=dim, a=-0.5, b=-6)
    n_samples_expectation = int(1e5)
    n_samples = int(1e4)

    learnt_sampler = FlowModel(x_dim=dim, scaling_factor=2.0, flow_type="ReverseIAF", n_flow_steps=n_flow_steps)
    tester = AIS_trainer(target, learnt_sampler, loss_type=False, n_distributions=n_distributions
                         , n_steps_transition_operator=1,
                         step_size=1.0, transition_operator="HMC", learnt_dist_kwargs={"lr": 1e-4},
                         loss_type_2="alpha_2", train_AIS_params=True, inner_loop_steps=5)

    history = tester.train(epochs, batch_size=batch_size, intermediate_plots=True, n_plots=20, plotting_func=plotter)
    plot_history(history)
    multipage(str(save_path / "training.pdf"))
    plt.close("all")
    expectation, info_dict = tester.AIS_train.calculate_expectation(n_samples_expectation,
                                                                    expectation_function=expectation_function,
                                                                    batch_size=int(1e3),
                                                                    drop_nan_and_infs=True)
    print(info_dict['effective_sample_size'].item() / info_dict["normalised_sampling_weights"].shape[0])
    plot_samples_vs_contours_many_well(tester, n_samples=1000,
                                       title=f"training epoch, samples from flow")
    plot_samples_vs_contours_many_well(tester, n_samples=None,
                                       title=f"training epoch, samples from AIS",
                                       samples_q=info_dict["samples"])
    multipage(str(save_path / "after_training.pdf"))

    expectation_flo, info_dict_flo = tester.AIS_train.calculate_expectation_over_flow(n_samples_expectation,
                                                                                      expectation_function=expectation_function,
                                                                                      batch_size=int(1e3),
                                                                                      drop_nan_and_infs=True)
    print(info_dict_flo['effective_sample_size'].item() / info_dict_flo["normalised_sampling_weights"].shape[0])






