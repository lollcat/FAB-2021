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
if __name__ == '__main__':
    epochs = 30
    n_plots = 5
    def plotter(*args, **kwargs):
        # wrap plotting function like this so it displays during training
        plot_samples_vs_contours_many_well(*args, **kwargs)
        plt.show()
    dim = 8
    target = ManyWellEnergy(dim=dim, a=-0.5, b=-6)
    n_samples_expectation = int(1e4)
    n_samples = int(1e4)

    learnt_sampler = FlowModel(x_dim=dim, scaling_factor=2.0, flow_type="RealNVP", n_flow_steps=60)
    vanilla_IS = VanillaImportanceSampling(sampling_distribution=learnt_sampler, target_distribution=target)
    with torch.no_grad():
        expectation_vanilla, info_dict_vanilla = \
        vanilla_IS.calculate_expectation(n_samples_expectation, expectation_function=expectation_function)
        print(f"ESS is {info_dict_vanilla['effective_sample_size']/n_samples_expectation}, \
              var is {torch.var(info_dict_vanilla['normalised_sampling_weights'])}")

    learnt_sampler = FlowModel(x_dim=dim, scaling_factor=2.0, flow_type="RealNVP", n_flow_steps=60)
    tester = AIS_trainer(target, learnt_sampler, loss_type=False, n_distributions=5, n_steps_transition_operator=2,
                         step_size=1.0, transition_operator="HMC", learnt_dist_kwargs={"lr": 1e-3},
                         loss_type_2="alpha_2")


    expectation_before, info_dict_before = tester.AIS_train.calculate_expectation(n_samples_expectation,
                                                                        expectation_function=expectation_function,
                                                                                 batch_size=int(1e3))
    print(info_dict_before['effective_sample_size'].item() / n_samples_expectation)

    plot_samples_vs_contours_many_well(tester, n_samples=1000,
                                       title=f"training epoch, samples from flow")
    plot_samples_vs_contours_many_well(tester, n_samples=None,
                                       title=f"training epoch, samples from AIS",
                                       samples_q=info_dict_before["samples"])

    history = tester.train(epochs, batch_size=int(1e3), intermediate_plots=True, n_plots=n_plots, plotting_func=plotter)

    expectation, info_dict = tester.AIS_train.calculate_expectation(n_samples_expectation,
                                                                        expectation_function=expectation_function,
                                                                                 batch_size=int(1e3),
                                                                   drop_nan_and_infs=True)
    print(info_dict['effective_sample_size'].item() / info_dict["normalised_sampling_weights"].shape[0])
