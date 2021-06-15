from FittedModels.Models.FlowModel import FlowModel
from AIS_train.train_AIS import AIS_trainer
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
from FittedModels.utils.plotting_utils import plot_history, plot_samples
import matplotlib.pyplot as plt
import torch
from TargetDistributions.MoG import MoG
from Utils.plotting_utils import plot_distribution
from Utils.numerical_utils import MC_estimate_true_expectation
from Utils.numerical_utils import quadratic_function as expectation_function
torch.set_default_dtype(torch.float64)
if __name__ == '__main__':
    get_useful_info_before_train = False  # compare to vanilla IS before, plot stuff

    torch.manual_seed(2)
    epochs = 100
    dim = 2
    n_samples_estimation = int(1e4)
    n_samples_expectation = int(1e6)
    n_samples = int(1e3)
    flow_scaling = 2.0

    def plotter(*args, **kwargs):
        plot_samples(*args, **kwargs)
        plt.show()


    target = MoG(dim=dim, n_mixes=10, min_cov=1, loc_scaling=5)
    if dim == 2:
        fig = plot_distribution(target, bounds=[[-30, 20], [-20, 20]])

    learnt_sampler = FlowModel(x_dim=dim, scaling_factor=5.0, flow_type="RealNVP", n_flow_steps=64)
    tester = AIS_trainer(target, learnt_sampler, loss_type="DReG", n_distributions=10, n_steps_transition_operator=3,
                         step_size=1.0, transition_operator="HMC", train_AIS_params=True,
                         learnt_dist_kwargs={"lr": 1e-3})


    if get_useful_info_before_train:
        true_expectation = MC_estimate_true_expectation(target, expectation_function, int(1e6))
        print(true_expectation)

        vanilla_IS = VanillaImportanceSampling(sampling_distribution=learnt_sampler, target_distribution=target)
        expectation_vanilla, info_dict_vanilla = \
        vanilla_IS.calculate_expectation(n_samples_expectation, expectation_function=expectation_function)
        print(f"true expectation is {true_expectation}, estimated expectation is {expectation_vanilla}")
        print(f"ESS is {info_dict_vanilla['effective_sample_size']/n_samples_expectation}, \
              var is {torch.var(info_dict_vanilla['normalised_sampling_weights'])}")


        expectation, info_dict = tester.AIS_train.calculate_expectation(n_samples_expectation,
                                                                            expectation_function=expectation_function)
        print(f"true expectation is {true_expectation}, estimated expectation is {expectation}")
        print(
            f"ESS is {info_dict['effective_sample_size'] / n_samples_expectation}, "
            f"var is {torch.var(info_dict['normalised_sampling_weights'])}")

        plt.figure()
        learnt_dist_samples = learnt_sampler.sample((n_samples,)).cpu().detach()
        plt.scatter(learnt_dist_samples[:, 0], learnt_dist_samples[:, 1])
        plt.title("approximating distribution samples")
        plt.show()
        plt.figure()
        plt.scatter(info_dict["samples"][:, 0].cpu(), info_dict["samples"][:, 1].cpu())
        plt.title("annealed samples")
        plt.show()
        plt.figure()
        true_samples = target.sample((n_samples,)).cpu().detach()
        plt.scatter(true_samples[:, 0], true_samples[:, 1])
        plt.title("true samples")
        plt.show()

        plot_samples(tester)
        plt.show()

    history = tester.train(epochs, batch_size=int(1e4), intermediate_plots=True, n_plots=3, plotting_func=plotter)

    plot_history(history)
    plt.show()
    plot_samples(tester)
    plt.show()

