from TargetDistributions.Fast_BNN import FastPosteriorBNN
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


def plotter(tester, n_samples=int(1e3),
            title=f"samples_vs_contours", samples_q=None):
    from Utils.plotting_utils import plot_distribution
    from Utils.plotting_utils import plot_samples
    if samples_q is None:
        samples_q = tester.learnt_sampling_dist.sample((n_samples,))
    plt.plot(samples_q[:, 0], samples_q[:, 1], "o")
    plt.show()
    plot_distribution(tester.target_dist, n_points=100)
    plt.show()

if __name__ == '__main__':
    torch.manual_seed(2)
    n_plots = 5
    epochs = 100
    step_size = 1.0
    batch_size = int(1e3)

    n_samples_estimation = int(1e4)
    flow_type = "ReverseIAF"  # "ReverseIAF_MIX" #"ReverseIAF" #IAF"  # "RealNVP"
    n_flow_steps = 5
    from TargetDistributions.MoG import Difficult_MoG

    # target = Difficult_MoG(loc_scaling = 3.0, cov_scaling=1.0)
    target = FastPosteriorBNN(n_datapoints=2, x_dim=1, y_dim=1, n_hidden_layers=0, layer_width=0
                             , linear_activations=False, fixed_variance=True, use_bias=True)
    dim = target.n_parameters
    fig = plot_distribution(target, bounds=[[-30, 20], [-20, 20]])
    plt.show()
    learnt_sampler = FlowModel(x_dim=dim, scaling_factor=3.0, flow_type=flow_type, n_flow_steps=n_flow_steps)
    tester = AIS_trainer(target, learnt_sampler, n_distributions=6, n_steps_transition_operator=2,
                         step_size=step_size, train_AIS_params=True, loss_type=False,  # "DReG",
                         transition_operator="HMC", learnt_dist_kwargs={"lr": 5e-4},
                         loss_type_2="alpha_2")
    history = tester.train(epochs, batch_size=batch_size, intermediate_plots=True,
                           plotting_func=plotter, n_plots=n_plots)
    plot_history(history)
    plt.show()

