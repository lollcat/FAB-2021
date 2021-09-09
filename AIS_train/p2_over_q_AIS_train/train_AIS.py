from AIS_train.train_AIS import AIS_trainer as base
from AIS_train.p2_over_q_AIS_train.AnnealedImportanceSampler import AnnealedImportanceSampler
import torch
import torch.nn.functional as F
import numpy as np

class AIS_trainer(base):
    def __init__(self, target_distribution, fitted_model,
                 n_distributions=2+2, transition_operator="HMC",
                 AIS_kwargs={}, tranistion_operator_kwargs={}, use_GPU = True,
                 optimizer="AdamW", lr=1e-3, use_memory_buffer=False,
                 memory_n_batches=100, allow_ignore_nan_loss=True, clip_grad_norm=True,
                 alpha=2.0
                 ):
        self.AIS_train = AnnealedImportanceSampler(fitted_model, target_distribution,
                                                   transition_operator=transition_operator,
                                                   n_distributions=n_distributions,
                                                   **AIS_kwargs,
                                                   transition_operator_kwargs=tranistion_operator_kwargs)
        if use_GPU is True:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self.loss_type = "alpha_2_IS" # hack to get it to use the base class as we desire
        self.allow_ignore_nan_loss = allow_ignore_nan_loss
        self.clip_grad_norm = clip_grad_norm
        self.max_grad_norm = 1.0
        self.importance_sampler = self.AIS_train
        self.learnt_sampling_dist = fitted_model
        self.target_dist = target_distribution
        torch_optimizer = getattr(torch.optim, optimizer)
        self.optimizer = torch_optimizer(self.learnt_sampling_dist.parameters(), lr=lr)
        self.to(device=self.device)
        self.use_memory_buffer = use_memory_buffer
        if self.use_memory_buffer:
            self.memory_buffer_x = None # deque(maxlen=memory_n_batches)
            self.memory_buffer_log_w = None
            self.memory_position_counter = 0
            self.max_memory_points = None
            self.max_memory_batches = memory_n_batches
            self.n_gradient_update_batches = int(self.max_memory_batches / 10)  # update using a 10_th of the memory

    def loss(self, x_samples, log_w):
        # first remove samples that have inf/nan log w
        valid_indices = ~torch.isinf(log_w) & ~torch.isnan(log_w)
        if torch.sum(valid_indices) == 0:  # no valid indices
            print("no valid indices")
            return torch.tensor(float("nan")), None
        if valid_indices.all():
            pass
        else:
            log_w = log_w[valid_indices]
            x_samples = x_samples[valid_indices, :]

        log_q_x = self.learnt_sampling_dist.log_prob(x_samples.detach())
        # log_p_x = self.target_dist.log_prob(x_samples.detach())

        # also check that we have valid log probs
        valid_indices = ~torch.isinf(log_q_x) & ~torch.isnan(log_q_x)
        if valid_indices.all():
            return - torch.sum(F.softmax(log_w, dim=-1) * log_q_x)
            # return torch.logsumexp((2*log_p_x - log_q_x) - (2*log_p_x - log_q_x).detach() + log_w.detach(), dim=0)
        else:
            log_w = log_w[valid_indices]
            log_q_x = log_q_x[valid_indices]
            return - torch.sum(F.softmax(log_w, dim=-1) * log_q_x)
            # log_p_x = log_p_x[valid_indices]
            # return torch.logsumexp((2*log_p_x - log_q_x) - (2*log_p_x - log_q_x).detach() + log_w.detach(), dim=0)

if __name__ == '__main__':
    from FittedModels.Models.FlowModel import FlowModel
    from FittedModels.utils.plotting_utils import plot_history
    import matplotlib.pyplot as plt
    from TargetDistributions.MoG import MoG
    from Utils.plotting_utils import plot_distribution
    from Utils.numerical_utils import MC_estimate_true_expectation
    from Utils.numerical_utils import quadratic_function as expectation_function
    from FittedModels.utils.plotting_utils import plot_samples


    torch.manual_seed(2)
    n_plots = 5
    epochs = 500
    step_size = 1.0
    batch_size = int(1e3)
    dim = 2
    n_samples_estimation = int(1e4)
    flow_type = "RealNVP"  # "ReverseIAF"  #"ReverseIAF_MIX" #"ReverseIAF" #IAF"  #
    n_flow_steps = 5
    HMC_transition_operator_args = {"step_tuning_method": "p_accept", "L":10} # "Expected_target_prob", "No-U", "p_accept"
    print(HMC_transition_operator_args)
    target = MoG(dim=dim, n_mixes=5, min_cov=1, loc_scaling=10)
    true_expectation = MC_estimate_true_expectation(target, expectation_function, int(1e5))
    fig = plot_distribution(target, bounds=[[-30, 20], [-20, 20]])
    plt.show()
    learnt_sampler = FlowModel(x_dim=dim, scaling_factor=1.0, flow_type=flow_type, n_flow_steps=n_flow_steps)
    tester = AIS_trainer(target, learnt_sampler, n_distributions=4,
                         transition_operator="HMC", lr=1e-2,
                         tranistion_operator_kwargs=HMC_transition_operator_args,
                         use_memory_buffer=False)
    plot_samples(tester)
    plt.show()

    def plotter(*args, **kwargs):
        plot_samples(*args, **kwargs)
        plt.show()

    expectation, info_dict = tester.AIS_train.calculate_expectation(n_samples_estimation,
                                                                        expectation_function=expectation_function)
    print(f"true expectation is {true_expectation}, estimated expectation is {expectation}")
    print(
        f"ESS is {info_dict['effective_sample_size'] / n_samples_estimation}, "
        f"var is {torch.var(info_dict['normalised_sampling_weights'])}")

    plt.figure()
    learnt_dist_samples = learnt_sampler.sample((1000,)).cpu().detach()
    plt.scatter(learnt_dist_samples[:, 0], learnt_dist_samples[:, 1])
    plt.title("approximating distribution samples")
    plt.show()
    plt.figure()
    plt.scatter(info_dict["samples"][:, 0].cpu(), info_dict["samples"][:, 1].cpu())
    plt.title("annealed samples")
    plt.show()
    plt.figure()
    true_samples = target.sample((1000,)).cpu().detach()
    plt.scatter(true_samples[:, 0], true_samples[:, 1])
    plt.title("true samples")
    plt.show()

    history = tester.train(epochs, batch_size=batch_size, intermediate_plots=True,
                           plotting_func=plotter, n_plots=n_plots, n_progress_updates=20)
    plot_history(history)
    plt.show()
    plot_samples(tester)
    plt.show()


    expectation, info_dict = tester.AIS_train.calculate_expectation(n_samples_estimation,
                                                                        expectation_function=expectation_function)
    print(f"AFTER TRAINING: \n true expectation is {true_expectation}, estimated expectation is {expectation}")
    print(
        f"ESS is {info_dict['effective_sample_size'] / n_samples_estimation}, "
        f"var is {torch.var(info_dict['normalised_sampling_weights'])}")