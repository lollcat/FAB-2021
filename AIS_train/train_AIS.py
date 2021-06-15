
Notebook = False
if Notebook:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
from FittedModels.train import LearntDistributionManager
from FittedModels.utils.plotting_utils import plot_samples
from AIS_train.AnnealedImportanceSampler import AnnealedImportanceSampler
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt


class AIS_trainer(LearntDistributionManager):
    """
    Merges annealed importance sampling into the training
    """
    def __init__(self, target_distribution, fitted_model,
                 n_distributions=10, n_steps_transition_operator=5, save_for_visualisation=False, save_spacing=20,
                 loss_type="kl", step_size=1.0, train_AIS_params=True, alpha=2, importance_param_lr=1e-2,
                 transition_operator="Metropolis", HMC_inner_steps=3,
                 learnt_dist_kwargs={}, AIS_kwargs={}):
        assert loss_type in ["kl", "DReG", "var"]
        self.AIS_train = AnnealedImportanceSampler(loss_type, train_AIS_params, fitted_model, target_distribution,
                                                   transition_operator=transition_operator,
                                                   n_distributions=n_distributions, n_steps_transition_operator=n_steps_transition_operator,
                                                   save_for_visualisation=save_for_visualisation, save_spacing=save_spacing,
                                                   step_size=step_size, HMC_inner_steps=HMC_inner_steps, **AIS_kwargs)
        super(AIS_trainer, self).__init__(target_distribution, fitted_model, self.AIS_train,
                 loss_type, alpha, **learnt_dist_kwargs)
        self.train_AIS_params = train_AIS_params
        if train_AIS_params:
            self.noise_optimizer = torch.optim.Adam([self.AIS_train.log_step_size], lr=importance_param_lr)

    def to(self, device):
        """device is cuda or cpu"""
        self.device = device
        self.learnt_sampling_dist.to(self.device)
        self.target_dist.to(self.device)
        if hasattr(self, "fixed_learnt_sampling_dist"):
            self.fixed_learnt_sampling_dist.to(self.device)
        if hasattr(self.AIS_train, "log_step_size"):
            self.AIS_train.log_step_size = nn.Parameter(self.AIS_train.log_step_size.to(device))

    def setup_loss(self, loss_type, alpha=2, k=None, new_lr=None, annealing=False):
        self.AIS_train.loss_type = loss_type
        self.annealing = annealing
        self.k = k  # if DReG then k is number of samples that going inside the log sum, if none then put all of them inside
        if loss_type == "kl":
            self.loss = self.KL_loss
            self.alpha = 1
        elif loss_type == "DReG":
            self.loss = self.dreg_alpha_divergence_loss
            self.alpha = alpha  # alpha for alpha-divergence
            self.alpha_one_minus_alpha_sign = torch.sign(torch.tensor(self.alpha * (1 - self.alpha)))
        elif loss_type == "var":
            self.loss = self.var_loss
        else:
            raise Exception("loss_type incorrectly specified")
        if new_lr is not None:
            self.optimizer.param_groups[0]["lr"] = new_lr


    def train(self, epochs=100, batch_size=1000, intermediate_plots=False,
              plotting_func=plot_samples, n_plots=10,
              KPI_batch_size=int(1e4)):
        epoch_per_print = max(int(epochs / 10), 1)
        epoch_per_save = max(int(epochs / 100), 1)
        if "DReG" in self.loss_type and self.k is None:
            self.k = batch_size
        if intermediate_plots is True:
            epoch_per_plot = max(int(epochs / n_plots), 1)
        history = {"loss": [],
                   "log_p_x_after_AIS": [],
                   "log_w": []}
        if self.train_AIS_params is True:
            history.update({"noise_scaling": []})
        history.update({
           "kl": [],
           "alpha_2_divergence": []
        })
        pbar = tqdm(range(epochs))
        for self.current_epoch in pbar:
            self.optimizer.zero_grad()
            x_samples, log_w = self.AIS_train.run(batch_size)
            loss = self.loss(log_w)
            if torch.isnan(loss) or torch.isinf(loss):
                raise Exception("NaN loss encountered")
            loss.backward()
            self.optimizer.step()
            if self.train_AIS_params:
                self.noise_optimizer.step()
            # save info
            log_p_x = self.target_dist.log_prob(x_samples)
            history["loss"].append(loss.item())
            history["log_p_x_after_AIS"].append(torch.mean(log_p_x).item())
            history["log_w"].append(torch.mean(log_w).item())
            if self.train_AIS_params is True:
                history["noise_scaling"].append(self.AIS_train.step_size.item())
            if self.current_epoch % epoch_per_print == 0 or self.current_epoch == epochs:
                pbar.set_description(f"loss: {history['loss'][-1]}, mean log p_x {torch.mean(log_p_x)}")
            if self.current_epoch % epoch_per_save == 0 or self.current_epoch == epochs:
                history["kl"].append(self.kl_MC_estimate(KPI_batch_size))
                history["alpha_2_divergence"].append(self.alpha_divergence_MC_estimate(KPI_batch_size))
            if intermediate_plots:
                if self.current_epoch % epoch_per_plot == 0:
                    plotting_func(self, n_samples=1000, title=f"training epoch, samples from flow {self.current_epoch}")
                    rows = int(self.learnt_sampling_dist.dim / 2)
                    fig, axs = plt.subplots(rows, sharex="all", sharey="all", figsize=(7, 3 * rows))
                    x_samples = x_samples.cpu().detach()  # for plotting
                    for row in range(rows):
                        if rows == 1:
                            ax = axs
                        else:
                            ax = axs[row]
                        if row == 0:
                            ax.set_title("plot of samples")
                        ax.scatter(x_samples[:, row], x_samples[:, row + 1])
                        ax.set_title(f"q(x) samples after AIS dim {row * 2}-{row * 2 + 1}")
                    plt.show()
        return history

    def dreg_alpha_divergence_loss(self, log_w):
        outside_dim = log_w.shape[0]/self.k  # this is like a batch dimension that we average DReG estimation over
        assert outside_dim % 1 == 0  # always make k & n_samples work together nicely for averaging
        outside_dim = int(outside_dim)
        log_w = log_w.reshape((outside_dim, self.k))
        with torch.no_grad():
            w_alpha_normalised_alpha = F.softmax(self.alpha*log_w, dim=-1)
        DreG_for_each_batch_dim = - self.alpha_one_minus_alpha_sign * \
                    torch.sum(((1 - self.alpha) * w_alpha_normalised_alpha + self.alpha * w_alpha_normalised_alpha**2)
                              * log_w, dim=-1)
        dreg_loss = torch.mean(DreG_for_each_batch_dim)
        return dreg_loss

    def KL_loss(self, log_w):
        kl = -log_w
        return torch.mean(kl)

    def var_loss(self, log_w):
        return torch.var(torch.exp(log_w))


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
    epochs = 5
    dim = 2
    n_samples_estimation = int(1e4)
    target = MoG(dim=dim, n_mixes=2, min_cov=1, loc_scaling=3)
    true_expectation = MC_estimate_true_expectation(target, expectation_function, int(1e5))
    fig = plot_distribution(target, bounds=[[-30, 20], [-20, 20]])
    plt.show()
    learnt_sampler = FlowModel(x_dim=dim, scaling_factor=3.0)  # , flow_type="RealNVP")
    tester = AIS_trainer(target, learnt_sampler, n_distributions=3, n_steps_transition_operator=3,
                         step_size=1.0, train_AIS_params=True, loss_type="kl",
                         transition_operator="HMC")
    plot_samples(tester)
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

    history = tester.train(epochs, batch_size=2000, intermediate_plots=True)
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




    """
    from TargetDistributions.BayesianNN import PosteriorBNN
    from FittedModels.Models.FlowModel import FlowModel
    from FittedModels.utils import plot_history
    import matplotlib.pyplot as plt

    epochs = 10
    target = PosteriorBNN(n_datapoints=10, x_dim=1, y_dim=1, n_hidden_layers=1, layer_width=1)
    dim = target.n_parameters
    learnt_sampler = FlowModel(x_dim=dim, flow_type="RealNVP")
    tester = AIS_trainer(target, learnt_sampler, loss_type="DReG", n_distributions=10, n_updates_Metropolis=3)
    history = tester.train(epochs, batch_size=200)
    plot_history(history)
    plt.show()
    """


