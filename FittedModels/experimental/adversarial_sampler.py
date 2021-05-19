import torch
import torch.nn as nn
import torch.nn.functional as F
from FittedModels.Models.base import BaseLearntDistribution
Notebook = False
if Notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
import numpy as np

class LearntDistributionManager:
    def __init__(self, target_distribution, fitted_model, adversarial_model, importance_sampler,
                 loss_type="kl", alpha=2):
        self.adversarial_model = adversarial_model
        self.importance_sampler = importance_sampler
        self.learnt_sampling_dist: BaseLearntDistribution
        self.learnt_sampling_dist = fitted_model
        self.target_dist = target_distribution
        self.optimizer_q = torch.optim.AdamW(self.learnt_sampling_dist.parameters())
        self.optimizer_g = torch.optim.AdamW(self.adversarial_model.parameters())
        self.loss_type = loss_type
        self.loss = self.alpha_divergence_loss

        if loss_type == "kl":
            print("TODO")
            self.alpha = 1
        elif loss_type == "DReG":
            print("TODO")
            self.alpha = alpha  # alpha for alpha-divergence
            self.alpha_one_minus_alpha_sign = torch.sign(torch.tensor(self.alpha * (1 - self.alpha)))
        else:
            raise Exception("loss_type incorrectly specified")


    def train(self, epochs=100, batch_size=32, n_switches=10, Train_adversarial_init=True, switcher=True):
        epoch_per_print = max(int(epochs / 10), 1)
        epoch_per_switch = max(int(epochs / n_switches), 1)
        Train_adversarial = Train_adversarial_init
        history = {"loss": [],
                   "log_p_x": [],
                   "log_q_x": [],
                   "log_g_x": []}
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            if (epoch + 1) % epoch_per_switch == 0:
                Train_adversarial = not Train_adversarial
            self.optimizer_q.zero_grad()
            self.optimizer_g.zero_grad()
            x_samples, log_g_x = self.adversarial_model(batch_size)
            log_q_x = self.learnt_sampling_dist.log_prob(x_samples)
            log_p_x = self.target_dist.log_prob(x_samples)
            loss_q = self.loss(log_q_x, log_p_x, log_g_x)
            loss_g = -loss_q
            if True in torch.isnan(log_p_x) or True in torch.isinf(loss_q):
                print("NaN/-inf loss encountered in log_p_x")
            if True in torch.isnan(log_q_x) or True in torch.isinf(log_q_x):
                print("NaN/-inf loss encountered in log_q_x")
            if True in torch.isnan(log_g_x) or True in torch.isinf(log_g_x):
                print("NaN/-inf loss encountered in log_g_x")
            if torch.isnan(loss_q) or torch.isinf(loss_q):
                from FittedModels.utils import plot_history
                import matplotlib.pyplot as plt
                plot_history(history)
                plt.show()
                raise Exception(f"NaN loss encountered on epoch {epoch}")
            if switcher is True:
                if Train_adversarial is False:
                    loss_q.backward()
                    self.optimizer_q.step()
                else:
                    loss_g.backward()
                    self.optimizer_g.step()
            else:
                loss_q.backward(retain_graph=True)
                loss_g.backward()
                self.optimizer_q.step()
                self.optimizer_g.step()
            history["loss"].append(loss_q.item())
            history["log_p_x"].append(torch.mean(log_p_x).item())
            history["log_q_x"].append(torch.mean(log_q_x).item())
            history["log_g_x"].append(torch.mean(log_g_x).item())
            if epoch % epoch_per_print == 0 or epoch == epochs:
                pbar.set_description(f"loss: {np.mean(history['loss'][-epoch_per_print:])}, "
                                     f"mean log p_x {np.mean(history['log_p_x'][-epoch_per_print:])}")
        return history

    def KL_loss(self, log_q_x, log_p_x, log_g_x):
        pass

    def alpha_divergence_loss(self, log_q_x, log_p_x, log_g_x):
        mc_alpha_div = 2*log_p_x - log_q_x - log_g_x
        mc_alpha_div = torch.masked_select(mc_alpha_div, ~torch.isinf(mc_alpha_div) & ~torch.isnan(mc_alpha_div))
        return torch.mean(mc_alpha_div)

    @torch.no_grad()
    def estimate_expectation(self, n_samples=int(1e4), expectation_function=lambda x: torch.sum(x, dim=-1)):
        importance_sampler = self.importance_sampler(self.learnt_sampling_dist, self.target_dist)
        expectation, normalised_sampling_weights = importance_sampler.calculate_expectation(n_samples, expectation_function)
        return expectation, normalised_sampling_weights

    def effective_sample_size(self, normalised_sampling_weights):
        return self.importance_sampler.effective_sample_size(normalised_sampling_weights)


def plot_distributions(learnt_dist_manager: LearntDistributionManager, range=10, n_points=100, title=""):
    from FittedModels.utils import plot_3D
    import itertools
    import matplotlib.pyplot as plt
    x_min = -range/2
    x_max = range/2
    x_points_1D = torch.linspace(x_min, x_max, n_points)
    x_points = torch.tensor(list(itertools.product(x_points_1D, repeat=2)))
    with torch.no_grad():
        q_x = torch.exp(learnt_dist_manager.learnt_sampling_dist.log_prob(x_points))
        p_x = torch.exp(learnt_dist_manager.target_dist.log_prob(x_points))
        g_x = torch.exp(learnt_dist_manager.adversarial_model.log_prob(x_points))

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    plot_3D(x_points, q_x, n_points, ax, title="q(x)")
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    plot_3D(x_points, p_x, n_points, ax, title="p(x)")
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    plot_3D(x_points, g_x, n_points, ax, title="g(x)")
    fig.suptitle(title)
    return fig

def plot_samples(learnt_dist_manager: LearntDistributionManager):
    samples_q = learnt_dist_manager.learnt_sampling_dist.sample((500,))
    samples_q = torch.clamp(samples_q , -100, 100).detach()
    samples_g = learnt_dist_manager.adversarial_model.sample((500,))
    samples_g = torch.clamp(samples_g, -100, 100).detach()
    samples_p = learnt_dist_manager.target_dist.sample((500, )).detach()
    fig, axs = plt.subplots(1, 3, sharex="all", sharey="all")
    axs[0].scatter(samples_q[:, 0], samples_q[:, 1])
    axs[0].set_title("q(x) samples")
    axs[1].scatter(samples_p[:, 0], samples_p[:, 1])
    axs[1].set_title("p(x) samples")
    axs[2].scatter(samples_g[:, 0], samples_g[:, 1])
    axs[2].set_title("g(x) samples")
    return fig


if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt
    from FittedModels.utils import plot_history
    #from FittedModels.utils import plot_distributions
    torch.manual_seed(5)
    from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
    from TargetDistributions.Guassian_FullCov import Guassian_FullCov
    from FittedModels.Models.DiagonalGaussian import DiagonalGaussian
    epochs = 5000
    dim = 2
    target = Guassian_FullCov(dim=dim, scale_covariance=4)
    learnt_sampler = DiagonalGaussian(dim=dim)
    adversarial_distribution = DiagonalGaussian(dim=dim)
    tester = LearntDistributionManager(target, learnt_sampler, adversarial_distribution, VanillaImportanceSampling)
    if dim == 2:
        fig_before = fig_before_train = plot_distributions(tester, title="before")
    expectation_before, sampling_weights_before = tester.estimate_expectation()
    plt.show()

    history = tester.train(epochs)
    expectation, info_dict = tester.estimate_expectation(int(1e5))

    true_expectation = torch.sum(tester.target_dist.mean)

    print(f"true expectation is {true_expectation} \n"
          f"estimate before training is {expectation_before} \n"
          f"estimate after training is {expectation}")

    if dim == 2:
        fig_after_train = plot_distributions(tester, title="after")
        plt.show()

    plot_history(history)

    plt.violinplot([info_dict["normalised_sampling_weights"]])
    plt.yscale("log")

    print(f"means {tester.learnt_sampling_dist.means, tester.target_dist.loc}")
    print(f"learnt dist is scale tril {tester.learnt_sampling_dist.distribution.scale_tril}")
    print(f"target dist scale tril {tester.target_dist.scale_tril}")
    print(f"learnt dist log_std is {tester.learnt_sampling_dist.log_std}")
