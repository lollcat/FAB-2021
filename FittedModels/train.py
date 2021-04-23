import torch
import torch.nn as nn
import torch.nn.functional as F
from FittedModels.Models.base import BaseLearntDistribution
Notebook = False
if Notebook:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

class LearntDistributionManager:
    def __init__(self, target_distribution, fitted_model, importance_sampler,
                 loss_type="kl", alpha=2):
        self.importance_sampler = importance_sampler
        self.learnt_sampling_dist: BaseLearntDistribution
        self.learnt_sampling_dist = fitted_model
        self.target_dist = target_distribution
        self.optimizer = torch.optim.Adam(self.learnt_sampling_dist.parameters())
        self.loss_type = loss_type
        if loss_type == "kl":
            self.loss = self.KL_loss
            self.alpha = 1
        elif loss_type == "DReG":
            self.loss = self.dreg_alpha_divergence_loss
            self.alpha = alpha  # alpha for alpha-divergence
            self.alpha_one_minus_alpha_sign = torch.sign(torch.tensor(self.alpha * (1 - self.alpha)))
        else:
            raise Exception("loss_type incorrectly specified")


    def train(self, epochs=100, batch_size=1000):
        epoch_per_print = int(epochs/10)
        history = {"loss": []}
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            self.optimizer.zero_grad()
            x_samples, log_q_x = self.learnt_sampling_dist(batch_size)
            log_p_x = self.target_dist.log_prob(x_samples)
            loss = self.loss(log_q_x, log_p_x)
            loss.backward()
            self.optimizer.step()
            history["loss"].append(loss.item())
            if epoch % epoch_per_print == 0 or epoch == epochs:
                pbar.set_description(f"loss: {history['loss'][-1]}")
                #for key in history:
                #    print(f"{key}: {history[key][-1]}")
        return history

    def KL_loss(self, log_q_x, log_p_x):
        kl = log_q_x - log_p_x
        return torch.mean(kl)

    def dreg_alpha_divergence_loss(self, log_q_x, log_p_x):
        # summing all samples within the log
        log_w = log_p_x - log_q_x
        with torch.no_grad():
            w_alpha_normalised_alpha = F.softmax(self.alpha*log_w, dim=-1)
        return torch.sum(((1 + self.alpha) * w_alpha_normalised_alpha + self.alpha * w_alpha_normalised_alpha**2) * log_w)

    @torch.no_grad()
    def estimate_expectation(self, n_samples=int(1e4), expectation_function=lambda x: torch.sum(x, dim=-1)):
        importance_sampler = self.importance_sampler(self.learnt_sampling_dist, self.target_dist)
        expectation, normalised_sampling_weights = importance_sampler.calculate_expectation(n_samples, expectation_function)
        return expectation, normalised_sampling_weights

    def effective_sample_size(self, normalised_sampling_weights):
        return self.importance_sampler.effective_sample_size(normalised_sampling_weights)


if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt
    from FittedModels.utils import plot_distributions
    torch.manual_seed(5)
    from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
    from TargetDistributions.Guassian_FullCov import Guassian_FullCov
    from FittedModels.Models.DiagonalGaussian import DiagonalGaussian
    from FittedModels.utils import plot_distributions
    epochs = 5000
    dim = 2
    target = Guassian_FullCov(dim=dim)
    learnt_sampler = DiagonalGaussian(dim=dim)
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type="DReG")
    if dim == 2:
        fig_before = fig_before_train = plot_distributions(tester)
    expectation_before, sampling_weights_before = tester.estimate_expectation()
    plt.show()

    history = tester.train(epochs)
    expectation, sampling_weights = tester.estimate_expectation(int(1e5))

    true_expectation = torch.sum(tester.target_dist.mean)

    print(f"true expectation is {true_expectation} \n"
          f"estimate before training is {expectation_before} \n"
          f"estimate after training is {expectation}")

    if dim == 2:
        fig_after_train = plot_distributions(tester)
        plt.show()
    plt.plot(history["loss"])
    plt.show()
    plt.violinplot([sampling_weights])
    plt.yscale("log")
    #tester.train(2)
    print(f"means {tester.learnt_sampling_dist.means, tester.target_dist.loc}")
    print(f"learnt dist is scale tril {tester.learnt_sampling_dist.distribution.scale_tril}")
    print(f"target dist scale tril {tester.target_dist.scale_tril}")
    print(f"learnt dist log_std is {tester.learnt_sampling_dist.log_std}")
