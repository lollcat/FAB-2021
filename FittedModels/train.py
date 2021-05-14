import torch
import torch.nn as nn
import torch.nn.functional as F
from FittedModels.Models.base import BaseLearntDistribution
from DebuggingUtils import check_gradients
Notebook = True
if Notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

class LearntDistributionManager:
    def __init__(self, target_distribution, fitted_model, importance_sampler,
                 loss_type="kl", alpha=2, lr=1e-3):
        self.importance_sampler = importance_sampler
        self.learnt_sampling_dist: BaseLearntDistribution
        self.learnt_sampling_dist = fitted_model
        self.target_dist = target_distribution
        self.optimizer = torch.optim.Adamax(self.learnt_sampling_dist.parameters(), lr=lr)
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


    def train(self, epochs=100, batch_size=256):
        epoch_per_print = max(int(epochs / 10), 1)
        history = {"loss": [],
                   "log_p_x": [],
                   "log_q_x": []}
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            self.optimizer.zero_grad()
            x_samples, log_q_x = self.learnt_sampling_dist(batch_size)
            log_p_x = self.target_dist.log_prob(x_samples)
            loss = self.loss(log_q_x, log_p_x)
            if True in torch.isnan(log_p_x) or True in torch.isinf(log_p_x):
                print("NaN/-inf loss encountered in log_p_x")
            if True in torch.isnan(log_q_x) or True in torch.isinf(log_q_x):
                print("NaN/-inf loss encountered in log_q_x")
            if torch.isnan(loss) or torch.isinf(loss):
                from FittedModels.utils import plot_history
                import matplotlib.pyplot as plt
                plot_history(history)
                plt.show()
                raise Exception(f"NaN loss encountered on epoch {epoch}")
            loss.backward()
            self.optimizer.step()
            history["loss"].append(loss.item())
            history["log_p_x"].append(torch.mean(log_p_x).item())
            history["log_q_x"].append(torch.mean(log_q_x).item())
            if epoch % epoch_per_print == 0 or epoch == epochs:
                pbar.set_description(f"loss: {history['loss'][-1]}, mean log p_x {torch.mean(log_p_x)}")
        return history

    def KL_loss(self, log_q_x, log_p_x, clamp=False):
        if clamp is True:
            log_q_x = log_q_x.clamp(min=-1)
        kl = log_q_x - log_p_x
        kl = torch.masked_select(kl, ~torch.isinf(kl) & ~torch.isnan(kl))
        kl_loss = torch.mean(kl)
        return kl_loss

    def dreg_alpha_divergence_loss(self, log_q_x, log_p_x):
        # summing all samples within the log
        #log_q_x = log_q_x.clamp(max= 2)
        log_w = log_p_x - log_q_x
        # prevent -inf from low density regions breaking things
        log_w = torch.masked_select(log_w, ~torch.isinf(log_w) & ~torch.isnan(log_w))
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


    def debugging_check_jacobians(self, loss, log_q_x, log_p_x, x_samples):
        first_level_differentiate = torch.autograd.grad(loss, log_q_x, retain_graph=True)
        last_level_differentiate = \
            torch.autograd.grad(loss,
                                self.learnt_sampling_dist.flow_blocks[
                                    0].AutoregressiveNN.FirstLayer.latent_to_layer.weight,
                                retain_graph=True) # gives nans
        first_to_mid_level_grad = torch.autograd.grad(torch.sum(log_q_x), self.learnt_sampling_dist.flow_blocks[
            0].AutoregressiveNN.FirstLayer.latent_to_layer.weight,
                                                      retain_graph=True)
        x_sample = x_samples[0, :]
        log_prob_x_sample = self.target_dist.log_prob(x_sample)
        reparam_now = torch.autograd.grad(log_prob_x_sample, x_sample,
                                                      retain_graph=True)
        reparam_equiv = torch.autograd.grad(log_p_x[0], x_samples[:, 0],
                                                      retain_graph=True)
        return

    def debugging(self, epochs=100, batch_size=256):
        # https://github.com/pytorch/pytorch/issues/15131
        #torch.manual_seed(1)
        history = {"loss": [],
                   "log_p_x": [],
                   "log_q_x": []}
        pbar = tqdm(range(epochs))
        #torch.autograd.set_detect_anomaly(True)
        for epoch in pbar:
            self.optimizer.zero_grad()
            x_samples, log_q_x = self.learnt_sampling_dist(batch_size)
            log_p_x = self.target_dist.log_prob(x_samples)
            loss = self.loss(log_q_x, log_p_x)

            #x_samples.register_hook(lambda grad: print("\n\nmin max grad x_samples", grad.min(), grad.max()))
            #log_p_x.register_hook(lambda grad: print("\n\nmin max grad log_p_x", grad.min(), grad.max()))
            #log_q_x.register_hook(lambda grad: print("\n\nmin max grad log_q_x", grad.min(), grad.max()))
            #self.learnt_sampling_dist.flow_blocks[
            #    0].AutoregressiveNN.FirstLayer.latent_to_layer.weight.register_hook(
            #    lambda grad: print("\n\nmax, min grad x_samples", grad.max(), grad.min()))
            #log_q_x.register_hook(lambda grad: print("\n\ngrad log_q_x", grad))

            #self.debugging_check_jacobians(loss, log_q_x, log_p_x, x_samples)
            if True in torch.isnan(log_p_x) or True in torch.isinf(log_p_x):
                print("NaN/-inf loss encountered in log_p_x")
            if True in torch.isnan(log_q_x) or True in torch.isinf(log_q_x):
                print("NaN/-inf loss encountered in log_q_x")
            if torch.isnan(loss) or torch.isinf(loss):
                from FittedModels.utils import plot_history
                import matplotlib.pyplot as plt
                plot_history(history)
                plt.show()
                raise Exception(f"NaN loss encountered on epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.learnt_sampling_dist.parameters(), 10)
            names, is_nan = check_gradients(self.learnt_sampling_dist.named_parameters())
            self.optimizer.step()
        return history


if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt
    from FittedModels.utils import plot_distributions
    torch.manual_seed(0)
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

    figure, axs = plt.subplots(len(history), 1, figsize=(6, 10))
    for i, key in enumerate(history):
        axs[i].plot(history[key])
        axs[i].set_title(key)
        if key == "alpha_divergence":
            axs[i].set_yscale("log")
    plt.show()

    plt.violinplot([sampling_weights])
    plt.yscale("log")

    print(f"means {tester.learnt_sampling_dist.means, tester.target_dist.loc}")
    print(f"learnt dist is scale tril {tester.learnt_sampling_dist.distribution.scale_tril}")
    print(f"target dist scale tril {tester.target_dist.scale_tril}")
    print(f"learnt dist log_std is {tester.learnt_sampling_dist.log_std}")
