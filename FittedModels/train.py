import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from FittedModels.Models.base import BaseLearntDistribution
from copy import deepcopy
from DebuggingUtils import check_gradients
Notebook = False
if Notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class LearntDistributionManager:
    def __init__(self, target_distribution, fitted_model, importance_sampler,
                 loss_type="kl", alpha=2, lr=1e-3, k=None, use_GPU=True, optimizer="Adam",
                 annealing=False):
        if use_GPU is True:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self.importance_sampler = importance_sampler
        self.learnt_sampling_dist: BaseLearntDistribution
        self.learnt_sampling_dist = fitted_model.to(self.device)
        self.target_dist = target_distribution.to(self.device)
        self.loss_type = loss_type
        if optimizer == "Adam":
            torch_optimizer = torch.optim.Adam
        elif optimizer == "Adamax":
                torch_optimizer = torch.optim.Adamax
        else:
            raise Exception(f"optimizer type: '{optimizer}' not recognised")
        self.optimizer = torch_optimizer(self.learnt_sampling_dist.parameters(), lr=lr)
        self.setup_loss(loss_type, alpha, k, annealing=annealing)
        #self.fixed_learnt_sampling_dist = deepcopy(self.learnt_sampling_dist) # deep copy and weight norm not compatible


    def setup_loss(self, loss_type, alpha=2, k=None, new_lr=None, annealing=False):
        if loss_type == "kl":
            self.loss = self.KL_loss
            self.alpha = 1
            self.annealing = annealing
        elif loss_type == "DReG":
            self.loss = self.dreg_alpha_divergence_loss
            self.alpha = alpha  # alpha for alpha-divergence
            self.alpha_one_minus_alpha_sign = torch.sign(torch.tensor(self.alpha * (1 - self.alpha)))
            self.k = k  # number of samples that going inside the log sum, if none then put all of them inside
        elif loss_type == "DReG_kl":
            self.loss = self.dreg_kl_loss
            self.alpha = 1
            self.k = k  # number of samples that going inside the log sum, if none then put all of them inside

        elif loss_type == "alpha_MC":  # this does terribly
            self.loss = self.alpha_MC_loss
            self.alpha = alpha  # alpha for alpha-divergence
            self.alpha_one_minus_alpha_sign = torch.sign(torch.tensor(self.alpha * (1 - self.alpha)))
        else:
            raise Exception("loss_type incorrectly specified")
        if new_lr is not None:
            self.optimizer.param_groups[0]["lr"] = new_lr
        if "DReG" in loss_type:
            self.fixed_learnt_sampling_dist = type(self.learnt_sampling_dist)(
                *self.learnt_sampling_dist.class_args, **self.fixed_learnt_sampling_dist.class_kwargs)\
                .to(device=self.device)  # for computing d weights dz

    def train(self, epochs=100, batch_size=256, extra_info=True,
              clip_grad_norm=False, max_grad_norm=2, clip_grad_max=False,
              max_grad_value=1,
              KPI_batch_size=int(1e4), inner_batch_size=int(1e5)):
        """
        :param epochs:
        :param batch_size:
        :param extra_info: print MC estimates of divergences, and importance sampling info
        :param clip_grad_norm: max norm gradient clipping
        :param max_grad_norm: for gradient clipping
        :param KPI_batch_size:  n_samples used for MC estimates of divergences and importance sampling info
        :param inner_batch_size: batch size for each forward pass of the model
        :return: dictionary of training history
        """

        if "DReG" in self.loss_type and self.k is None:
            self.k = batch_size
        self.total_epochs = epochs  # we need this for annealing if we use it

        epoch_per_print = max(int(epochs / 20), 1)
        epoch_per_save = max(int(epochs / 100), 1)
        history = {"loss": [],
                   "log_p_x": [],
                   "log_q_x": []}
        if extra_info is True:
            history.update({
               "kl": [],
               "alpha_2_divergence": [],
               "importance_weights_var": [],
               "normalised_importance_weights_var": [],
                "effective_sample_size": []})
            if hasattr(self.target_dist, "sample"):
                history.update({"alpha_2_divergence_over_p": []})

        pbar = tqdm(range(epochs), position=0, leave=True)
        for self.current_epoch in pbar:
            self.optimizer.zero_grad()
            x_samples, log_q_x = self.learnt_sampling_dist(batch_size)
            log_p_x = self.target_dist.log_prob(x_samples)
            loss = self.loss(x_samples, log_q_x, log_p_x)
            loss.backward()
            if clip_grad_max is True:
                torch.nn.utils.clip_grad_value_(self.learnt_sampling_dist.parameters(), max_grad_value)
            if clip_grad_norm is True:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.learnt_sampling_dist.parameters(), max_grad_norm)
            self.optimizer.step()
            history["loss"].append(loss.item())
            history["log_p_x"].append(torch.mean(log_p_x).item())
            history["log_q_x"].append(torch.mean(log_q_x).item())
            if self.current_epoch % epoch_per_print == 0 or self.current_epoch == epochs:
                pbar.set_description(f"loss: {history['loss'][-1]}, mean log p_x {torch.mean(log_p_x)}")
            if self.current_epoch % epoch_per_save == 0 or self.current_epoch == epochs:
                history["kl"].append(self.kl_MC_estimate(KPI_batch_size))
                history["alpha_2_divergence"].append(self.alpha_divergence_MC_estimate(KPI_batch_size))
                if hasattr(self.target_dist, "sample"):  # check if sample func exists
                    try:
                        history["alpha_2_divergence_over_p"].append(self.alpha_divergence_over_p_MC_estimate(KPI_batch_size))
                    except:
                        print("Couldn't calculate alpha divergence over p")
                importance_weights_var, normalised_importance_weights_var, ESS = self.importance_weights_key_info(KPI_batch_size)
                history["importance_weights_var"].append(importance_weights_var)
                history["normalised_importance_weights_var"].append(normalised_importance_weights_var)
                history["effective_sample_size"].append(ESS)
        return history

    def to(self, device):
        """device is cuda or cpu"""
        self.device = device
        self.learnt_sampling_dist.to(self.device)
        self.target_dist.to(self.device)
        if hasattr(self, "fixed_learnt_sampling_dist"):
            self.fixed_learnt_sampling_dist.to(self.device)

    def KL_loss(self, x_samples_not_used, log_q_x, log_p_x):
        kl = log_q_x - log_p_x*self.beta
        kl_loss = torch.mean(kl)
        return kl_loss

    @property
    def beta(self):
        if self.annealing is False:
            return 1.0
        else:
            annealing_period = int(self.total_epochs)/2  #  anneal during first half of training
            return min(1.0, 0.01 + self.current_epoch/annealing_period)

    def dreg_alpha_divergence_loss(self, x_samples, log_q_x_not_used, log_p_x):
        self.update_fixed_version_of_learnt_distribution()
        log_q_x = self.fixed_learnt_sampling_dist.log_prob(x_samples)
        log_w = log_p_x - log_q_x
        outside_dim = log_q_x.shape[0]/self.k  # this is like a batch dimension that we average DReG estimation over
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

    def dreg_kl_loss(self, x_samples, log_q_x_not_used, log_p_x):
        self.update_fixed_version_of_learnt_distribution()
        log_q_x = self.fixed_learnt_sampling_dist.log_prob(x_samples)
        log_w = log_p_x - log_q_x
        outside_dim = log_q_x.shape[0]/self.k  # this is like a batch dimension that we average DReG estimation over
        assert outside_dim % 1 == 0  # always make k & n_samples work together nicely for averaging
        outside_dim = int(outside_dim)
        log_w = log_w.reshape((outside_dim, self.k))
        with torch.no_grad():
            w_normalised_squared = F.softmax(log_w, dim=-1)**2
        DreG_for_each_batch_dim = - torch.sum(w_normalised_squared * log_w, dim=-1)
        dreg_loss = torch.mean(DreG_for_each_batch_dim)
        return dreg_loss

    def importance_weights_key_info(self, batch_size=1000):
        x_samples, log_q_x = self.learnt_sampling_dist(batch_size)
        log_p_x = self.target_dist.log_prob(x_samples)
        # variance in unnormalised weights
        weights = torch.exp(log_p_x - log_q_x)
        normalised_weights = torch.softmax(log_p_x - log_q_x, dim=-1)
        ESS = self.importance_sampler.effective_sample_size(normalised_weights)/batch_size
        return torch.var(weights).item(), torch.var(normalised_weights).item(), ESS.item()

    def get_gradients(self, n_batches=100, batch_size=100):
        grads = []
        for i in range(n_batches):
            self.optimizer.zero_grad()
            x_samples, log_q_x = self.learnt_sampling_dist(batch_size)
            log_p_x = self.target_dist.log_prob(x_samples)
            loss = self.loss(x_samples, log_q_x, log_p_x)
            self.learnt_sampling_dist.flow_blocks[0].AutoregressiveNN.FinalLayer.layer_to_m.weight.register_hook(
                lambda grad: grads.append(grad.detach())
            )
            loss.backward()
        grads = torch.stack(grads)
        return grads

    def alpha_MC_loss(self, x_samples_not_used, log_q_x, log_p_x):
        alpha_div = -self.alpha_one_minus_alpha_sign*self.alpha*(log_p_x - log_q_x)
        MC_loss = torch.mean(alpha_div)
        return MC_loss

    def kl_MC_estimate(self, batch_size=1000):
        x_samples, log_q_x = self.learnt_sampling_dist(batch_size)
        log_p_x = self.target_dist.log_prob(x_samples)
        kl = log_q_x - log_p_x
        kl_loss = torch.mean(kl)
        return kl_loss.item()


    def alpha_divergence_MC_estimate(self, batch_size=1000, alpha=2):
        alpha_one_minus_alpha_sign = torch.sign(torch.tensor(alpha * (1 - alpha)))
        x_samples, log_q_x = self.learnt_sampling_dist(batch_size)
        log_p_x = self.target_dist.log_prob(x_samples)
        N = torch.tensor(log_p_x.shape[0])
        log_alpha_divergence = -alpha_one_minus_alpha_sign * \
                               (torch.logsumexp(alpha*(log_p_x - log_q_x), dim=-1) - torch.log(N))
        return log_alpha_divergence.item()

    def alpha_divergence_over_p_MC_estimate(self, batch_size=1000, alpha=2):
        alpha_one_minus_alpha_sign = torch.sign(torch.tensor(alpha * (1 - alpha)))
        x_samples = self.target_dist.sample((batch_size,))
        log_q_x = self.learnt_sampling_dist.log_prob(x_samples)
        log_p_x = self.target_dist.log_prob(x_samples)
        N = torch.tensor(log_p_x.shape[0])
        log_alpha_divergence = -alpha_one_minus_alpha_sign * \
                               (torch.logsumexp((alpha - 1) * (log_p_x - log_q_x), dim=-1) - torch.log(N))
        return log_alpha_divergence.item()

    def update_fixed_version_of_learnt_distribution(self):
        self.fixed_learnt_sampling_dist.load_state_dict(self.learnt_sampling_dist.state_dict())

    @torch.no_grad()
    def estimate_expectation(self, n_samples, expectation_function, device="cpu"):
        # run on cpu to handle big batch size, we could also use our batchify function here instead
        original_device = self.device
        self.to(device)
        importance_sampler = self.importance_sampler(self.learnt_sampling_dist, self.target_dist)
        expectation, expectation_info = importance_sampler.calculate_expectation(n_samples, expectation_function)
        self.to(original_device)
        return expectation, expectation_info

    def effective_sample_size(self, normalised_sampling_weights):
        return self.importance_sampler.effective_sample_size(normalised_sampling_weights)

    def compare_loss_gradients(self, n_batches=100, batch_size=100):
        self.k = batch_size  # only used in DReG
        self.alpha = 2  # this is only used in alpha div calc, so doesn't effect other losses
        self.alpha_one_minus_alpha_sign = torch.sign(torch.tensor(self.alpha * (1 - self.alpha)))
        alpha_2_grads_1 = []
        alpha_2_grads_2 = []
        kl_DReG_grads_1 = []
        kl_DReG_grads_2 = []
        kl_grads_1 = []
        kl_grads_2 = []

        param_set_1 = lambda: self.learnt_sampling_dist.flow_blocks[0].AutoregressiveNN.FirstLayer.latent_to_layer.weight
        param_set_2 = lambda: self.learnt_sampling_dist.flow_blocks[0].AutoregressiveNN.FinalLayer.layer_to_m.weight

        for i in range(n_batches):
            self.optimizer.zero_grad()
            x_samples, log_q_x = self.learnt_sampling_dist(batch_size)
            log_p_x = self.target_dist.log_prob(x_samples)

            # alpha 2 divergence
            alpha_2_loss = self.dreg_alpha_divergence_loss(x_samples, log_q_x, log_p_x)
            alpha_2_grads_1.append(torch.autograd.grad(alpha_2_loss, param_set_1(), retain_graph=True)[0].cpu().numpy())
            alpha_2_grads_2.append(torch.autograd.grad(alpha_2_loss, param_set_2(), retain_graph=True)[0].cpu().numpy())

            # kl DReG
            kl_DReG_loss = self.dreg_kl_loss(x_samples, log_q_x, log_p_x)
            kl_DReG_grads_1.append(torch.autograd.grad(kl_DReG_loss, param_set_1(), retain_graph=True)[0].cpu().numpy())
            kl_DReG_grads_2.append(torch.autograd.grad(kl_DReG_loss, param_set_2(), retain_graph=True)[0].cpu().numpy())

            # kl
            kl_loss = self.KL_loss(x_samples, log_q_x, log_p_x)
            kl_grads_1.append(torch.autograd.grad(kl_loss, param_set_1(), retain_graph=True)[0].cpu().numpy())
            kl_grads_2.append(torch.autograd.grad(kl_loss, param_set_2(), retain_graph=True)[0].cpu().numpy())

        alpha_2_grads_1 = np.stack(alpha_2_grads_1)
        alpha_2_grads_2 = np.stack(alpha_2_grads_2)
        kl_DReG_grads_1 = np.stack(kl_DReG_grads_1)
        kl_DReG_grads_2 = np.stack(kl_DReG_grads_2)
        kl_grads_1 = np.stack(kl_grads_1)
        kl_grads_2 = np.stack(kl_grads_2)

        return alpha_2_grads_1, alpha_2_grads_2, kl_DReG_grads_1, kl_DReG_grads_2, kl_grads_1, kl_grads_2




if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt
    from FittedModels.utils import plot_distributions
    torch.manual_seed(0)
    from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
    from TargetDistributions.Guassian_FullCov import Guassian_FullCov
    from FittedModels.Models.DiagonalGaussian import DiagonalGaussian
    from FittedModels.utils import plot_distributions
    epochs = 500
    dim = 2
    target = Guassian_FullCov(dim=dim)
    learnt_sampler = DiagonalGaussian(dim=dim)
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type="DReG")
    
    if dim == 2:
        fig_before = fig_before_train = plot_distributions(tester)
    expectation_before, sampling_weights_before = tester.estimate_expectation()
    plt.show()

    history = tester.train(epochs)
    expectation, expectation_info = tester.estimate_expectation(int(1e5))

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



