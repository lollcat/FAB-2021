import torch
import torch.nn as nn
import torch.nn.functional as F
from FittedModels.Models.base import BaseLearntDistribution
import copy
from DebuggingUtils import check_gradients
Notebook = False
if Notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

class LearntDistributionManager:
    def __init__(self, target_distribution, fitted_model, importance_sampler,
                 loss_type="kl", alpha=2, lr=1e-3, k=None):
        self.importance_sampler = importance_sampler
        self.learnt_sampling_dist: BaseLearntDistribution
        self.learnt_sampling_dist = fitted_model
        self.target_dist = target_distribution
        self.optimizer = torch.optim.Adam(self.learnt_sampling_dist.parameters(), lr=lr)
        self.loss_type = loss_type
        if loss_type == "kl":
            self.loss = self.KL_loss
            self.alpha = 1
        elif loss_type == "DReG":
            self.loss = self.dreg_alpha_divergence_loss
            self.alpha = alpha  # alpha for alpha-divergence
            self.alpha_one_minus_alpha_sign = torch.sign(torch.tensor(self.alpha * (1 - self.alpha)))
            self.k = k  # number of samples that going inside the log sum, if none then put all of them inside
        elif loss_type == "alpha_MC":
            self.loss = self.alpha_MC_loss
            self.alpha = alpha  # alpha for alpha-divergence
            self.alpha_one_minus_alpha_sign = torch.sign(torch.tensor(self.alpha * (1 - self.alpha)))
            self.k = k  # number of samples that going inside the log sum, if none then put all of them inside
        else:
            raise Exception("loss_type incorrectly specified")

        self.fixed_learnt_sampling_dist = type(fitted_model)(*fitted_model.class_definition)  # for computing d weights dz

    def train(self, epochs=100, batch_size=256, extra_info=True,
              clip_grad=False, max_grad_norm=2, break_on_inf=True):
        if self.loss_type == "DReG" and self.k is None:
            self.k = batch_size
        epoch_per_print = max(int(epochs / 20), 1)
        epoch_per_save = max(int(epochs / 100), 1)
        history = {"loss": [],
                   "log_p_x": [],
                   "log_q_x": []}
        if extra_info is True:
            history.update({
               "kl": [],
               "alpha_2_divergence": [],
                "alpha_2_divergence_over_p": [],
               "importance_weights_var": [],
               "normalised_importance_weights_var": [],
            "effective_sample_size": []})
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            self.optimizer.zero_grad()
            x_samples, log_q_x = self.learnt_sampling_dist(batch_size)
            log_p_x = self.target_dist.log_prob(x_samples)
            loss = self.loss(x_samples, log_q_x, log_p_x)
            if break_on_inf is False and torch.isinf(loss):
                print("continuing run after getting infinity loss")
                loss = torch.clip(loss, -1e16, 1e16)
            self.check_infs_and_NaNs(epoch, log_p_x, log_q_x, loss, history)
            loss.backward()
            if clip_grad is True:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.learnt_sampling_dist.parameters(), max_grad_norm)
                # torch.nn.utils.clip_grad_value_(self.learnt_sampling_dist.parameters(), 0.001)
            self.optimizer.step()
            history["loss"].append(loss.item())
            history["log_p_x"].append(torch.mean(log_p_x).item())
            history["log_q_x"].append(torch.mean(log_q_x).item())
            if epoch % epoch_per_print == 0 or epoch == epochs:
                pbar.set_description(f"loss: {history['loss'][-1]}, mean log p_x {torch.mean(log_p_x)}")
            if epoch % epoch_per_save == 0 or epoch == epochs:
                history["kl"].append(self.kl_MC_estimate())
                if hasattr(self.target_dist, "sample"):  # check if sample func exists
                    history["alpha_2_divergence"].append(self.alpha_divergence_MC_estimate())
                    try:
                        history["alpha_2_divergence_over_p"].append(self.alpha_divergence_over_p_MC_estimate())
                    except:
                        print("Couldn't calculate alpha divergence over p")
                importance_weights_var, normalised_importance_weights_var, ESS = self.importance_weights_key_info()
                history["importance_weights_var"].append(importance_weights_var)
                history["normalised_importance_weights_var"].append(normalised_importance_weights_var)
                history["effective_sample_size"].append(ESS)
        return history

    def KL_loss(self, x_samples_not_used, log_q_x, log_p_x):
        kl = log_q_x - log_p_x
        # kl = torch.masked_select(kl, ~torch.isinf(kl) & ~torch.isnan(kl))
        kl_loss = torch.mean(kl)
        return kl_loss

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
        alpha_div = alpha_div.clamp(-1e16, 1e16)
        alpha_div = torch.masked_select(alpha_div, ~torch.isinf(alpha_div) & ~torch.isnan(alpha_div))
        MC_loss = torch.mean(alpha_div)
        return MC_loss

    def kl_MC_estimate(self, batch_size=1000):
        x_samples, log_q_x = self.learnt_sampling_dist(batch_size)
        log_p_x = self.target_dist.log_prob(x_samples)
        kl = log_q_x - log_p_x
        kl = torch.masked_select(kl, ~torch.isinf(kl) & ~torch.isnan(kl))
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

    def check_infs_and_NaNs(self, epoch, log_p_x, log_q_x, loss, history):
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


    def update_fixed_version_of_learnt_distribution(self):
        self.fixed_learnt_sampling_dist.load_state_dict(self.learnt_sampling_dist.state_dict())

    @torch.no_grad()
    def estimate_expectation(self, n_samples=int(1e4), expectation_function=lambda x: torch.sum(x, dim=-1)):
        importance_sampler = self.importance_sampler(self.learnt_sampling_dist, self.target_dist)
        expectation, normalised_sampling_weights = importance_sampler.calculate_expectation(n_samples, expectation_function)
        return expectation, normalised_sampling_weights

    def effective_sample_size(self, normalised_sampling_weights):
        return self.importance_sampler.effective_sample_size(normalised_sampling_weights)

    def debugging_check_dreg(self, x_samples):
        # checks that we have some gradient!
        self.update_fixed_version_of_learnt_distribution()
        log_q_x = self.fixed_learnt_sampling_dist.log_prob(x_samples)
        reparam_equiv = torch.autograd.grad(log_q_x[0], x_samples,
                                            retain_graph=True)
        pass


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


