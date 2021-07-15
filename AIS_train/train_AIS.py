
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
import numpy as np


class AIS_trainer(LearntDistributionManager):
    """
    Merges annealed importance sampling into the training
    """
    def __init__(self, target_distribution, fitted_model,
                 n_distributions=10, n_steps_transition_operator=5,
                 loss_type=False, loss_type_2="alpha_2", step_size= 1.0, train_AIS_params=False, alpha=2,
                 transition_operator="Metropolis", inner_loop_steps=5,
                 learnt_dist_kwargs={}, AIS_kwargs={}):
        self.loss_type = loss_type
        self.loss_type_2 = loss_type_2
        assert loss_type in [False, "kl", "DReG", "var", "ESS", "alpha_2_non_DReG"]
        assert loss_type_2 in [False, "kl", "alpha_2", "alpha_2_resample"]
        if train_AIS_params and transition_operator == "AIS":
            assert loss_type != "DReG"  # not able to back-prop through trainining HMC
        self.AIS_train = AnnealedImportanceSampler(loss_type, train_AIS_params, fitted_model, target_distribution,
                                                   transition_operator=transition_operator,
                                                   n_distributions=n_distributions,
                                                   n_steps_transition_operator=n_steps_transition_operator,
                                                   step_size=step_size, inner_loop_steps=inner_loop_steps, **AIS_kwargs)
        self.log_prob_annealed_scaling_factor = torch.tensor(1.0)
        super(AIS_trainer, self).__init__(target_distribution, fitted_model, self.AIS_train,
                 loss_type, alpha, **learnt_dist_kwargs)
        self.use_2nd_loss = loss_type_2   # add to loss function
        if loss_type_2 is not False:
            if loss_type_2 == "kl":
                self.loss_2 = self.log_prob_annealed_samples_loss_resample
            elif loss_type_2 == "alpha_2_resample":
                self.loss_2 = self.alpha_div_annealed_samples_re_sample
                self.alpha = 2
                self.alpha_one_minus_alpha_sign = torch.sign(torch.tensor(self.alpha * (1 - self.alpha)))
                # currently sharing a self.alpha param between losses so need to check for consistency here
                assert loss_type in [False, "DReG"]
            elif loss_type_2 == "alpha_2":
                self.loss_2 = self.alpha_div_annealed_samples_re_weight
                self.alpha = 2
                self.alpha_one_minus_alpha_sign = torch.sign(torch.tensor(self.alpha * (1 - self.alpha)))
                # currently sharing a self.alpha param between losses so need to check for consistency here
                assert loss_type in [False, "DReG"]

    def to(self, device):
        """device is cuda or cpu"""
        print(f"setting device as {device}")
        self.device = device
        self.learnt_sampling_dist.to(self.device)
        self.target_dist.to(self.device)
        if hasattr(self, "fixed_learnt_sampling_dist"):
            self.fixed_learnt_sampling_dist.to(self.device)
        self.AIS_train.to(device)


    def setup_loss(self, loss_type, alpha=2, k=None, new_lr=None, annealing=False):
        self.AIS_train.loss_type = loss_type
        self.annealing = annealing
        if loss_type == "kl":
            self.loss = self.KL_loss
            self.alpha = 1
        elif loss_type == "DReG":
            self.loss = self.dreg_alpha_divergence_loss
            self.alpha = alpha  # alpha for alpha-divergence
            self.alpha_one_minus_alpha_sign = torch.sign(torch.tensor(self.alpha * (1 - self.alpha)))
        elif loss_type == "var":
            self.loss = self.var_loss
        elif loss_type == "ESS":
            self.loss = self.ESS_loss
        elif loss_type == "alpha_2_non_DReG":
            self.loss = self.alpha_divergence_loss
            self.alpha = 2
            self.alpha_one_minus_alpha_sign = torch.sign(torch.tensor(self.alpha * (1 - self.alpha)))
        elif loss_type == False:
            self.loss = lambda x: torch.zeros(1).to(x.device)  # train likelihood only
        else:
            raise Exception("loss_type incorrectly specified")
        if new_lr is not None:
            self.optimizer.param_groups[0]["lr"] = new_lr


    def train(self, epochs=100, batch_size=1000, intermediate_plots=False,
              plotting_func=plot_samples, n_plots=3,
              KPI_batch_size=int(1e4),
              allow_ignore_nan_loss=True, clip_grad_norm=True,
              max_grad_norm=1, plotting_batch_size=int(1e3)):
        epoch_per_save_and_print = max(int(epochs / 100), 1)
        if intermediate_plots is True:
            epoch_per_plot = max(int(epochs / n_plots), 1)
        history = {"loss": [],
                   "log_p_x_after_AIS": [],
                   "log_w": [],
                   "kl": [],
                   "alpha_2_divergence": [],
                   "log_q_AIS_x": [],
                   "ESS": [],
                   }
        history.update(dict([(key, []) for key in self.AIS_train.transition_operator_class.interesting_info()]))
        if hasattr(self.target_dist, "sample"):
            history.update({'mean_log_prob_true_samples': []})
        elif hasattr(self.target_dist, "test_set"):
            history.update({'mean_log_q_x_test_samples': []})
        pbar = tqdm(range(epochs))
        for self.current_epoch in pbar:
            x_samples, log_w = self.AIS_train.run(batch_size)
            loss_1 = self.loss(log_w)
            if self.use_2nd_loss:
                loss_2, re_sampled_x = self.loss_2(x_samples, log_w)
                loss_1 = loss_1 + loss_2
            if torch.isnan(loss_1) or torch.isinf(loss_1):
                if allow_ignore_nan_loss:
                    print("Nan/Inf loss encountered in loss_1")
                    continue
                else:
                    raise Exception("Nan/Inf loss_1 encountered")
            self.optimizer.zero_grad()
            loss_1.backward()
            if clip_grad_norm is True:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.learnt_sampling_dist.parameters(), max_grad_norm)
                torch.nn.utils.clip_grad_value_(self.learnt_sampling_dist.parameters(), 1)
            self.optimizer.step()
            with torch.no_grad(): # none of the below steps should require gradients
                # save info
                history["loss"].append(loss_1.item())
                history["log_w"].append(torch.mean(log_w).item())
                history["ESS"].append(
                    self.AIS_train.effective_sample_size_unnormalised_log_weights(log_w, drop_nan=True).item() /
                    log_w.shape[0])
                transition_operator_info = self.AIS_train.transition_operator_class.interesting_info()
                for key in transition_operator_info:
                    history[key].append(transition_operator_info[key])
                if self.current_epoch % epoch_per_save_and_print == 0 or self.current_epoch == epochs:
                    history["kl"].append(self.kl_MC_estimate(KPI_batch_size))
                    history["alpha_2_divergence"].append(self.alpha_divergence_MC_estimate(KPI_batch_size))
                    history["log_q_AIS_x"].append(self.log_prob_annealed_samples_loss_resample(x_samples, log_w)[0].item())
                    if self.current_epoch > 0:
                        try:
                            log_p_x = self.target_dist.log_prob(x_samples).detach()
                            log_p_x = log_p_x[~(torch.isnan(log_p_x) | torch.isinf(log_p_x))]
                            history["log_p_x_after_AIS"].append(torch.mean(log_p_x).item())
                        except:
                            print("Couldn't calculate log prob over target distribution")
                            log_p_x = 0.0
                        if hasattr(self.target_dist, "sample"):
                            true_samples = self.target_dist.sample((batch_size,))
                            mean_log_q_x_true_samples = torch.mean(self.learnt_sampling_dist.log_prob(true_samples)).item()
                            history['mean_log_prob_true_samples'].append(mean_log_q_x_true_samples)
                            pbar.set_description(
                                f"loss: {np.mean(history['loss'][-epoch_per_save_and_print:])},"
                                f""f"mean_log_prob_true_samples {mean_log_q_x_true_samples},"
                                f"ESS {np.mean(history['ESS'][-epoch_per_save_and_print:])}")
                        elif hasattr(self.target_dist, "test_set"):
                            test_samples = self.target_dist.test_set(self.device)
                            mean_log_q_x_test_samples = torch.mean(self.learnt_sampling_dist.log_prob(test_samples)).item()
                            history['mean_log_q_x_test_samples'].append(mean_log_q_x_test_samples)
                            pbar.set_description(
                                f"loss: {np.mean(history['loss'][-epoch_per_save_and_print:])},"
                                f""f"mean_log_q_x_test_samples {mean_log_q_x_test_samples},"
                                f"ESS {np.mean(history['ESS'][-epoch_per_save_and_print:])}")
                        else:
                            pbar.set_description(
                                f"loss: {np.mean(history['loss'][-epoch_per_save_and_print:])},"
                                f""f"log_p_x_post_AIS {np.mean(history['log_p_x_after_AIS'][-epoch_per_save_and_print:])},"
                                f"ESS {np.mean(history['ESS'][-epoch_per_save_and_print:])}")
                if intermediate_plots:
                    if self.current_epoch % epoch_per_plot == 0:
                        plotting_func(self, n_samples=plotting_batch_size,
                                      title=f"training epoch, samples from flow {self.current_epoch}")
                        # make sure plotting func has option to enter x_samples directly
                        plotting_func(self, n_samples=batch_size,
                                      title=f"training epoch, samples from AIS {self.current_epoch}",
                                      samples_q=x_samples.cpu().detach())
                        if "re_sampled_x" in locals():
                            if re_sampled_x is not None:
                                plotting_func(self, n_samples=batch_size,
                                              title=f"training epoch, samples from AIS re-sampled {self.current_epoch}",
                                              samples_q=re_sampled_x.cpu().detach())
        return history

    def dreg_alpha_divergence_loss(self, log_w, drop_nans_and_infs=True):
        if drop_nans_and_infs:
            log_w = log_w[~(torch.isinf(log_w) | torch.isnan(log_w))]
        with torch.no_grad():
            w_alpha_normalised_alpha = F.softmax(self.alpha*log_w, dim=-1)
        DreG_for_each_batch_dim = - self.alpha_one_minus_alpha_sign * \
                    torch.sum(((1 - self.alpha) * w_alpha_normalised_alpha + self.alpha * w_alpha_normalised_alpha**2)
                              * log_w, dim=-1)
        dreg_loss = torch.mean(DreG_for_each_batch_dim)
        return dreg_loss

    def alpha_divergence_loss(self, log_w, drop_nans_and_infs=True):
        if drop_nans_and_infs:
            log_w = log_w[~(torch.isinf(log_w) | torch.isnan(log_w))]
        # no DReG
        return - self.alpha_one_minus_alpha_sign * (torch.logsumexp(self.alpha * log_w, dim=0) -
                                                    np.log(log_w.shape[0]))

    def KL_loss(self, log_w, drop_nans_and_infs=True):
        if drop_nans_and_infs:
            log_w = log_w[~(torch.isinf(log_w) | torch.isnan(log_w))]
        kl = -log_w
        return torch.mean(kl)


    def ESS_loss(self, log_w):
        return -self.AIS_train.effective_sample_size_unnormalised_log_weights(log_w)/log_w.shape[0]

    def var_loss(self, log_w):
        return torch.var(torch.exp(log_w))

    def log_prob_annealed_samples_loss_resample(self, x_samples, log_w):
        batch_size = x_samples.shape[0]
        # not we return - log_prob_annealed to get the loss
        # first remove samples that have inf/nan log w
        valid_indices = ~torch.isinf(log_w) & ~torch.isnan(log_w)
        if valid_indices.all():
            pass
        else:
            log_w = log_w[valid_indices]
            x_samples = x_samples[valid_indices, :]
        indx = torch.multinomial(torch.softmax(log_w, dim=0), num_samples=batch_size, replacement=True)
        x_samples = x_samples[indx, :]
        log_probs = self.learnt_sampling_dist.log_prob(x_samples.detach())

        # also check that we have valid log probs
        valid_indices = ~torch.isinf(log_probs) & ~torch.isnan(log_probs)
        if valid_indices.all():
            return -self.log_prob_annealed_scaling_factor*torch.mean(log_probs), x_samples
        else:  # placing no log_prob by some of the samples
            x_samples = x_samples[valid_indices, :]
            log_probs = log_probs[valid_indices]
            return -self.log_prob_annealed_scaling_factor * torch.mean(log_probs), x_samples

    def alpha_div_annealed_samples_re_sample(self, x_samples, log_w):
        # first remove samples that have inf/nan log w
        valid_indices = ~torch.isinf(log_w) & ~torch.isnan(log_w)
        if valid_indices.all():
            pass
        else:
            log_w = log_w[valid_indices]
            x_samples = x_samples[valid_indices, :]

        batch_size = x_samples.shape[0]
        indx = torch.multinomial(torch.softmax(log_w, dim=0), num_samples=batch_size, replacement=True)
        x_samples = x_samples[indx, :]
        log_q_x = self.learnt_sampling_dist.log_prob(x_samples.detach())
        log_p_x = self.target_dist.log_prob(x_samples.detach())

        # also check that we have valid log probs
        valid_indices = ~torch.isinf(log_q_x) & ~torch.isnan(log_q_x)
        if valid_indices.all():
            return - self.log_prob_annealed_scaling_factor * \
                   self.alpha_one_minus_alpha_sign*(torch.logsumexp((self.alpha - 1)*(log_p_x - log_q_x), dim=0)
                                                     - np.log(log_q_x.shape[0])), x_samples
        else:  # placing no log_prob by some of the samples
            x_samples = x_samples[valid_indices, :]
            log_q_x = log_q_x[valid_indices]
            log_p_x = log_p_x[valid_indices]
            return - self.log_prob_annealed_scaling_factor * \
                   self.alpha_one_minus_alpha_sign*(torch.logsumexp((self.alpha - 1) * (log_p_x - log_q_x), dim=0)
                                                     - np.log(log_q_x.shape[0])), x_samples


    def alpha_div_annealed_samples_re_weight(self, x_samples, log_w):
        # in this version we weight each term using log_w
        batch_size = x_samples.shape[0]
        # first remove samples that have inf/nan log w
        valid_indices = ~torch.isinf(log_w) & ~torch.isnan(log_w)
        if torch.sum(valid_indices) == 0: # no valid indices
            print("no valid indices")
            return torch.tensor(float("nan")), None
        if valid_indices.all():
            pass
        else:
            log_w = log_w[valid_indices]
            x_samples = x_samples[valid_indices, :]
        indx = torch.multinomial(torch.softmax(log_w, dim=0), num_samples=batch_size, replacement=True)
        x_re_sampled = x_samples[indx, :] # use so we can plot

        log_q_x = self.learnt_sampling_dist.log_prob(x_samples.detach())
        log_p_x = self.target_dist.log_prob(x_samples.detach())

        # also check that we have valid log probs
        valid_indices = ~torch.isinf(log_q_x) & ~torch.isnan(log_q_x)
        if valid_indices.all():
            return - self.log_prob_annealed_scaling_factor * \
                   self.alpha_one_minus_alpha_sign * \
                   torch.logsumexp((self.alpha - 1) * (log_p_x - log_q_x + log_w.detach()), dim=0), \
                   x_re_sampled
        else:
            log_w = log_w[valid_indices]
            log_q_x = log_q_x[valid_indices]
            log_p_x = log_p_x[valid_indices]
            return - self.log_prob_annealed_scaling_factor * \
                   self.alpha_one_minus_alpha_sign * \
                   torch.logsumexp((self.alpha - 1) * (log_p_x - log_q_x + log_w.detach()), dim=0), \
                   x_re_sampled


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
    epochs = 200
    step_size = 1.0
    batch_size = int(1e3)
    dim = 2
    n_samples_estimation = int(1e4)
    flow_type = "ReverseIAF"  #"ReverseIAF_MIX" #"ReverseIAF" #IAF"  # "RealNVP"
    n_flow_steps = 5
    from TargetDistributions.MoG import Difficult_MoG
    #target = Difficult_MoG(loc_scaling = 3.0, cov_scaling=1.0)
    target = MoG(dim=dim, n_mixes=5, min_cov=1, loc_scaling=10)
    true_expectation = MC_estimate_true_expectation(target, expectation_function, int(1e5))
    fig = plot_distribution(target, bounds=[[-30, 20], [-20, 20]])
    plt.show()
    learnt_sampler = FlowModel(x_dim=dim, scaling_factor=3.0, flow_type=flow_type, n_flow_steps=n_flow_steps)
    tester = AIS_trainer(target, learnt_sampler, n_distributions=6, n_steps_transition_operator=2,
                         step_size=step_size, train_AIS_params=True, loss_type=False, #"DReG",
                         transition_operator="HMC", learnt_dist_kwargs={"lr": 5e-4},
                         loss_type_2="alpha_2")
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
                           plotting_func=plotter, n_plots=n_plots)
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


