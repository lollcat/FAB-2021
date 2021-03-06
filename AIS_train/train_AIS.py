import pandas as pd

from FittedModels.train import LearntDistributionManager
from FittedModels.utils.plotting_utils import plot_samples
from AIS_train.AnnealedImportanceSampler import AnnealedImportanceSampler
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from Utils.numerical_utils import quadratic_function as expectation_function
import pathlib
from collections import deque
from FittedModels.utils.model_utils import sample_and_log_w_big_batch_drop_nans
from Utils.plotting_utils import add_to_history_dict


class AIS_trainer(LearntDistributionManager):
    """
    Merges annealed importance sampling into the training
    """
    def __init__(self, target_distribution, fitted_model,
                 n_distributions=2+2, loss_type="alpha_2_IS", transition_operator="HMC",
                 AIS_kwargs={}, tranistion_operator_kwargs={}, use_GPU = True,
                 optimizer="AdamW", lr=1e-3, use_memory_buffer=False,
                 memory_n_batches=100, allow_ignore_nan_loss=True, clip_grad_norm=True,
                 alpha=2.0
                 ):
        assert loss_type in ["alpha_2_IS", "alpha_2_q", "kl_q", "kl_p", "alpha_2_NIS"]
        self.loss_type = loss_type
        self.AIS_train = AnnealedImportanceSampler(fitted_model, target_distribution,
                                                   transition_operator=transition_operator,
                                                   n_distributions=n_distributions,
                                                   **AIS_kwargs,
                                                   transition_operator_kwargs=tranistion_operator_kwargs)
        if use_GPU is True:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self.allow_ignore_nan_loss = allow_ignore_nan_loss
        self.clip_grad_norm = clip_grad_norm
        self.max_grad_norm = 1.0
        self.alpha = alpha
        self.alpha_one_minus_alpha_sign = np.sign((1 - self.alpha)*self.alpha)
        self.importance_sampler = self.AIS_train
        self.learnt_sampling_dist = fitted_model
        self.target_dist = target_distribution
        if loss_type == "alpha_2_IS": # main method
            self.loss = self.alpha_div_annealed_samples_re_weight
            assert self.alpha == 2
        elif loss_type == "alpha_2_q":
            assert self.AIS_train.n_distributions < 3
            self.loss = lambda x_samples, log_w: torch.logsumexp(2*log_w, dim=-1)
        elif loss_type == "kl_p":  # as used in Neural Importance Sampling Paper
            self.loss = lambda x_samples, log_w: \
                -torch.mean(torch.exp(log_w.detach()) * self.learnt_sampling_dist.log_prob(x_samples))
        elif loss_type == "alpha_2_NIS":
            self.loss = lambda x_samples, log_w: -torch.mean(torch.exp(2*log_w.detach())* \
                                                 self.learnt_sampling_dist.log_prob(x_samples))
        else:
            assert loss_type == "kl_q"
            assert self.AIS_train.n_distributions < 3
            self.loss = lambda x_samples, log_w: -torch.mean(log_w)
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
            self.n_gradient_update_batches = int(self.max_memory_batches / 20)  # update using a 10_th of the memory


    def to(self, device):
        """device is cuda or cpu"""
        print(f"setting device as {device}")
        self.device = device
        self.learnt_sampling_dist.to(self.device)
        self.target_dist.to(self.device)
        self.AIS_train.to(device)

    @property
    def n_memory_samples(self):
        return self.memory_buffer_x.shape[0]

    def add_to_memory(self, x_samples, log_w):
        batch_size = x_samples.shape[0]
        if self.memory_position_counter == 0:
            self.max_memory_points = self.max_memory_batches * batch_size
            self.memory_buffer_x = x_samples.detach().cpu()
            self.memory_buffer_log_w = log_w.detach().cpu()
        elif self.n_memory_samples < self.max_memory_points: # continue to fill memory
            self.memory_buffer_x = torch.cat([self.memory_buffer_x, x_samples.detach().cpu()])
            self.memory_buffer_log_w = torch.cat([self.memory_buffer_log_w, log_w.detach().cpu()])
        else: # replace old memory
            self.memory_buffer_x[self.memory_position_counter*batch_size:
                                 (self.memory_position_counter + 1)*batch_size] = x_samples.detach().cpu()
            self.memory_buffer_log_w[self.memory_position_counter * batch_size:
                                 (self.memory_position_counter + 1) * batch_size] = log_w.detach().cpu()
        if self.memory_position_counter >= self.max_memory_batches-1:
            self.memory_position_counter = 0
        else:
            self.memory_position_counter += 1
        return

    def train_loop_with_memory(self, x_samples, log_w):
        total_loss = 0
        self.add_to_memory(x_samples, log_w)
        batch_size = x_samples.shape[0]
        for i in range(self.n_gradient_update_batches):
            batch_indices = np.random.choice(np.arange(self.n_memory_samples), batch_size,
                                             replace=False)
            x_samples = self.memory_buffer_x[batch_indices].to(self.device)
            log_w = self.memory_buffer_log_w[batch_indices].to(self.device)
            loss = self.train_inner_loop(x_samples, log_w)
            total_loss += torch.nan_to_num(loss)
        return total_loss


    def train_inner_loop(self, x_samples, log_w):
        self.optimizer.zero_grad()
        loss = self.loss(x_samples, log_w)
        if torch.isnan(loss) or torch.isinf(loss):
            if self.allow_ignore_nan_loss:
                print("Nan/Inf loss encountered in loss_1")
                return loss
            else:
                raise Exception("Nan/Inf loss_1 encountered")
        loss.backward()
        if self.clip_grad_norm is True:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.learnt_sampling_dist.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_value_(self.learnt_sampling_dist.parameters(), 1)
        self.optimizer.step()
        return loss


    def train(self, epochs=100, batch_size=1000, intermediate_plots=False,
              plotting_func=plot_samples, n_plots=3,
              KPI_batch_size=int(1e4), plotting_batch_size=int(1e3),
              jupyter=False, n_progress_updates=20, save=False, save_path=None):
        if save is True:
            assert save_path is not None
            from datetime import datetime
            current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
            save_path = save_path / f"training{current_time}"
            save_path.mkdir(parents=True, exist_ok=False)
            model_during_training_path = save_path / "model_checkpoints"
            model_during_training_path.mkdir(parents=True, exist_ok=False)
            samples_dict = {
                "epoch": [],
                "flow_samples": [],
                "AIS_samples": [],
                "log_w_AIS": []}
            performance_metrics_long = {}
        if jupyter:
            from tqdm import tqdm_notebook as tqdm
        else:
            from tqdm import tqdm
        epoch_per_save_and_print = max(int(epochs / n_progress_updates), 1)
        if intermediate_plots is True:
            epoch_per_plot = max(int(epochs / n_plots), 1)
        history = {"ESS_batch": [],
                    "loss": [],
                   "log_w": [],
                   "log_p_x_after_AIS": [],
                   }
        history.update(dict([(key, []) for key in self.AIS_train.transition_operator_class.interesting_info()]))
        if hasattr(self.target_dist, "sample"):
            history.update({'mean_log_prob_true_samples': [], 'min_log_prob_true_samples': []})
        elif hasattr(self.target_dist, "test_set"):
            history.update({'mean_log_q_x_test_samples': [], 'min_log_q_x_test_samples': []})
        pbar = tqdm(range(epochs))
        for self.current_epoch in pbar:
            if self.AIS_train.n_distributions > 2:
                x_samples, log_w = self.AIS_train.run(batch_size)
                x_samples, log_w = x_samples.detach(), log_w.detach() # be extra careful that these are detached
            else:
                x_samples, log_q = self.learnt_sampling_dist(batch_size)
                log_p = self.target_dist.log_prob(x_samples)
                log_w = log_p - log_q
            if self.use_memory_buffer:
                loss = self.train_loop_with_memory(x_samples, log_w)
            else:
                loss = self.train_inner_loop(x_samples, log_w)
            if self.current_epoch % epoch_per_save_and_print == 0 or self.current_epoch == epochs:
                # must do this outside of the torch.no_grad below
                summary_dict_AIS, long_dict_AIS = self.get_performance_metrics_AIS(KPI_batch_size,
                                                                                   batch_size)
                history = add_to_history_dict(history, summary_dict_AIS, additional_name="_AIS")
                if save:
                    performance_metrics_long = add_to_history_dict(performance_metrics_long, long_dict_AIS,
                                                                   additional_name="_AIS")
            with torch.no_grad(): # none of the below steps should require gradients
                # save info
                history["loss"].append(loss.item())
                history["log_w"].append(torch.mean(log_w).item())
                history["ESS_batch"].append(
                    self.AIS_train.effective_sample_size_unnormalised_log_weights(log_w, drop_nan=True).item() /
                    batch_size)
                transition_operator_info = self.AIS_train.transition_operator_class.interesting_info()
                for key in transition_operator_info:
                    history[key].append(transition_operator_info[key])
                if self.current_epoch % epoch_per_save_and_print == 0 or self.current_epoch == epochs:
                    summary_dict, long_dict = self.get_performance_metrics_flow(KPI_batch_size, batch_size)
                    history = add_to_history_dict(history, summary_dict,
                                                  additional_name="_flow")
                    if save:
                        performance_metrics_long = add_to_history_dict(performance_metrics_long, long_dict,
                                                                       additional_name="_flow")
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
                            log_probs_true = self.learnt_sampling_dist.log_prob(true_samples)
                            mean_log_q_x_true_samples = torch.mean(log_probs_true).item()
                            min_log_q_x_true_samples = torch.min(log_probs_true).item()
                            history['mean_log_prob_true_samples'].append(mean_log_q_x_true_samples)
                            history['min_log_prob_true_samples'].append(min_log_q_x_true_samples)
                            pbar.set_description(
                                f"loss: {np.round(np.mean(history['loss'][-epoch_per_save_and_print:]), 2)},"
                                f""f"mean_log_prob_true_samples {round(mean_log_q_x_true_samples, 2)},"
                                f"ESS {round(history['ESS_mean_AIS'][-1], 6)}")
                        elif hasattr(self.target_dist, "test_set"):
                            test_samples = self.target_dist.test_set(self.device)
                            log_probs_test = self.learnt_sampling_dist.log_prob(test_samples)
                            mean_log_q_x_test_samples = torch.mean(log_probs_test).item()
                            min_log_q_x_test_samples = torch.min(log_probs_test).item()
                            history['mean_log_q_x_test_samples'].append(mean_log_q_x_test_samples)
                            history['min_log_q_x_test_samples'].append(min_log_q_x_test_samples)
                            pbar.set_description(
                                f"loss: {np.round(np.mean(history['loss'][-epoch_per_save_and_print:]), 2)},"
                                f"mean_log_q_x_test_samples {round(mean_log_q_x_test_samples, 2)},"
                                f"min_log_q_x_test_samples {round(min_log_q_x_test_samples, 2)}"
                                f"ESS {round(history['ESS_mean_AIS'][-1], 6)}")
                        else:
                            pbar.set_description(
                                f"loss: {np.round(np.mean(history['loss'][-epoch_per_save_and_print:]), 2)},"
                                f"ESS {round(history['ESS_mean_AIS'][-1], 6)}")
                if intermediate_plots:
                    if self.current_epoch % epoch_per_plot == 0:
                        if save: # save model checkpoint, this makes it easy to replicate plots if we want to
                            self.learnt_sampling_dist.save_model(model_during_training_path, epoch=self.current_epoch)
                            self.AIS_train.transition_operator_class.save_model(model_during_training_path,
                                                                                epoch=self.current_epoch)
                        flow_samples = self.learnt_sampling_dist(plotting_batch_size)[0].cpu()
                        plotting_func(self, n_samples=plotting_batch_size,
                                      title=f"epoch {self.current_epoch}: samples from flow",
                                      samples_q=flow_samples)
                        if save:
                            samples_dict["epoch"].append(self.current_epoch)
                            samples_dict["flow_samples"].append(flow_samples.numpy())
                            plt.savefig(str(save_path /f"Samples_from_flow_epoch{self.current_epoch}.pdf"))
                        plt.show()
                        n_samples_AIS_plot = min(batch_size, plotting_batch_size) # so plots look consistent
                        # make sure plotting func has option to enter x_samples directly
                        plotting_func(self, n_samples=n_samples_AIS_plot ,
                                      title=f"epoch {self.current_epoch}: samples from AIS",
                                      samples_q=x_samples[:n_samples_AIS_plot].cpu())
                        if save:
                            samples_dict["AIS_samples"].append(x_samples[:n_samples_AIS_plot].cpu().numpy())
                            samples_dict["log_w_AIS"].append(log_w[:n_samples_AIS_plot].cpu().numpy())
                            plt.savefig(str(save_path /f"Samples_from_AIS_epoch{self.current_epoch}.pdf"))
                        plt.show()
        if save:
            import pickle
            with open(str(save_path / "history.pkl"), "wb") as f:
                pickle.dump(history, f)
            with open(str(save_path / "samples.pkl"), "wb") as f:
                pickle.dump(samples_dict, f)
            with open(str(save_path / "long_performance_metrics.pkl"), "wb") as f:
                pickle.dump(performance_metrics_long, f)
            self.learnt_sampling_dist.save_model(save_path)
            self.AIS_train.transition_operator_class.save_model(save_path)
        return history

    def get_performance_metrics_AIS(self, KPI_batch_size, batch_size, return_samples=False,
                                    n_batches_stat_aggregation=10):
        x, log_w = sample_and_log_w_big_batch_drop_nans(self.AIS_train, KPI_batch_size,
                                                        batch_size, AIS=True)
        ESS = []
        samples_per_batch = log_w.shape[0] // n_batches_stat_aggregation
        for i, batch_number in enumerate(range(n_batches_stat_aggregation)):
            if i != n_batches_stat_aggregation-1:
                log_w_batch = log_w[batch_number * samples_per_batch:(batch_number + 1) * samples_per_batch]
            else:
                log_w_batch = log_w[batch_number * samples_per_batch:]
            ESS_batch = self.AIS_train.effective_sample_size_unnormalised_log_weights(log_w_batch) / (
                        KPI_batch_size / n_batches_stat_aggregation)
            ESS.append(ESS_batch.item())
        summary_dict, long_dict = self.target_dist.performance_metrics(self, x, log_w)
        summary_dict["ESS_mean"] = np.mean(ESS)
        summary_dict["ESS_std"] = np.std(ESS)
        if not return_samples:
            return summary_dict, long_dict
        else:
            return summary_dict, long_dict, x

    def get_performance_metrics_flow(self, KPI_batch_size, batch_size, return_samples=False,
                                     n_batches_stat_aggregation=10):
        x, log_w = sample_and_log_w_big_batch_drop_nans(self.AIS_train, KPI_batch_size,
                                                        batch_size, AIS=False)
        ESS = []
        samples_per_batch = log_w.shape[0] // n_batches_stat_aggregation
        for i, batch_number in enumerate(range(n_batches_stat_aggregation)):
            if i != n_batches_stat_aggregation-1:
                log_w_batch = log_w[batch_number * samples_per_batch:(batch_number + 1) * samples_per_batch]
            else:
                log_w_batch = log_w[batch_number * samples_per_batch:]
            ESS_batch = self.AIS_train.effective_sample_size_unnormalised_log_weights(log_w_batch) / (
                        KPI_batch_size / n_batches_stat_aggregation)
            ESS.append(ESS_batch.item())
        summary_dict, long_dict = self.target_dist.performance_metrics(self, x, log_w)
        summary_dict["ESS_mean"] = np.mean(ESS)
        summary_dict["ESS_std"] = np.std(ESS)

        kl_MC = -torch.mean(log_w)
        alpha_2_MC = torch.logsumexp(2*log_w, dim=-1)
        summary_dict["kl"] = kl_MC.item()
        summary_dict["log_alpha_2_div"] = alpha_2_MC.item()
        if not return_samples:
            return summary_dict, long_dict
        else:
            return summary_dict, long_dict, x

    def alpha_div_annealed_samples_re_weight(self, x_samples, log_w):
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

        log_q_x = self.learnt_sampling_dist.log_prob(x_samples.detach())
        log_p_x = self.target_dist.log_prob(x_samples.detach())

        # also check that we have valid log probs
        valid_indices = ~torch.isinf(log_q_x) & ~torch.isnan(log_q_x)
        if valid_indices.all():
            log_w_normed = log_w.detach() - torch.logsumexp(log_w.detach(), dim=0)
            return - self.alpha_one_minus_alpha_sign * \
                   torch.logsumexp((self.alpha - 1) * (log_p_x - log_q_x) + log_w_normed, dim=0)
        else:
            log_w = log_w[valid_indices]
            log_q_x = log_q_x[valid_indices]
            log_p_x = log_p_x[valid_indices]
            log_w_normed = log_w.detach() - torch.logsumexp(log_w.detach(), dim=0)
            return - self.alpha_one_minus_alpha_sign * \
                   torch.logsumexp((self.alpha - 1) * (log_p_x - log_q_x) + log_w_normed, dim=0)


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
    HMC_transition_operator_args = {"step_tuning_method": "p_accept"}  # "Expected_target_prob", "No-U", "p_accept"
    print(HMC_transition_operator_args)
    target = MoG(dim=dim, n_mixes=5, min_cov=1, loc_scaling=10)
    true_expectation = MC_estimate_true_expectation(target, expectation_function, int(1e5))
    fig = plot_distribution(target, bounds=[[-30, 20], [-20, 20]])
    plt.show()
    learnt_sampler = FlowModel(x_dim=dim, scaling_factor=1.0, flow_type=flow_type, n_flow_steps=n_flow_steps)
    tester = AIS_trainer(target, learnt_sampler, n_distributions=4,
                         transition_operator="HMC", lr=1e-2,
                         tranistion_operator_kwargs=HMC_transition_operator_args,
                         use_memory_buffer=False, AIS_kwargs={"Beta_end": 1.0})
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


