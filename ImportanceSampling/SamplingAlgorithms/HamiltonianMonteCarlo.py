import torch
import torch.nn as nn
from ImportanceSampling.SamplingAlgorithms.base import BaseTransitionModel
from NormalisingFlow.utils import Monitor_NaN

class HMC(BaseTransitionModel):
    """
    Following: https: // arxiv.org / pdf / 1206.1901.pdf
    """
    def __init__(self, n_distributions, dim, epsilon=1.0, n_outer=1, L=5, step_tuning_method="p_accept",
                 target_p_accept=0.65, lr=1e-1, tune_period=False, common_epsilon_init_weight=0.1):
        self.class_args = locals().copy()
        del(self.class_args["self"])
        del(self.class_args["__class__"])
        super(HMC, self).__init__()
        assert step_tuning_method in ["p_accept", "Expected_target_prob", "No-U", "No-U-unscaled"]
        self.dim = dim
        self.tune_period = tune_period
        self.step_tuning_method = step_tuning_method
        if step_tuning_method in ["Expected_target_prob", "No-U", "No-U-unscaled"]:
            self.train_params = True
            self.counter = 0
            self.epsilons = nn.ParameterDict()
            self.epsilons["common"] = nn.Parameter(
                torch.log(torch.tensor([epsilon])) * common_epsilon_init_weight)
            # have to store epsilons like this otherwise we get weird erros
            for i in range(n_distributions-2):
                for n in range(n_outer):
                    self.epsilons[f"{i}_{n}"] = nn.Parameter(torch.log(
                        torch.ones(dim)*epsilon*(1 - common_epsilon_init_weight)))
            self.Monitor_NaN = Monitor_NaN()
            self.register_nan_hooks_model()
            if self.step_tuning_method == "No-U":
                self.register_buffer("characteristic_length", torch.ones(n_distributions, dim))  # initialise
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        else:
            self.train_params = False
            self.register_buffer("common_epsilon", torch.tensor([epsilon * common_epsilon_init_weight]))
            self.register_buffer("epsilons", torch.ones([n_distributions-2, n_outer])*epsilon *
                                 (1 - common_epsilon_init_weight))
        self.n_outer = n_outer
        self.L = L
        self.n_distributions = n_distributions
        self.target_p_accept = target_p_accept
        self.first_dist_p_accepts = [torch.tensor([0.0]) for _ in range(n_outer)]
        self.last_dist_p_accepts = [torch.tensor([0.0]) for _ in range(n_outer)]
        self.average_distance = 0

    def save_model(self, save_path, epoch=None):
        model_description = str(self.class_args)
        if epoch is None:
            summary_results_path = str(save_path / "HMC_model_info.txt")
            model_path = str(save_path / "HMC_model")
        else:
            summary_results_path = str(save_path / f"HMC_model_info_epoch{epoch}.txt")
            model_path = str(save_path / f"HMC_model_epoch{epoch}")
        with open(summary_results_path, "w") as g:
            g.write(model_description)
        torch.save(self.state_dict(), model_path)

    def load_model(self, save_path, epoch=None, device='cpu'):
        if epoch is None:
            model_path = str(save_path / "HMC_model")
        else:
            model_path = str(save_path / f"HMC_model_epoch{epoch}")
        self.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        print("loaded HMC model")

    def register_nan_hooks(self, parameter):
        parameter.register_hook(
            lambda grad: self.Monitor_NaN.overwrite_NaN_grad(grad, print_=False, replace_with=0.0))


    def register_nan_hooks_model(self):
        for parameter in self.parameters():
            # replace with positive number so it decreases step size when we get nans
            parameter.register_hook(
                lambda grad: self.Monitor_NaN.overwrite_NaN_grad(grad, print_=False, replace_with=0.0))


    def interesting_info(self):
        interesting_dict = {}
        if self.n_distributions > 2:
            for i, val in enumerate(self.first_dist_p_accepts):
                interesting_dict[f"dist1_p_accept_{i}"] = val.item()
            if self.n_distributions > 3:
                for i, val in enumerate(self.last_dist_p_accepts):
                    interesting_dict[f"dist{self.n_distributions-3}_p_accept_{i}"] = val.item()
            epsilon_first_dist_first_loop = self.get_epsilon(0,0)
            if epsilon_first_dist_first_loop.numel() == 1:
                interesting_dict[f"epsilons_dist0_loop0"] = epsilon_first_dist_first_loop.cpu().item()
            else:
                interesting_dict[f"epsilons_dist0_0_dim0"] = epsilon_first_dist_first_loop[0].cpu().item()
                interesting_dict[f"epsilons_dist0_0_dim-1"] = epsilon_first_dist_first_loop[-1].cpu().item()
            if self.n_distributions > 3:
                last_dist_n = self.n_distributions - 2 - 1
                epsilon_last_dist_first_loop = self.get_epsilon(last_dist_n, 0)
                if epsilon_last_dist_first_loop.numel() == 1:
                    interesting_dict[f"epsilons_dist{last_dist_n}_loop0"] = epsilon_last_dist_first_loop.cpu().item()
                else:
                    interesting_dict[f"epsilons_dist{last_dist_n}_0_dim0"] = epsilon_last_dist_first_loop[0].cpu().item()
                    interesting_dict[f"epsilons_dist{last_dist_n}_0_dim-1"] = epsilon_last_dist_first_loop[-1].cpu().item()

            interesting_dict["average_distance"] = self.average_distance
        return interesting_dict

    def get_epsilon(self, i, n):
        if self.train_params:
            return torch.exp(self.epsilons[f"{i}_{n}"]) + torch.exp(self.epsilons["common"])
        else:
            return self.epsilons[i, n] + self.common_epsilon


    def HMC_func(self, U, current_q, grad_U, i):
        if self.step_tuning_method == "Expected_target_prob":
            # need this for grad function
            current_q = current_q.clone().detach().requires_grad_(True)
            current_q = torch.clone(current_q)  # so we can do in place operations, kinda weird hac
        else:
            current_q = current_q.detach()  # otherwise just need to block grad flow
        loss = 0
        # need this for grad function
        # base function for HMC written in terms of potential energy function U
        for n in range(self.n_outer):
            original_q = torch.clone(current_q).detach()
            if self.train_params:
                self.optimizer.zero_grad()
            epsilon = self.get_epsilon(i, n).detach()
            q = current_q
            p = torch.randn_like(q)
            current_p = p
            # make momentum half step
            p = p - epsilon * grad_U(q) / 2

            # Now loop through position and momentum leapfrogs
            for l in range(self.L):
                epsilon = self.get_epsilon(i, n)
                if (l != self.L - 1 and self.step_tuning_method in ["No-U", "No-U-unscaled"]) or not self.train_params:
                    epsilon = epsilon.detach()
                # Make full step for position
                q = q + epsilon * p
                # Make a full step for the momentum if not at end of trajectory
                if l != self.L-1:
                    p = p - epsilon * grad_U(q)

            if self.train_params:
                self.register_nan_hooks(q)  # to prevent Nan gradients in the loss
            # make a half step for momentum at the end
            p = p - epsilon * grad_U(q) / 2
            if self.train_params:
                self.register_nan_hooks(p)
            # Negate momentum at end of trajectory to make proposal symmetric
            p = -p

            U_current = U(current_q)
            U_proposed = torch.nan_to_num(U(q))
            current_K = torch.sum(current_p**2, dim=-1) / 2
            proposed_K = torch.sum(p**2, dim=-1) / 2

            # Accept or reject the state at the end of the trajectory, returning either the position at the
            # end of the trajectory or the initial position
            acceptance_probability = torch.exp(U_current - U_proposed + current_K - proposed_K)
            # reject samples with nan acceptance probability
            acceptance_probability = torch.nan_to_num(acceptance_probability, nan=0.0, posinf=0.0, neginf=0.0)
            acceptance_probability = torch.clamp(acceptance_probability, min=0.0, max=1.0)
            accept = acceptance_probability > torch.rand(acceptance_probability.shape).to(q.device)
            current_q[accept] = q[accept]
            p_accept = torch.mean(acceptance_probability)
            if self.step_tuning_method == "p_accept":
                if p_accept > self.target_p_accept: # too much accept
                    self.epsilons[i, n] = self.epsilons[i, n] * 1.1
                    self.common_epsilon = self.common_epsilon * 1.05
                else:
                    self.epsilons[i, n] = self.epsilons[i, n] / 1.1
                    self.common_epsilon = self.common_epsilon / 1.05
            else: # self.step_tuning_method == "No-U":
                if p_accept < 0.01: # or (self.counter < 100 and p_accept < 0.4):
                    # if p_accept is very low manually decrease step size, as this means that no acceptances so no
                    # gradient flow to use
                    self.epsilons[f"{i}_{n}"].data = self.epsilons[f"{i}_{n}"].data - 0.05
                    self.epsilons["common"].data = self.epsilons["common"].data - 0.05
                if i == 0:
                    self.counter += 1
            if i == 0: # save fist and last distribution info
                # save as interesting info for plotting
                self.first_dist_p_accepts[n] = torch.mean(acceptance_probability).cpu().detach()
            elif i == self.n_distributions - 3:
                self.last_dist_p_accepts[n] = torch.mean(acceptance_probability).cpu().detach()

            if i == 0 or self.step_tuning_method == "No-U-unscaled":
                distance = torch.linalg.norm((original_q - current_q), ord=2, dim=-1)
                if i == 0:
                    self.average_distance = torch.mean(distance).detach().cpu()
                if self.step_tuning_method == "No-U-unscaled":
                    # torch.autograd.grad(torch.mean(weighted_mean_square_distance), self.epsilons[f"{i}_{n}"], retain_graph=True)
                    # torch.autograd.grad(loss, self.epsilons[f"{i}_{n}"], retain_graph=True)
                    weighted_mean_square_distance = acceptance_probability * distance ** 2
                    loss = loss - torch.mean(weighted_mean_square_distance)
            if self.step_tuning_method == "No-U":
                distance_scaled = torch.linalg.norm((original_q - current_q) / self.characteristic_length[i, :], ord=2, dim=-1)
                weighted_scaled_mean_square_distance = acceptance_probability * distance_scaled ** 2
                if (self.tune_period is False or self.counter < self.tune_period) and self.step_tuning_method == "No-U":
                    # remove zeros so we don't get infs when we divide
                    weighted_scaled_mean_square_distance[weighted_scaled_mean_square_distance == 0.0] = 1.0
                    loss = loss + torch.mean(1.0/weighted_scaled_mean_square_distance -
                                             weighted_scaled_mean_square_distance)

        if self.train_params:
            if self.tune_period is False or self.counter < self.tune_period:
                if self.step_tuning_method == "Expected_target_prob":
                    loss = torch.mean(U(current_q))
                if not (torch.isinf(loss) or torch.isnan(loss)):
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                    torch.nn.utils.clip_grad_value_(self.parameters(), 1)
                    # torch.autograd.grad(loss, self.epsilons["0_1"], retain_graph=True)
                    self.optimizer.step()
        if self.step_tuning_method == "No-U":
            # set next characteristc lengths
            self.characteristic_length.data[i, :] = torch.std(current_q.detach(), dim=0)
        return current_q.detach()  # stop gradient flow

    def run(self, current_q, log_q_x, i):
        # currently mainly written with grad_log_q_x = None in mind
        # using diagonal, would be better to use vmap (need to wait until this is released doh)
        """
        U is potential energy, q is position, p is momentum
        """
        def U(x: torch.Tensor):
            return - log_q_x(x)

        def grad_U(q: torch.Tensor):
            q = q.clone().requires_grad_(True) #  need this to get gradients
            y = U(q)
            return torch.nan_to_num(
                torch.clamp(
                torch.autograd.grad(y, q, grad_outputs=torch.ones_like(y))[0],
                max=1e4, min=-1e4),
                nan=0.0, posinf=0.0, neginf=0.0)

        current_q = self.HMC_func(U, current_q, grad_U, i)
        return current_q

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    from TargetDistributions.MoG import MoG
    from FittedModels.Models.DiagonalGaussian import DiagonalGaussian
    from FittedModels.utils.plotting_utils import plot_history
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    n_samples = 1000
    n_distributions_pretend = 3
    dim = 2
    train_params = True
    torch.manual_seed(2)
    target = MoG(dim=dim, n_mixes=5, loc_scaling=5)
    learnt_sampler = DiagonalGaussian(dim=dim, log_std_initial_scaling=1.0)
    hmc = HMC(n_distributions=n_distributions_pretend, n_outer=1, epsilon=1.0, L=5, dim=dim,
              step_tuning_method="p_accept")
    # "Expected_target_prob", "No-U", "p_accept"
    n = 100
    history = {}
    history.update(dict([(key, []) for key in hmc.interesting_info()]))
    for i in tqdm(range(n)):
        for j in range(n_distributions_pretend-2):
            sampler_samples = learnt_sampler(n_samples)[0]
            x_HMC = hmc.run(sampler_samples, target.log_prob, j)
        transition_operator_info = hmc.interesting_info()
        for key in transition_operator_info:
            history[key].append(transition_operator_info[key])
        if i == 0 or i == n - 1 or i == int(n/2):
            x_HMC = x_HMC.cpu().detach()
            plt.plot(x_HMC[:, 0], x_HMC[:, 1], "o", alpha=0.5)
            plt.title("HMC samples")
            plt.show()
    plot_history(history)
    plt.show()

    true_samples = target.sample((n_samples,)).cpu().detach()
    plt.plot(true_samples[:, 0], true_samples[:, 1], "o", alpha=0.5)
    plt.title("true samples")
    plt.show()

    sampler_samples = learnt_sampler(n_samples)[0].cpu().detach()
    plt.plot(sampler_samples[:, 0], sampler_samples[:, 1], "o", alpha=0.5)
    plt.title("sampler samples")
    plt.show()


