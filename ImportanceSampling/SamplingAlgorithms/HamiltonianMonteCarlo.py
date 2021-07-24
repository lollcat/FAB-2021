import torch
import torch.nn as nn
from ImportanceSampling.SamplingAlgorithms.base import BaseTransitionModel
from NormalisingFlow.utils import Monitor_NaN

class HMC(BaseTransitionModel):
    """
    Following: https: // arxiv.org / pdf / 1206.1901.pdf
    """
    def __init__(self, n_distributions, dim, epsilon=1.0, n_outer=1, L=5, step_tuning_method="No-U",
                 target_p_accept=0.65, lr=1e-3, tune_period=False):
        self.class_args = locals().copy()
        del(self.class_args["self"])
        del(self.class_args["__class__"])
        super(HMC, self).__init__()
        assert step_tuning_method in ["p_accept", "Expected_target_prob", "No-U"]
        self.dim = dim
        self.tune_period = tune_period
        self.step_tuning_method = step_tuning_method
        if step_tuning_method in ["Expected_target_prob", "No-U"]:
            self.train_params = True
            self.counter = 0
            self.epsilons = nn.ParameterDict()
            # have to store epsilons like this otherwise we get weird erros
            self.epsilons["common"] = nn.Parameter(torch.tensor([epsilon*0.5]))
            for i in range(n_distributions-2):
                for n in range(n_outer):
                    self.epsilons[f"{i}_{n}"] = nn.Parameter(torch.ones(dim)*epsilon*0.5)
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
            self.Monitor_NaN = Monitor_NaN()
            self.register_nan_hooks()
        else:
            self.train_params = False
            self.register_buffer("common_epsilon", torch.tensor([epsilon*0.5]))
            self.register_buffer("epsilons", torch.ones([n_distributions-2, n_outer])*epsilon*0.5)
        self.n_outer = n_outer
        self.L = L
        self.n_distributions = n_distributions
        self.target_p_accept = target_p_accept
        self.first_dist_p_accepts = [torch.tensor([0.0]) for _ in range(n_outer)]
        self.last_dist_p_accepts = [torch.tensor([0.0]) for _ in range(n_outer)]
        self.weighted_scaled_mean_square_distance = 0
        self.average_distance = 0

    def save_model(self, save_path):
        model_description = str(self.class_args)
        summary_results_path = str(save_path / "HMC_model_info.txt")
        model_path = str(save_path / "HMC_model")
        with open(summary_results_path, "w") as g:
            g.write(model_description)
        torch.save(self.state_dict(), model_path)

    def load_model(self, save_path):
        model_path = str(save_path / "HMC_model")
        self.load_state_dict(torch.load(model_path))
        print("loaded HMC model")

    def register_nan_hooks(self):
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
            if self.train_params:
                interesting_dict["epsilon_shared"] = self.epsilons["common"].item()
                interesting_dict[f"epsilons_0_0_0"] = self.epsilons[f"{0}_{0}"][0].cpu().item()
                interesting_dict[f"epsilons_0_0_-1"] = self.epsilons[f"{0}_{0}"][-1].cpu().item()
                if self.n_outer != 1:
                    interesting_dict[f"epsilons_0_-1_0"] = self.epsilons[f"{0}_{self.n_outer-1}"][0].cpu().item()
                    interesting_dict[f"epsilons_0_-1_-1"] = self.epsilons[f"{0}_{self.n_outer-1}"][-1].cpu().item()
            else:
                interesting_dict["epsilon_shared"] = self.common_epsilon.item()
                interesting_dict[f"epsilons_0_0"] = self.epsilons[0, 0].cpu().item()
                interesting_dict[f"epsilons_0_-1"] = self.epsilons[0, -1].cpu().item()
            interesting_dict["average_distance"] = self.average_distance
            interesting_dict[f"p_accept_weighted_mean_square_distance"] = self.weighted_scaled_mean_square_distance
        return interesting_dict

    def get_epsilon(self, i, n):
        if self.train_params:
            return torch.abs(self.epsilons[f"{i}_{n}"] + self.epsilons["common"])
        else:
            return torch.abs(self.epsilons[i, n] + self.common_epsilon)

    def HMC_func(self, U, current_q, grad_U, i):
        if self.step_tuning_method == "Expected_target_prob":
            # need this for grad function
            current_q = current_q.clone().detach().requires_grad_(True)
            current_q = torch.clone(current_q)  # so we can do in place operations, kinda weird hac
        else:
            current_q = current_q.detach()  # otherwise just need to block grad flow
        characteristic_length = torch.std(current_q.detach(), dim=0)
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
                if (l != self.L - 1 and self.step_tuning_method == "No-U") or not self.train_params:
                    epsilon = epsilon.detach()
                # Make full step for position
                q = q + epsilon * p
                # Make a full step for the momentum if not at end of trajectory
                if l != self.L-1:
                    p = p - epsilon * grad_U(q)

            # make a half step for momentum at the end
            p = p - epsilon * grad_U(q) / 2
            # Negate momentum at end of trajectory to make proposal symmetric
            p = -p

            U_current = U(current_q)
            U_proposed = U(q)
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
                if p_accept < 0.05 or (self.counter < 100 and p_accept < 0.4):
                    # if p_accept is very low manually decrease step size, as this means that no acceptances so no
                    # gradient flow to use
                    self.epsilons[f"{i}_{n}"].data = self.epsilons[f"{i}_{n}"].data / 1.5
                    self.epsilons["common"].data = self.epsilons["common"].data / 1.2
                if i == 0:
                    self.counter += 1
            if i == 0: # save fist and last distribution info
                # save as interesting info for plotting
                self.first_dist_p_accepts[n] = torch.mean(acceptance_probability).cpu().detach()
            elif i == self.n_distributions - 3:
                self.last_dist_p_accepts[n] = torch.mean(acceptance_probability).cpu().detach()

            if i == 0 or self.step_tuning_method == "No-U":
                distance_scaled = torch.linalg.norm((original_q - current_q) / characteristic_length, ord=2, dim=-1)
                weighted_scaled_mean_square_distance = acceptance_probability * distance_scaled ** 2
                if i == 0:
                    self.weighted_scaled_mean_square_distance = \
                        torch.mean(weighted_scaled_mean_square_distance).detach().cpu()
                    self.average_distance = \
                        torch.mean(torch.linalg.norm((original_q - current_q), ord=2, dim=-1).detach().cpu())

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
            q = q.clone().detach().requires_grad_(True) #  need this to get gradients
            y = U(q)
            return torch.clamp(torch.autograd.grad(y, q, grad_outputs=torch.ones_like(y))[0], max=1e6, min=-1e6)

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


