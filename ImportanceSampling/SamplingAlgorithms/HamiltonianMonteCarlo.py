import torch
import torch.nn as nn
from ImportanceSampling.SamplingAlgorithms.base import BaseTransitionModel

class HMC(BaseTransitionModel):
    """
    Following: https: // arxiv.org / pdf / 1206.1901.pdf
    """
    def __init__(self, n_distributions, epsilon, dim, n_outer=2, L=5, train_params=True,
                 auto_adjust_step_size=False,
                 target_p_accept=0.2, lr=1e-3):
        super(HMC, self).__init__()
        self.train_params = train_params
        if train_params:
            assert auto_adjust_step_size == False

            self.epsilons = nn.ParameterDict()
            # have to store epsilons like this otherwise we get weird erros
            for i in range(n_distributions):
                for n in range(n_outer):
                    self.epsilons[f"{i}_{n}"] = nn.Parameter(torch.ones(dim)*epsilon)
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        else:
            self.register_buffer("epsilons", torch.ones([n_distributions, n_outer])*epsilon)
        self.n_outer = n_outer
        self.L = L
        self.auto_adjust_step_size = auto_adjust_step_size
        self.target_p_accept = target_p_accept
        self.first_p_accept = torch.tensor([0.0])
        self.last_p_accept = torch.tensor([0.0])

    def interesting_info(self):
        interesting_dict = {}
        interesting_dict["first_p_accept"] = self.first_p_accept.item()
        interesting_dict["last_p_accept"] = self.last_p_accept.item()
        if self.train_params:
            interesting_dict[f"epsilons_0_0_0"] = self.get_epsilon(0, 0)[0].cpu().item()
            interesting_dict[f"epsilons_0_-1_0"] = self.get_epsilon(0, self.n_outer-1)[0].cpu().item()
        else:
            interesting_dict[f"epsilons_0_0"] = self.epsilons[0, 0].cpu().item()
            interesting_dict[f"epsilons_0_-1"] = self.epsilons[0, -1].cpu().item()
        return interesting_dict

    def get_epsilon(self, i, n):
        if self.train_params:
            return self.epsilons[f"{i}_{n}"]
        else:
            return self.epsilons[i, n]


    def HMC_func(self, U, current_q, grad_U, i):
        # base function for HMC written in terms of potential energy function U
        if self.train_params:
            self.optimizer.zero_grad()
            loss = 0
        for n in range(self.n_outer):
            epsilon = self.get_epsilon(i, n)
            q = current_q
            p = torch.randn_like(q)
            current_p = p
            # make momentum half step
            p = p - epsilon * grad_U(q) / 2

            # Now loop through position and momentum leapfrogs
            for l in range(self.L):
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

            # comment out this, as let's rather just maximise the final log prob
            #if self.train_params:
            #    loss = loss + torch.mean(U_proposed)  # - mean log target prob

            # Accept or reject the state at the end of the trajectory, returning either the position at the
            # end of the trajectory or the initial position
            acceptance_probability = torch.exp(U_current - U_proposed + current_K - proposed_K)
            accept = (acceptance_probability > torch.rand(acceptance_probability.shape).to(q.device)).int()
            accept = accept[:, None].repeat(1, q.shape[-1])
            current_q = accept * q + (1 - accept) * current_q

            if self.auto_adjust_step_size:
                p_accept = torch.mean(torch.clamp_max(acceptance_probability, 1))
                if p_accept > self.target_p_accept: # to much accept
                    self.epsilons[i, n] = self.epsilons[i, n] * 1.1
                else:
                    self.epsilons[i, n] = self.epsilons[i, n] * 0.9

            if n == 0:
                # save as interesting info for plotting
                self.first_p_accept = torch.mean(torch.clamp_max(acceptance_probability, 1))
        # save as interesting info for plotting
        self.last_p_accept = torch.mean(torch.clamp_max(acceptance_probability, 1))

        loss = torch.mean(U_proposed)
        if self.train_params:
            loss.backward(retain_graph=True)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
            torch.nn.utils.clip_grad_value_(self.parameters(), 1)
            # torch.autograd.grad(loss, self.epsilons["0_1"], retain_graph=True)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                self.optimizer.step()
        return current_q

    def run(self, current_q, log_q_x, i):
        # currently mainly written with grad_log_q_x = None in mind
        # using diagonal, would be better to use vmap (need to wait until this is released doh)
        """
        U is potential energy, q is position, p is momentum
        """
        def U(x: torch.Tensor):
            return - log_q_x(x)

        def grad_U(q: torch.Tensor):
            y = U(q)
            return torch.autograd.grad(y, q, grad_outputs=torch.ones_like(y))[0]

        current_q = self.HMC_func(U, current_q, grad_U, i)
        return current_q

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    from TargetDistributions.MoG import MoG
    from FittedModels.Models.DiagonalGaussian import DiagonalGaussian
    import matplotlib.pyplot as plt
    n_samples = 4000
    dim = 2
    torch.manual_seed(2)
    target = MoG(dim=dim, n_mixes=5, loc_scaling=5)
    learnt_sampler = DiagonalGaussian(dim=dim, log_std_initial_scaling=2.0)
    sampler_samples = learnt_sampler(n_samples)[0]
    hmc = HMC(n_distributions=2, n_outer=40, epsilon=1.0, L=6, dim=dim)
    for i in range(10):
        x_HMC = hmc.run(sampler_samples, target.log_prob, 0)

    sampler_samples = sampler_samples.cpu().detach()
    x_HMC = x_HMC.cpu().detach()
    plt.plot(sampler_samples[:, 0], sampler_samples[:, 1], "o", alpha=0.5)
    plt.title("sampler samples")
    plt.show()
    plt.plot(x_HMC[:, 0], x_HMC[:, 1], "o", alpha=0.5)
    plt.title("annealed samples")
    plt.show()
    true_samples = target.sample((n_samples,)).cpu().detach()
    plt.plot(true_samples[:, 0], true_samples[:, 1], "o", alpha=0.5)
    plt.title("true samples")
    plt.show()
    print(hmc.interesting_info())
