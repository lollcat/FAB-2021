import torch
import numpy as np
torch.set_default_dtype(torch.float64)

class NUTS:
    """
    https://arxiv.org/pdf/1111.4246.pdf
    """
    def __init__(self, dim, log_q_x, sigma=0.6):
        self.dim = dim
        self.log_q_x = log_q_x
        self.sigma = sigma

    def L(self, theta: torch.Tensor):
        return self.log_q_x(theta)

    def grad_L(self, theta: torch.Tensor):
        y = self.L(theta)
        return torch.autograd.grad(y, theta, grad_outputs=torch.ones_like(y))[0]

    def L_and_grad_L(self, theta):
        y = self.L(theta)
        return y, torch.autograd.grad(y, theta, grad_outputs=torch.ones_like(y))[0]

    def joint_unnormalised_log_prob(self, theta, r):
        return self.L(theta) - 0.5 * torch.einsum("ij,ij->i", r,r)

    def joint_unnofrmalised_log_prob_direct(self, L, r):
        # if we already have L
        return L - 0.5 * torch.einsum("ij,ij->i", r,r)

    def Leapfrog(self, theta, r, epsilon, return_L_and_grad=False):
        """
        Leapfrog integrator
        :param theta: parameters, particles position in Hamiltonian system
        :param r: Momentum
        :param epsilon:
        :return: theta, r
        """
        r = r + epsilon/2 * self.grad_L(theta)
        theta = theta + epsilon*r
        L, grad_L = self.L_and_grad_L(theta)
        r = r + epsilon/2 * grad_L
        if return_L_and_grad: # useful if we want to run checks on L and grad_L
            return theta, r, L, grad_L
        else:
            return theta, r


    def FindReasonableEpsilon(self, theta):
        eps = torch.ones(theta.shape[0], 1).to(theta.device)
        r = torch.randn_like(theta)
        _, r_new, L_new, grad_L_new = self.Leapfrog(theta, r, eps, return_L_and_grad=True)
        nan_L_or_grad =  torch.isinf(L_new) | torch.sum(torch.isinf(grad_L_new), dim=-1).bool()
        while nan_L_or_grad.any():
            eps = eps * 0.5
            theta_new, r_new, L_new, _ = self.Leapfrog(theta, r, eps, return_L_and_grad=True)
            nan_L_or_grad = torch.isinf(L_new) | torch.isinf(grad_L_new)
        L = self.L(theta)
        a = 2 * \
            (L_new - 0.5*torch.sum(r_new**2, dim=-1) - (L - 0.5*torch.sum(r**2, dim=-1)) >
             np.log(0.5)) - 1.0
        indices = ((L_new - 0.5*torch.sum(r_new**2, dim=-1) - (L - 0.5*torch.sum(r**2, dim=-1)))*a) > -a*np.log(2.0)
        while indices.any():
            eps[indices] = (2.0**a[indices])[:, None]*eps[indices]
            _, r_new[indices], L_new[indices], _ = \
                self.Leapfrog(theta[indices], r[indices], eps[indices], return_L_and_grad=True)
            indices[indices.clone()] = \
                (L_new[indices] - 0.5*torch.sum(r_new[indices]**2, dim=-1) - \
                (L[indices] - 0.5*torch.sum(r[indices]**2, dim=-1)))*a[indices] > -a[indices]*np.log(2.0)
        return eps.detach()

    def BuildTree(self, theta, r, log_u, v, j, eps, theta_0, r_0, delta_max = 1000.0):
        """
        :param theta: batch_size, x_dim
        :param r: batch_size, x_dim
        :param u: batch_size
        :param v: -1 or 1
        :param j: int
        :param eps: batch_size
        :param theta_0: batch_size, x_dim
        :param r_0: batch_size, x_dim
        """
        if j == 0:
            theta_dash, r_dash, L_dash, _= self.Leapfrog(theta, r, v*eps, return_L_and_grad=True)
            joint_log_p_dash = self.joint_unnofrmalised_log_prob_direct(L_dash, r_dash)
            n_dash = (log_u <= joint_log_p_dash[:, None]).double()
            s_dash = (log_u < joint_log_p_dash[:, None] + delta_max).double()
            theta_minus = theta_dash.clone()
            theta_plus = theta_dash.clone()
            r_minus = r_dash.clone()
            r_plus = r_dash.clone()
            a_dash = torch.clamp_max(torch.exp(joint_log_p_dash - self.joint_unnormalised_log_prob(theta_0, r_0))[:, None],
                                   1.0)
            n_a_dash = torch.ones_like(n_dash)
            return theta_minus, r_minus, theta_plus, r_plus, theta_dash, n_dash, s_dash, a_dash, n_a_dash
        else:
            theta_minus, r_minus, theta_plus, r_plus, theta_dash, n_dash, s_dash, a_dash, n_a_dash = \
                self.BuildTree(theta, r, log_u, v, j-1, eps, theta_0, r_0)
            s_equals_1_indices = (s_dash == 1).squeeze(dim=-1)
            if s_equals_1_indices.any():
                if v == -1:
                    theta_minus[s_equals_1_indices], r_minus[s_equals_1_indices], _, _, \
                    theta_dash_dash, n_dash_dash, s_dash_dash, a_dash_dash, n_a_dash_dash = \
                        self.BuildTree(theta_minus[s_equals_1_indices], r_minus[s_equals_1_indices],
                                       log_u[s_equals_1_indices], v, j - 1, eps[s_equals_1_indices],
                                       theta_0[s_equals_1_indices], r_0[s_equals_1_indices])
                else:
                    _, _, theta_plus[s_equals_1_indices], r_plus[s_equals_1_indices], \
                    theta_dash_dash, n_dash_dash, s_dash_dash, a_dash_dash, n_a_dash_dash = \
                        self.BuildTree(theta_plus[s_equals_1_indices], r_plus[s_equals_1_indices],
                                       log_u[s_equals_1_indices], v, j - 1, eps[s_equals_1_indices],
                                       theta_0[s_equals_1_indices], r_0[s_equals_1_indices])
                # note that dash_dash objects only have values for which s == 1
                prob_accept_theta_dash_dash = n_dash_dash/(n_dash[s_equals_1_indices] + n_dash_dash)
                theta_update_indices = (prob_accept_theta_dash_dash > torch.rand_like(prob_accept_theta_dash_dash))\
                    .squeeze()
                theta_dash[s_equals_1_indices][theta_update_indices] = theta_dash_dash[theta_update_indices]
                a_dash[s_equals_1_indices] = a_dash[s_equals_1_indices] + a_dash_dash
                n_a_dash[s_equals_1_indices] = n_a_dash[s_equals_1_indices] + n_a_dash_dash
                s_dash[s_equals_1_indices] = \
                    s_dash_dash* \
                    (torch.einsum("ij,ij->i",
                                 theta_plus[s_equals_1_indices] - theta_minus[s_equals_1_indices],
                                 r_minus[s_equals_1_indices]) >= 0)[:, None]  * \
                    (torch.einsum("ij,ij->i",
                                  theta_plus[s_equals_1_indices] - theta_minus[s_equals_1_indices],
                                  r_plus[s_equals_1_indices]) >= 0)[:, None]
                n_dash[s_equals_1_indices] = n_dash[s_equals_1_indices] + n_dash_dash
            return theta_minus, r_minus, theta_plus, r_plus, theta_dash, n_dash, s_dash, a_dash, n_a_dash


    def run(self, theta_0, M, M_adapt, delta=0.65):
        """
        theta: params
        M: number of iter
        M_adapt: number of iter in which we allow adaption of epsilon
        delta: target number of acceptances
        """
        theta_m_minus_1 = theta_0.clone()
        epsilon = self.FindReasonableEpsilon(theta_0)
        mu = torch.log10(epsilon)
        epsilon_bar = torch.ones_like(epsilon)
        H_bar = torch.zeros_like(epsilon)
        gamma = 0.05
        t_0 = 10
        k = 0.75

        theta_m = theta_0.clone()
        for m in range(1, M+1):
            r_0 = torch.randn_like(theta_m_minus_1)
            log_u = torch.log(torch.rand_like(epsilon)) + self.joint_unnormalised_log_prob(theta_m_minus_1, r_0)[:, None]
            s = torch.ones_like(epsilon)
            theta_minus = theta_m_minus_1.clone()
            theta_plus = theta_m_minus_1.clone()
            r_minus = r_0.clone()
            r_plus = r_0.clone()
            j = 0
            theta_m = theta_m_minus_1.clone()
            n = torch.ones_like(epsilon)

            s_equals_1 = torch.squeeze(s == 1, dim=-1)

            theta_dash = torch.tensor(float("nan")).expand(theta_m_minus_1.shape)
            n_dash = torch.tensor(float("nan")).expand(n.shape)
            s_dash = torch.tensor(float("nan")).expand(s.shape)
            a = torch.tensor(float("nan")).expand(epsilon.shape)
            n_a = torch.tensor(float("nan")).expand(epsilon.shape)
            while s_equals_1.any():
                # note inside this loop we only only updates indices for which s == 1

                # currently sharing v across samples to make parellel easier
                # I think this should be okay, as sign of r is different across batch
                v = torch.tensor([1, -1])[torch.randint(low=0, high=2, size=(1,))]
                if v == -1:
                    theta_minus[s_equals_1], r_minus[s_equals_1], _, _, \
                    theta_dash[s_equals_1], n_dash[s_equals_1], s_dash[s_equals_1], a[s_equals_1], n_a[s_equals_1] \
                        = \
                        self.BuildTree(theta_minus[s_equals_1], r_minus[s_equals_1],
                                       log_u[s_equals_1], v, j, epsilon[s_equals_1],
                                       theta_m_minus_1[s_equals_1], r_0[s_equals_1])
                else:
                    _, _, theta_plus[s_equals_1], r_plus[s_equals_1], theta_dash[s_equals_1], \
                    n_dash[s_equals_1], s_dash[s_equals_1], a[s_equals_1], n_a[s_equals_1] = \
                        self.BuildTree(theta_plus[s_equals_1], r_plus[s_equals_1],
                                       log_u[s_equals_1], v, j, epsilon[s_equals_1],
                                       theta_m_minus_1[s_equals_1], r_0[s_equals_1])
                s_dash_equals_1 = (s_dash == 1).squeeze(dim=-1)
                s_dash_equals_1[~s_equals_1] = False  # these indices aren't partaking
                if s_dash_equals_1.any():
                    update_theta = torch.squeeze((n_dash/n > torch.rand_like(n_dash))[s_dash_equals_1])
                    theta_m[s_dash_equals_1][update_theta] = theta_dash[s_dash_equals_1][update_theta]
                n[s_equals_1] = n[s_equals_1] + n_dash[s_equals_1]
                s[s_equals_1] = s_dash[s_equals_1]*(
                        torch.einsum("ik,ik->i",
                            (theta_plus[s_equals_1] - theta_minus[s_equals_1]), r_minus[s_equals_1])
                        >= 0 )[:, None] \
                        * (torch.einsum("ik,ik->i",
                                       (theta_plus[s_equals_1] - theta_minus[s_equals_1]), r_plus[s_equals_1])
                           >= 0)[:, None]
                j = j + 1
                s_equals_1 = torch.squeeze(s == 1, dim=-1)
            if m <= M_adapt:
                H_bar = (1 - 1/(m + t_0))*H_bar + 1/(m + t_0)*(delta - a / n_a)
                log_epsilon = mu - np.sqrt(m)/gamma * H_bar
                epsilon = torch.exp(log_epsilon)
                if 'log_epsilon_bar' not in locals():
                    log_epsilon_bar = torch.log(epsilon_bar)
                log_epsilon_bar = m**(-k) * log_epsilon + (1 - m**(-k)) * log_epsilon_bar
                epsilon_bar = torch.exp(log_epsilon_bar)
            else:
                epsilon = epsilon_bar
        return theta_m

    def run_all_samples(self, theta_0, M, M_adapt, delta=0.65, print_please=False):
        """
        theta: params
        M: number of iter
        M_adapt: number of iter in which we allow adaption of epsilon
        """
        assert theta_0.shape[0] == 1
        samples = np.empty((M, theta_0.shape[-1]))
        epsilon = self.FindReasonableEpsilon(theta_0)
        print(f"found epsilon = {epsilon}")
        mu = torch.log10(epsilon)
        epsilon_bar = torch.ones_like(epsilon)
        H_bar = torch.zeros_like(epsilon)
        gamma = 0.05
        t_0 = 10
        k = 0.75

        theta_m = theta_0.clone()
        for m in range(1, M+1):
            r_0 = torch.randn_like(theta_m)
            log_u = torch.log(torch.rand_like(epsilon)) + self.joint_unnormalised_log_prob(theta_m_minus_1, r_0)[:, None]
            s = torch.ones_like(epsilon)
            theta_minus = theta_m.clone()
            theta_plus = theta_m.clone()
            r_minus = r_0.clone()
            r_plus = r_0.clone()
            j = 0
            n = torch.ones_like(epsilon)

            s_equals_1 = torch.squeeze(s == 1, dim=-1)

            theta_dash = torch.tensor(float("nan")).expand(theta_m_minus_1.shape)
            n_dash = torch.tensor(float("nan")).expand(n.shape)
            s_dash = torch.tensor(float("nan")).expand(s.shape)
            a = torch.tensor(float("nan")).expand(epsilon.shape)
            n_a = torch.tensor(float("nan")).expand(epsilon.shape)
            while s_equals_1.any():
                # note inside this loop we only only updates indices for which s == 1

                # currently sharing v across samples to make parellel easier
                # I think this should be okay, as sign of r is different across batch
                v = torch.tensor([1, -1])[torch.randint(low=0, high=2, size=(1,))]
                if v == -1:
                    theta_minus[s_equals_1], r_minus[s_equals_1], _, _, \
                    theta_dash[s_equals_1], n_dash[s_equals_1], s_dash[s_equals_1], a[s_equals_1], n_a[s_equals_1] \
                        = \
                        self.BuildTree(theta_minus[s_equals_1], r_minus[s_equals_1],
                                       log_u[s_equals_1], v, j, epsilon[s_equals_1],
                                       theta_m_minus_1[s_equals_1], r_0[s_equals_1])
                else:
                    _, _, theta_plus[s_equals_1], r_plus[s_equals_1], theta_dash[s_equals_1], \
                    n_dash[s_equals_1], s_dash[s_equals_1], a[s_equals_1], n_a[s_equals_1] = \
                        self.BuildTree(theta_plus[s_equals_1], r_plus[s_equals_1],
                                       log_u[s_equals_1], v, j, epsilon[s_equals_1],
                                       theta_m_minus_1[s_equals_1], r_0[s_equals_1])
                s_dash_equals_1 = (s_dash == 1).squeeze(dim=-1)
                s_dash_equals_1[~s_equals_1] = False  # these indices aren't partaking
                if s_dash_equals_1.any():
                    update_theta = torch.squeeze((n_dash/n > torch.rand_like(n_dash))[s_dash_equals_1])
                    theta_m[s_dash_equals_1][update_theta] = theta_dash[s_dash_equals_1][update_theta]
                n[s_equals_1] = n[s_equals_1] + n_dash[s_equals_1]
                s[s_equals_1] = s_dash[s_equals_1]*(
                        torch.einsum("ik,ik->i",
                            (theta_plus[s_equals_1] - theta_minus[s_equals_1]), r_minus[s_equals_1])
                        >= 0 )[:, None] \
                        * (torch.einsum("ik,ik->i",
                                       (theta_plus[s_equals_1] - theta_minus[s_equals_1]), r_plus[s_equals_1])
                           >= 0)[:, None]
                j = j + 1
                s_equals_1 = torch.squeeze(s == 1, dim=-1)
            samples[m, :] = theta_m.detach().cpu().numpy()
            if m <= M_adapt:
                H_bar = (1 - 1/(m + t_0))*H_bar + 1/(m + t_0)*(delta - a / n_a)
                log_epsilon = mu - np.sqrt(m)/gamma * H_bar
                epsilon = torch.exp(log_epsilon)
                if 'log_epsilon_bar' not in locals():
                    log_epsilon_bar = torch.log(epsilon_bar)
                log_epsilon_bar = m**(-k) * log_epsilon + (1 - m**(-k)) * log_epsilon_bar
                epsilon_bar = torch.exp(log_epsilon_bar)
                if print_please is True:
                    print("epsilon = ", epsilon)
                    print("epsilon_bar = ", epsilon_bar)
            elif m == M_adapt + 1:
                epsilon = epsilon_bar
            else:
                pass
        return samples



if __name__ == '__main__':
    from TargetDistributions.MoG import MoG
    from FittedModels.Models.DiagonalGaussian import DiagonalGaussian
    import matplotlib.pyplot as plt
    torch.set_default_dtype(torch.float64)

    n_samples = 1000

    dim = 2
    torch.manual_seed(2)
    target = MoG(dim=dim, n_mixes=5, loc_scaling=5, min_cov=10.0)
    target.distribution.set_default_validate_args(False)
    learnt_sampler = DiagonalGaussian(dim=dim, log_std_initial_scaling=1.0)
    sampler_samples = learnt_sampler(n_samples)[0]
    tester = NUTS(dim=dim, log_q_x=target.log_prob)

    theta_0 = learnt_sampler(n_samples)[0]
    eps = tester.FindReasonableEpsilon(theta_0)
    print(eps)
    theta = tester.run(theta_0, M=10, M_adapt=5)

    theta_0 = theta_0.cpu().detach()
    theta = theta.cpu().detach()
    print(theta)
    plt.plot(theta_0[:, 0], theta_0[:, 1], "o", alpha=0.5)
    plt.title("sampler samples")
    plt.show()
    plt.plot(theta[:, 0], theta[:, 1], "o", alpha=0.5)
    plt.title("annealed samples")
    plt.show()
    true_samples = target.sample((n_samples,)).cpu().detach()
    plt.plot(true_samples[:, 0], true_samples[:, 1], "o", alpha=0.5)
    plt.title("true samples")
    plt.show()




