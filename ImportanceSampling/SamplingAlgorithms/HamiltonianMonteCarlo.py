import torch



def HMC_func(U, epsilon, n_outer, L, current_q, grad_U):
    # base function for HMC written in terms of potential energy function U
    for n in range(n_outer):
        q = current_q
        p = torch.randn_like(q)
        current_p = p

        # make momentum half step
        p = p - epsilon * grad_U(q) / 2

        # Now loop through position and momentum leapfrogs
        for i in range(L):
            # Make full step for position
            q = q + epsilon*p
            # Make a full step for the momentum if not at end of trajectory
            if i != L-1:
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
        #print(f"avg probability of accept is {torch.mean(torch.clip(acceptance_probability, max=1))}")
        accept = (acceptance_probability > torch.rand(acceptance_probability.shape).to(q.device)).int()
        accept = accept[:, None].repeat(1, q.shape[-1])
        current_q = accept * q + (1 - accept) * current_q
    return current_q


def HMC(log_q_x, epsilon, n_outer, L, current_q, grad_log_q_x=None):
    # currently mainly written with grad_log_q_x = None in mind
    # using diagonal, would be better to use vmap (need to wait until this is released doh)
    """
    Following: https://arxiv.org/pdf/1206.1901.pdf
    :return:
    :param log_q_x: target probility function (unnormalised)
    Note typically q(x) is sampling dist, but in this case its the target, as p is for momentum
    :param epsilon: tensor controlling step size
    :param L:
    :param current_q:
    :param grad_log_q_x: if None then use auto-grad
    :return: x_samples from p(x)

    U is potential energy, q is position, p is momentum
    """
    def U(x: torch.Tensor):
        return - log_q_x(x)
    if grad_log_q_x is None:
        assert current_q.requires_grad
        def grad_U(q: torch.Tensor):
            y = U(q)
            return torch.autograd.grad(y, q, grad_outputs=torch.ones_like(y))[0]
            #return torch.diagonal(torch.autograd.functional.jacobian(U, q, vectorize=False), dim1=0, dim2=1).T
    else:
        grad_U = -grad_log_q_x

    current_q = HMC_func(U, epsilon, n_outer, L, current_q, grad_U)
    return current_q

if __name__ == '__main__':
    from TargetDistributions.MoG import MoG
    from FittedModels.Models.DiagonalGaussian import DiagonalGaussian
    import matplotlib.pyplot as plt
    n_samples = 2000
    dim = 2
    torch.manual_seed(2)
    target = MoG(dim=dim, n_mixes=5, loc_scaling=5)
    learnt_sampler = DiagonalGaussian(dim=dim, log_std_initial_scaling=2.0)
    sampler_samples = learnt_sampler(n_samples)[0]
    x_HMC = HMC(log_q_x=target.log_prob, n_outer=10, epsilon=torch.tensor([1.0]), L=2, current_q=sampler_samples
                , grad_log_q_x=None)

    sampler_samples = sampler_samples.cpu().detach()
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









