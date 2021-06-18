import torch



def HMC_func(U, epsilon, n_outer, L, current_q, grad_U):
    # base function for HMC written in terms of potential energy function U
    assert 1 == n_outer
    for n in range(n_outer):
        trajectory = []
        q = current_q
        p = torch.randn_like(q)
        current_p = p

        # make momentum half step
        p = p - epsilon * grad_U(q) / 2

        # Now loop through position and momentum leapfrogs

        trajectory.append(q.detach())
        for i in range(L):
            # Make full step for position
            q = q + epsilon*p
            # Make a full step for the momentum if not at end of trajectory
            if i != L-1:
                p = p - epsilon * grad_U(q)
            trajectory.append(q.detach())

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
    return trajectory


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

    trajectory = HMC_func(U, epsilon, n_outer, L, current_q, grad_U)
    return torch.stack(trajectory, dim=0)

if __name__ == '__main__':
    from TargetDistributions.MoG import MoG
    from FittedModels.Models.DiagonalGaussian import DiagonalGaussian
    import matplotlib.pyplot as plt
    n_samples = 5
    dim = 2
    torch.manual_seed(0)
    target = MoG(dim=dim, n_mixes=5, loc_scaling=10)
    learnt_sampler = DiagonalGaussian(dim=dim, log_std_initial_scaling=5.0)
    sampler_samples = learnt_sampler(n_samples)[0]
    trajectory = HMC(log_q_x=target.log_prob, n_outer=1, epsilon=torch.tensor([1.0]), L=40, current_q=sampler_samples
                , grad_log_q_x=None)

    plt.plot(trajectory[:,0,  0], trajectory[:, 0, 1], "o-r")
    plt.plot(trajectory[:, 1, 0], trajectory[:, 1, 1], "o-b")
    plt.plot(trajectory[:,2,  0], trajectory[:, 0, 1], "o-g")
    plt.plot(trajectory[:, 3, 0], trajectory[:, 1, 1], "o-c")
    plt.plot(trajectory[:, 4, 0], trajectory[:, 1, 1], "o-y")

    n_samples = 1000
    true_samples = target.sample((n_samples,)).cpu().detach()
    plt.plot(true_samples[:, 0], true_samples[:, 1], "o", alpha=0.05)
    plt.show()









