from NUTS_np import build_tree
from ImportanceSampling.SamplingAlgorithms.NUTS import NUTS
from NUTS_Problems import correlated_normal_torch, correlated_normal_np
import numpy as np
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    n_runs = 10
    theta_batch = np.zeros((n_runs, 2))
    r_batch = np.zeros((n_runs, 2))
    log_u_batch = np.zeros((n_runs, 1))
    eps_batch = np.zeros((n_runs, 1))
    NUTTER = NUTS(2, correlated_normal_torch)

    theta_minus_results = np.zeros((n_runs, 2))

    for seed in range(n_runs):
        print(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        j = 10
        v = -1
        theta = np.random.randn(2)
        theta_0 = theta # np.randn(2)
        r = np.random.randn(2)
        r0 = r
        logp, grad = correlated_normal_np(theta)
        joint = logp - 0.5 * np.dot(r0, r0.T)
        logu = float(joint - np.random.exponential(1, size=1))
        eps = np.ones(1)*0.5

        theta_batch[seed] = theta
        r_batch[seed] = r
        log_u_batch[seed] = logu
        eps_batch[seed] = eps

        # pytorch
        theta = torch.tensor(theta, requires_grad=True)[None, :]
        theta = theta.clone()
        theta_0 = torch.tensor(theta)
        r = torch.tensor(r)[None, :]
        log_u = torch.tensor([logu])[None, :]
        r0 = torch.tensor(r0)[None, :]
        eps = torch.tensor(eps)[None, :]


        theta_minus, r_minus, theta_plus, r_plus, theta_dash, n_dash, s_dash, a_dash, n_a_dash = \
            NUTTER.BuildTree(theta, r, log_u, v, j, eps, theta_0, r0)
        """
        print("theta_minus", theta_minus)
        print("theta_plus", theta_plus)
        print("theta_prime", theta_dash)
        print("nprime", n_dash)
        print("sprime", s_dash)
        """
        theta_minus_results[seed] = theta_minus.detach().cpu().numpy()


    # batch
    theta_batch = torch.tensor(theta_batch, requires_grad=True)
    r_batch = torch.tensor(r_batch)
    log_u_batch = torch.tensor(log_u_batch)
    eps_batch = torch.tensor(eps_batch)

    theta_0_batch = theta_batch.clone()
    r_0_batch = r_batch.clone()

    theta_minus, r_minus, theta_plus, r_plus, theta_dash, n_dash, s_dash, a_dash, n_a_dash = \
        NUTTER.BuildTree(theta_batch, r_batch, log_u_batch, v, j, eps_batch, theta_0_batch, r_0_batch)

    print(theta_minus_results)
    print(theta_minus)
    #print(theta_plus)


