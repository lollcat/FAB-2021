from NUTS_np import build_tree
from ImportanceSampling.SamplingAlgorithms.NUTS import NUTS
from NUTS_Problems import correlated_normal_torch, correlated_normal_np
import numpy as np
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    for seed in range(10):
        #seed = 4
        print(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        j = 10
        v = 1
        theta = np.random.randn(2)
        print(theta)
        theta_0 = theta # np.randn(2)
        r = np.random.randn(2)
        r0 = r
        logp, grad = correlated_normal_np(theta)
        joint = logp - 0.5 * np.dot(r0, r0.T)
        logu = float(joint - np.random.exponential(1, size=1))
        eps = np.ones(1)*0.5
        thetaminus, rminus, gradminus, thetaplus, \
        rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alphaprime, nalphaprime = \
            build_tree(theta, r, grad, logu, v, j, epsilon=eps, f=correlated_normal_np, joint0=joint)

        print("theta_minus", thetaminus)
        print("theta_plus", thetaplus)
        print("thetaprime", thetaprime)
        print("nprime", nprime)
        print("sprime", sprime)


        # pytorch
        theta = torch.tensor(theta, requires_grad=True)[None, :]
        theta = theta.clone()
        theta_0 = torch.tensor(theta)
        r = torch.tensor(r)[None, :]
        log_u = torch.tensor([logu])[None, :]
        r0 = torch.tensor(r0)[None, :]
        eps = torch.tensor(eps)[None, :]

        NUTTER = NUTS(2, correlated_normal_torch)
        theta_minus, r_minus, theta_plus, r_plus, theta_dash, n_dash, s_dash, a_dash, n_a_dash = \
            NUTTER.BuildTree(theta, r, log_u, v, j, eps, theta_0, r0)
        print("theta_minus", theta_minus)
        print("theta_plus", theta_plus)
        print("theta_prime", theta_dash)
        print("nprime", n_dash)
        print("sprime", s_dash)


