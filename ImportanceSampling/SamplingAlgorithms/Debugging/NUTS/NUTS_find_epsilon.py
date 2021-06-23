from NUTS_np import find_reasonable_epsilon
from ImportanceSampling.SamplingAlgorithms.NUTS import NUTS
from NUTS_Problems import correlated_normal_torch, correlated_normal_np
import numpy as np
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    adjusters = np.array([[-2.0, 0, 0.001, 0.1, 0.5, 1.0, 10.0]]).T
    n_runs = 200
    np_results = np.zeros((n_runs, adjusters.size))
    torch_results = np.zeros((n_runs, adjusters.size))
    torch_results_batch = np.zeros((n_runs, adjusters.size))
    for j, adjuster in enumerate(adjusters):
        adjuster = adjuster[0]
        for i in range(n_runs):
            theta0 = np.zeros(2) + adjuster
            logp0, grad0 = correlated_normal_np(theta0)
            eps = find_reasonable_epsilon(theta0, grad0, logp0, correlated_normal_np)
            np_results[i, j] = eps
            #print(eps)

            # pytorch
            theta0 = torch.zeros(1, 2, requires_grad=True) + adjuster
            NUTTER = NUTS(2, correlated_normal_torch)
            eps = NUTTER.FindReasonableEpsilon(theta0)
            torch_results[i, j] = eps.cpu().detach().numpy()
            print(eps.item())

        # batch version of NUTS pytorch
        theta0 = torch.zeros(n_runs, 2, requires_grad=True) + adjuster
        NUTTER = NUTS(2, correlated_normal_torch)
        eps = NUTTER.FindReasonableEpsilon(theta0)
        torch_results_batch[:, j] = np.squeeze(eps.cpu().numpy())


    plt.plot(adjusters, np_results.T, "xr", label="numpy", markersize=10)
    plt.plot(adjusters, np.mean(np_results, axis=0), "--r", label="numpy", linewidth=10)
    plt.plot(adjusters, torch_results.T, "ob", label="torch")
    plt.plot(adjusters, np.mean(torch_results, axis=0), "-b", label="numpy", linewidth=5)
    plt.plot(adjusters, torch_results_batch.T, "black", label="torch_batch", marker="o", markersize=3, linestyle="None")
    plt.plot(adjusters, np.mean(torch_results_batch, axis=0), "black", label="numpy", linewidth=3)

    plt.show()





