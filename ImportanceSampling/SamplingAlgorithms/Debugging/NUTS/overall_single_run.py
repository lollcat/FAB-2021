from ImportanceSampling.SamplingAlgorithms.NUTS import NUTS
from NUTS_Problems import correlated_normal_torch, correlated_normal_np
from NUTS_np import nuts6
import numpy as np
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    torch.manual_seed(0)
    M = 30
    Madapt = int(M/2)
    NUTTER = NUTS(2, correlated_normal_torch)
    theta_init = torch.randn((1,2), requires_grad=True)
    #theta_out = NUTTER.run(theta_init, M=5, M_adapt=2)
    #print(theta_out)

    samples_torch = NUTTER.run_all_samples(theta_init, M=M, M_adapt=Madapt, print_please=True)

    #print(samples)
    #samples_torch = np.clip(samples_torch, -10, 10)
    plt.plot(samples_torch[:, 0], samples_torch[:, 1], "o-b")
    plt.show()

    print("\n\n running numpy version \n\n")
    samples, lnprob, epsilon = nuts6(correlated_normal_np, M, Madapt, np.squeeze(theta_init.detach().cpu().numpy())
                                     , delta=0.65, print_please=True)

    plt.plot(samples[:, 0], samples[:, 1], "o-r")
    plt.show()
