from ImportanceSampling.SamplingAlgorithms.NUTS import NUTS
from NUTS_Problems import correlated_normal_torch, correlated_normal_np
from NUTS_np import nuts6
import numpy as np
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    torch.manual_seed(0)
    M = 10
    Madapt = int(M/2)
    NUTTER = NUTS(2, correlated_normal_torch)
    theta_init = torch.randn((100,2), requires_grad=True)

    samples_torch = NUTTER.run(theta_init, M=M, M_adapt=Madapt, print_please=False)

    plt.plot(samples_torch[:, 0], samples_torch[:, 1], "o-b")
    plt.show()

