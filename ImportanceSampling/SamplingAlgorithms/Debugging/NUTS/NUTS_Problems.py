import numpy as np
import torch
torch.set_default_dtype(torch.float64)

class Counter:
    def __init__(self, c=0):
        self.c = c


c = Counter()


def correlated_normal_np(theta):
    """
    Example of a target distribution that could be sampled from using NUTS.
    (Although of course you could sample from it more efficiently)
    Doesn't include the normalizing constant.
    """

    # Precision matrix with covariance [1, 1.98; 1.98, 4].
    # A = np.linalg.inv( cov )
    A = np.asarray([[50.251256, -24.874372],
                    [-24.874372, 12.562814]])

    # add the counter to count how many times this function is called
    c.c += 1

    grad = -np.dot(theta, A)
    logp = 0.5 * np.dot(grad, theta.T)
    return logp, grad

def correlated_normal_torch(theta):
    """
    Example of a target distribution that could be sampled from using NUTS.
    (Although of course you could sample from it more efficiently)
    Doesn't include the normalizing constant.
    """

    # Precision matrix with covariance [1, 1.98; 1.98, 4].
    # A = np.linalg.inv( cov )
    A = torch.tensor(np.asarray([[50.251256, -24.874372],
                    [-24.874372, 12.562814]])).double()

    # add the counter to count how many times this function is called
    c.c += 1
    grad = -torch.einsum("ij,jk->ik", theta, A)
    logp = 0.5 * torch.einsum("ij,ij->i", grad, theta)
    return logp

if __name__ == '__main__':
    print(correlated_normal_torch(torch.zeros(10, 2, requires_grad=True)).shape)
    theta = torch.zeros(1, 2, requires_grad=True)+0.1
    L = correlated_normal_torch(theta)
    print(L)
    print(torch.autograd.grad(L, theta))
    print(correlated_normal_np(np.zeros((1, 2))+0.1))
