import torch

def MC_estimate_true_expectation(distribution, expectation_function, n_samples):
    # requires the distribution to be able to be sampled from
    x_samples = distribution.sample((n_samples,))
    f_x = expectation_function(x_samples)
    return torch.mean(f_x)

def expectation_function(x):
    # just an example expectation function
    A = torch.ones((x.shape[-1], x.shape[-1])).to(x.device)
    return torch.einsum("bi,ij,bj->b", x, A, x)

