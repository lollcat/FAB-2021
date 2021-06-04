import torch

def MC_estimate_true_expectation(distribution, expectation_function, n_samples):
    # requires the distribution to be able to be sampled from
    x_samples = distribution.sample((n_samples,))
    f_x = expectation_function(x_samples)
    return torch.mean(f_x)

def quadratic_function(x, seed=0):
    # example function that we may want to calculate expectations over
    torch.manual_seed(seed)
    x_shift = 2*torch.randn(x.shape[-1]).to(x.device)
    A = 2*torch.rand((x.shape[-1], x.shape[-1])).to(x.device)
    b = torch.rand(x.shape[-1]).to(x.device)
    x = x + x_shift
    return torch.einsum("bi,ij,bj->b", x, A, x) + torch.einsum("i,bi->b", b, x)

def Rosenbrock_function(x):
    return torch.sum(100.0*(x[:, 1:] - x[:, :-1]**2.0)**2.0 + (1 - x[:, :-1])**2.0, dim=-1)

if __name__ == '__main__':
    from Utils.plotting_utils import plot_func2D
    import matplotlib.pyplot as plt
    expectation_function = Rosenbrock_function
    expectation_func_fig = plot_func2D(expectation_function, n_points=150, range=15)
    plt.show()
    x = torch.randn(3, 4)
    f_x = expectation_function(x)
    print(f_x.shape)


