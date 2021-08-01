from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

def sample_energy(sample_and_log_w_func, x_index=0, n_samples=int(1e5), n_repeat=20, nbins=30):
    # frorm histogram - used to estimate free energy differences (by clumping within histo we are aggregating)
    hist_x = None
    whist_x = None
    hists_y = []
    whists_y = []
    for i in tqdm(range(n_repeat)):
        y, log_totalweights = sample_and_log_w_func(n_samples)
        in_bounds = torch.abs(y[:, x_index])  < 3 # only look inside these bounds for free energy diff
        y = y[in_bounds]
        log_totalweights = log_totalweights[in_bounds] # drop nan weights
        non_nan_weights = ~(torch.isnan(log_totalweights) | torch.isinf(log_totalweights) )
        y = y[non_nan_weights]
        log_totalweights = log_totalweights[non_nan_weights]
        log_totalweights -= log_totalweights.mean()
        samples = y[:, x_index]
        hist_y, edges = np.histogram(samples.detach().numpy(), bins=nbins, density=True)
        hists_y.append(-np.log(hist_y))
        hist_x = 0.5 * (edges[1:] + edges[:-1])
        #
        whist_y, edges = np.histogram(samples.detach().numpy(), bins=nbins, density=True,
                                      weights=np.exp(log_totalweights.detach().numpy()))
        whists_y.append(-np.log(whist_y))
        whist_x = 0.5 * (edges[1:] + edges[:-1])

    # align energies
    for i in range(n_repeat):
        hists_y[i] -= np.mean(hists_y[i][np.isfinite(hists_y[i])])
        whists_y[i] -= np.mean(whists_y[i][np.isfinite(whists_y[i])])

    return hist_x, hists_y, whist_x, whists_y


def bias_uncertainty(target_energy, x, energies, w_x, w_energies):
    X = torch.Tensor(np.vstack([x, np.zeros((1, len(x)))]).T)
    E_target = target_energy.energy(X)[:, 0]
    E_target -= E_target.min()

    # unweighted
    E_mean = np.mean(energies, axis=0)
    E_mean -= E_mean.min()

    # weighted
    Ew_mean = np.mean(w_energies, axis=0)
    Ew_mean -= Ew_mean.min()

    I = np.logical_and(x > -2.25, x < 2.25)
    # bias
    bias_unweighted = E_target - E_mean
    bias_unweighted = bias_unweighted.detach().numpy()
    J = np.isfinite(bias_unweighted)
    bias_unweighted = np.abs(bias_unweighted[I*J].mean())
    bias_reweighted = E_target - Ew_mean
    bias_reweighted = bias_reweighted.detach().numpy()
    J = np.isfinite(bias_reweighted)
    bias_reweighted = np.abs(bias_reweighted[I*J].mean())
    # uncertainty
    std_unweighted = np.array(energies)[:, I*J].std(axis=0).mean()
    std_reweighted = np.array(w_energies)[:, I*J].std(axis=0).mean()

    return bias_unweighted, std_unweighted, bias_reweighted, std_reweighted


def plot_energy(target_energy, x, energies, w_x, w_energies, ylabel=False, nstd=2.0, figsize=(4, 4)):
    fig = plt.figure(figsize=figsize)

    X = torch.Tensor(np.vstack([np.linspace(-3, 3, num=100), np.zeros((1, 100))]).T)
    E_target = target_energy.energy(X)
    E_target -= E_target.min()
    plt.plot(X[:, 0], E_target, linewidth=3, color='#444444')

    # unweighted
    E_mean = np.mean(energies, axis=0)
    E_mean -= E_mean.min()
    plt.errorbar(x, E_mean, nstd * np.std(energies, axis=0), color='red', linewidth=2)

    # weighted
    Ew_mean = np.mean(w_energies, axis=0)
    Ew_mean -= Ew_mean.min()
    plt.errorbar(w_x, Ew_mean, nstd * np.std(w_energies, axis=0), color='green', linewidth=2)

    plt.ylim(-1, 14)
    plt.xlabel('$x_1$')
    if ylabel:
        plt.ylabel('Energy (kT)')
    else:
        plt.yticks([])
    return fig
