import torch
import matplotlib.pyplot as plt
from FittedModels.utils import plot_distributions
torch.manual_seed(5)
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
from TargetDistributions.MoG import MoG
from TargetDistributions.Guassian_FullCov import Guassian_FullCov
from FittedModels.Models.DiagonalGaussian import DiagonalGaussian
from FittedModels.utils import plot_distributions, plot_samples
from FittedModels.train import LearntDistributionManager
from Utils import plot_func2D, MC_estimate_true_expectation, plot_distribution
from FittedModels.Models.FlowModel import FlowModel
from FittedModels.experimental.train_AIS import AIS_trainer
from FittedModels.utils import plot_history
import matplotlib.pyplot as plt
import torch
# setup expectation function
def expectation_function(x):
    A = torch.ones((x.shape[-1], x.shape[-1]))
    return torch.einsum("bi,ij,bj->b", x, A, x)



if __name__ == '__main__':
    """
    Bug Evidence
        - when we make sample size really big it's not happy
        - breaks for both kl and DReG
        - breaks only when we use p(x) in the equation
            - i.e. if we pytorch.no_grad then it un-breaks (but then reparam trick is wrong)
    
    Hypotheses for bug:
    - try different seed later on?
        - nope
    - does low learning rate help?
    - Function PowBackward0 error
        - loss is too big and negative
            - test clipping: so far doesn't fix
        - slicing
    - need to do big batches to keep stability?
    """
    seed = 0  # if we set this to 1 it is happy
    epochs = 500
    batch_size = 1000
    dim = 2
    n_samples_estimation = int(1e4)
    target = Guassian_FullCov(dim=dim, scale_covariance=1)

    torch.manual_seed(seed)
    learnt_sampler = FlowModel(x_dim=dim, n_flow_steps=3, prior_scaling=50)
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type="DReG", lr=1e-3) #
    #fig_before_train = plot_distributions(tester, n_points=200, grid=False, log_prob=True)
    #plot_samples(tester)
    #plt.show()

    history_debug = tester.debugging(epochs, batch_size=batch_size)
    history = tester.train(epochs, batch_size=batch_size)
    expectation, info = tester.estimate_expectation(n_samples_estimation, expectation_function)

    plot_history(history)
    plt.show()

    print(f"effective sample size is {info['effective_sample_size']} \n"
         f"variance in weights is {torch.var(info['normalised_sampling_weights'])}")
    fig_after_train = plot_distributions(tester, grid=False)
    plt.show()
