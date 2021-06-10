import torch
import torch.nn as nn
from FittedModels.Models.base import BaseLearntDistribution
from NormalisingFlow.ActNorm import ActNorm


class FlowModel(nn.Module, BaseLearntDistribution):
    """
    here forward goes from z -> x, and backwards from x-> z, could maybe re-order
    we are also assuming that we are only interested in p(x), so return this for both forwards and backwards,
    we could add methods for p(z) if this comes into play
    """
    def __init__(self, x_dim, flow_type="IAF", n_flow_steps=3, scaling_factor=1.0,
                 trainable_prior=True, *flow_args, **flow_kwargs):
        super(FlowModel, self).__init__()
        self.class_args = (x_dim, flow_type, n_flow_steps, scaling_factor, *flow_args)
        self.class_kwargs = flow_kwargs
        self.dim = x_dim
        self.scaling_factor = nn.Parameter(torch.tensor([scaling_factor]))

        if trainable_prior:
            self.prior_mean = nn.Parameter(torch.zeros(x_dim))
            self.log_prior_covariance_diag = nn.Parameter(torch.zeros(x_dim))
        else:
            self.register_buffer("prior_mean", torch.zeros(x_dim))
            self.register_buffer("log_prior_covariance_diag", torch.zeros(x_dim))

        if flow_type == "IAF":
            from NormalisingFlow.IAF import IAF
            flow = IAF
        elif flow_type == "RealNVP":
            from NormalisingFlow.RealNVP import RealNVP
            flow = RealNVP
        else:
            raise Exception("incorrectly specified flow")
        self.flow_blocks = nn.ModuleList([])
        for i in range(n_flow_steps):
            reversed = i % 2 == 0
            flow_block = flow(x_dim, reversed=reversed, *flow_args, **flow_kwargs)
            self.flow_blocks.append(flow_block)
            self.flow_blocks.append(ActNorm(x_dim))

    def set_flow_requires_grad(self, requires_grad):
        for parameter in self.flow_blocks.parameters():
            parameter.requires_grad = requires_grad

    @property
    def prior(self):
        covariance_matrix = torch.diag(torch.sigmoid(self.log_prior_covariance_diag)*2)
        return torch.distributions.MultivariateNormal(loc=self.prior_mean,
                                                                covariance_matrix=covariance_matrix)

    def forward(self, batch_size=1):
        # for the forward pass of the model we generate x samples
        # this shouldn't be confused with the flows forward vs inverse
        return self.z_to_x(batch_size=batch_size)

    def widen(self, x):
        x = x*self.scaling_factor
        log_det = x.shape[-1]*torch.log(self.scaling_factor)
        return x, log_det

    def un_widen(self, x):
        x = x/self.scaling_factor
        log_det = - x.shape[-1]*torch.log(self.scaling_factor)
        return x, log_det

    def z_to_x(self, batch_size=1):
        """
        Sample from z, transform to give x
        log p(x) = log p(z) - log |dx/dz|
        """
        x = self.prior.rsample((batch_size,))
        log_prob = self.prior.log_prob(x)
        for flow_step in self.flow_blocks:
            x, log_determinant = flow_step.inverse(x)
            log_prob -= log_determinant
        x, log_determinant = self.widen(x)
        log_prob -= log_determinant
        return x, log_prob

    def x_to_z(self, x):
        """
        Given x, find z and it's log probability
        log p(x) = log p(z) + log |dz/dx|
        """
        log_prob = torch.zeros(x.shape[0], device=x.device)
        x, log_det = self.un_widen(x)
        log_prob += log_det
        for flow_step in self.flow_blocks[::-1]:
            x, log_determinant = flow_step.forward(x)
            log_prob += log_determinant
        prior_prob = self.prior.log_prob(x)
        log_prob += prior_prob
        z = x
        return z, log_prob

    def log_prob(self, x):
        x, log_prob = self.x_to_z(x)
        return log_prob

    def sample(self, shape):
        # just a wrapper so we can call sample func for plotting like in torch.distributions
        x, log_prob = self.z_to_x(shape[0])
        return x


    def check_forward_backward_consistency(self, n=100):
        """p(x) generated from forward should be the same as log p(x) for the same samples"""
        """
        log p(x) = log p(z) - log |dx/dz|
        """
        # first let's go forward
        z = self.prior.rsample((n,))
        x = z
        prior_prob = self.prior.log_prob(x)
        log_prob = prior_prob.detach().clone() # use clone otherwise prior prob get's values get changed
        log_dets_forward = []
        for flow_step in self.flow_blocks:
            x, log_determinant = flow_step.inverse(x)
            log_prob -= log_determinant
            log_dets_forward.append(log_determinant.detach())
        x, log_determinant = self.widen(x)
        log_prob -= log_determinant
        log_dets_forward.append(log_determinant.detach())

        # now let's go backward
        log_dets_backward = []
        log_prob_backward = torch.zeros(x.shape[0], device=x.device)
        x, log_det = self.un_widen(x)
        log_dets_backward.append(log_det.detach())
        log_prob_backward += log_det
        for flow_step in self.flow_blocks[::-1]:
            x, log_determinant = flow_step.forward(x)
            log_prob_backward += log_determinant
            log_dets_backward.append(log_determinant.detach())
        prior_prob_back = self.prior.log_prob(x).detach()
        log_prob_backward += prior_prob_back
        z_backward = x

        print(f"Checking forward backward consistency of x, the following should be close to zero: "
              f"{torch.max(torch.abs(z - z_backward))}")
        print(f"Checking foward backward consistency p(x), the following number should be close to zero "
              f"{torch.max(torch.abs(log_prob - log_prob_backward))}")
        print(f"prior max difference {torch.abs(prior_prob - prior_prob_back).max()}")
        print("\n\nthe following should all be close to 0: \n\n")
        for i, log_det_forward in enumerate(log_dets_forward):
            print(torch.max(torch.abs(log_det_forward + log_dets_backward[-i-1])))

    @torch.no_grad()
    def check_normalisation_constant(self, n=int(1e6)):
        """This should be approximately one if things are working correctly, check with importance sampling"""
        normal_dist = torch.distributions.MultivariateNormal(torch.zeros(self.dim), 5*torch.eye(self.dim))
        x_samples = normal_dist.rsample((n,))
        log_prob_normal = normal_dist.log_prob(x_samples)
        log_prob_backward = self.log_prob(x_samples)
        importance_weights = torch.exp(log_prob_backward - log_prob_normal)
        Z_backward = torch.mean(importance_weights)
        print(f"normalisation constant is {Z_backward}")


if __name__ == '__main__':
    from Utils.plotting_utils import plot_distribution
    import matplotlib.pyplot as plt
    torch.manual_seed(1)
    model = FlowModel(x_dim=2, n_flow_steps=4, scaling_factor=1.5, use_exp=False) #, flow_type="RealNVP")  #
    model(100)
    x, log_prob = model.forward(100)
    #  x, log_prob = model.batch_forward(100, 10)
    print(f"std: {torch.std(x, dim=0)}") # test ActNorm
    print(f"std: {torch.mean(x, dim=0)}")
    log_prob_ = model.batch_log_prob(x, 10)
    samples = model.batch_sample((100,), 10)
    model.check_forward_backward_consistency()
    model.check_normalisation_constant(n=int(1e7))
    #plot_distribution(model, bounds=[[-2, 2], [-2, 2]])
    #plt.show()

