import torch
import torch.nn as nn
import torch.nn.functional as F
from NormalisingFlow.base import BaseFlow

class FlowModel(nn.Module):
    """
    here forward goes from z -> x, and backwards from x-> z, could maybe re-order
    we are also assuming that we are only interested in p(x), so return this for both forwards and backwards,
    we could add methods for p(z) if this comes into play
    """
    def __init__(self, x_dim, flow_type="IAF", n_flow_steps=3, scaling_factor=1.0, *flow_args, **flow_kwargs):
        self.class_definition = (x_dim, flow_type, n_flow_steps, scaling_factor, *flow_args, *flow_kwargs)
        self.dim = x_dim
        super(FlowModel, self).__init__()
        self.scaling_factor = torch.tensor([scaling_factor])
        self.register_buffer("prior_mean", torch.zeros(x_dim))
        self.register_buffer("prior_covariance", torch.eye(x_dim))
        self.prior = torch.distributions.MultivariateNormal(loc=self.prior_mean,
                                                                covariance_matrix=self.prior_covariance)

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
        x = self.prior.sample((batch_size,))
        log_prob = self.prior.log_prob(x)
        for flow_step in self.flow_blocks:
            x, log_determinant = flow_step.inverse(x)
            log_prob -= log_determinant
        if self.scaling_factor != 1:
            x, log_determinant = self.widen(x)
            log_prob -= log_determinant
        return x, log_prob

    def x_to_z(self, x):
        """
        Given x, find z and it's log probability
        log p(x) = log p(z) + log |dz/dx|
        """
        log_prob = torch.zeros(x.shape[0], device=x.device)
        if self.scaling_factor != 1:
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
        z = self.prior.sample((n,))
        x = z
        log_prob = self.prior.log_prob(x)
        for flow_step in self.flow_blocks:
            x, log_determinant = flow_step.inverse(x)
            log_prob -= log_determinant
        if self.scaling_factor != 1:
            x, log_determinant = self.widen(x)
            log_prob -= log_determinant
        log_prob_backward = self.log_prob(x)
        z_backward = self.x_to_z(x)[0]
        print(f"Checking forward backward consistency of x, the following should be close to zero: "
              f"{torch.max(torch.abs(z - z_backward))}")
        print(f"Checking foward backward consistency p(x), the following number should be close to zero "
              f"{torch.max(torch.abs(log_prob - log_prob_backward))}")


    def check_normalisation_constant(self, n=int(1e6)):
        """This should be approximately one if things are working correctly, check with importance sampling"""
        normal_dist = torch.distributions.MultivariateNormal(torch.zeros(self.dim), 5*torch.eye(self.dim))
        x_samples = normal_dist.sample((n,))
        log_prob_normal = normal_dist.log_prob(x_samples)
        log_prob_backward = self.log_prob(x_samples)
        importance_weights = torch.exp(log_prob_backward - log_prob_normal)
        Z_backward = torch.mean(importance_weights)
        print(f"normalisation constant is {Z_backward}")


if __name__ == '__main__':
    from Utils import plot_distribution
    import matplotlib.pyplot as plt
    torch.manual_seed(1)
    model = FlowModel(x_dim=2, n_flow_steps=3, scaling_factor=2.0, use_exp=False, flow_type="RealNVP")  #
    model(100)
    model.check_forward_backward_consistency()
    model.check_normalisation_constant(n=int(5e6))
    plot_distribution(model,)
    plt.show()

