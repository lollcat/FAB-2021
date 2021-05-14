import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowModel(nn.Module):
    """
    here forward goes from z -> x, and backwards from x-> z, could maybe re-order
    we are also assuming that we are only interested in p(x), so return this for both forwards and backwards,
    we could add methods for p(z) if this comes into play
    """
    def __init__(self, x_dim, flow_type="IAF", n_flow_steps=3, prior_scaling=1.0):
        self.dim = x_dim
        super(FlowModel, self).__init__()
        self.prior_scaling = torch.tensor([prior_scaling])
        self.prior = torch.distributions.MultivariateNormal(loc=torch.zeros(x_dim),
                                                                covariance_matrix=torch.eye(x_dim)*self.prior_scaling)
        if flow_type == "IAF":
            from NormalisingFlow.IAF import IAF
            flow = IAF
            self.flow_blocks = nn.ModuleList([])
            for i in range(n_flow_steps):
                self.flow_blocks.append(flow(x_dim))
        elif flow_type == "RealNVP":
            from NormalisingFlow.RealNVP import RealNVP
            flow = RealNVP
            self.flow_blocks = nn.ModuleList([])
            for i in range(n_flow_steps):
                reversed = i % 2 == 0
                self.flow_blocks.append(flow(x_dim, reversed=reversed))
        else:
            raise Exception("incorrectly specified flow")

    def forward(self, batch_size=1):
        """
        log p(x) = log p(z) - log |dx/dz|
        """
        x = self.prior.rsample((batch_size,))
        log_prob = self.prior.log_prob(x)
        for flow_step in self.flow_blocks:
            x, log_determinant = flow_step(x)
            log_prob -= log_determinant
        return x, log_prob

    def forward_with_hooks(self, batch_size=1):
    #def forward(self, batch_size=1):
        """
        for debugging, comment out forward and rename this function
        """
        x = self.prior.sample((batch_size,))
        log_prob = self.prior.log_prob(x)
        for i, flow_step in enumerate(self.flow_blocks):
            x, log_determinant = flow_step(x)
            log_prob -= log_determinant
            if i == 0:
                pass
                # x.register_hook(lambda grad: print("\n\ngrad x first", grad))
                # log_prob.register_hook(lambda grad: print("\n\ngrad log_prob first", grad))
        x.register_hook(lambda grad: print("\n\ngrad x final max, min, nana", grad.max(), grad.min(),
                                           torch.sum(torch.isnan(grad))))
        log_prob.register_hook(lambda grad: print("\n\ngrad log_prob final max min nan", grad.max(), grad.min(),
                                           torch.sum(torch.isnan(grad))))
        self.flow_blocks[-1].AutoregressiveNN.FirstLayer.latent_to_layer.weight. \
            register_hook(lambda grad: print("\n\ngrad layer first max min nan", grad.max(), grad.min(),
                                           torch.sum(torch.isnan(grad))))
        return x, log_prob

    def backward(self, x):
        """
        log p(x) = log p(z) + log |dz/dx|
        """
        log_prob = torch.zeros(x.shape[0])
        for flow_step in self.flow_blocks[::-1]:
            x, log_determinant = flow_step.backward(x)
            log_prob += log_determinant
        prior_prob = self.prior.log_prob(x)
        log_prob += prior_prob
        return x, log_prob

    def log_prob(self, x):
        x, log_prob = self.backward(x)
        return log_prob

    def sample(self, shape):
        # just a wrapper so we can call sample func for plotting like in torch.distributions
        x, log_prob = self.forward(shape[0])
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
            x, log_determinant = flow_step(x)
            log_prob -= log_determinant
        log_prob_backward = self.log_prob(x)
        z_backward = self.backward(x)[0]
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
    model = FlowModel(x_dim=2, n_flow_steps=2, prior_scaling=1)  # , flow_type="RealNVP"
    model.check_forward_backward_consistency()
    model.check_normalisation_constant(n=int(5e6))
    plot_distribution(model, range=15)
    plt.show()

