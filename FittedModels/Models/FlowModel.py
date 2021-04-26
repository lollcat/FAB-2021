import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowModel(nn.Module):
    """
    here forward goes from z -> x, and backwards from x-> z, could maybe re-order
    we are also assuming that we are only interested in p(x), so return this for both forwards and backwards,
    we could add methods for p(z) if this comes into play
    """
    def __init__(self, x_dim, flow_type="IAF", n_flow_steps=3):
        self.dim = x_dim
        super(FlowModel, self).__init__()
        self.prior = torch.distributions.MultivariateNormal(loc=torch.zeros(x_dim),
                                                            covariance_matrix=torch.eye(x_dim)*4)
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
        x = self.prior.sample((batch_size,))
        log_prob = self.prior.log_prob(x)
        for flow_step in self.flow_blocks:
            x, log_determinant = flow_step(x)
            log_prob -= log_determinant
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


    def check_normalisation_constant(self, n=500000):
        """This should be approximately one if things are working correctly, check with importance sampling"""
        normal_dist = torch.distributions.MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))
        x_samples = normal_dist.sample((n,))
        log_prob_normal = normal_dist.log_prob(x_samples)
        log_prob_backward = self.log_prob(x_samples)
        importance_weights = torch.exp(log_prob_backward - log_prob_normal)
        Z_backward = torch.mean(importance_weights)
        print(f"normalisation constant is {Z_backward}")



if __name__ == '__main__':
    model = FlowModel(x_dim=5, n_flow_steps=3, flow_type="RealNVP") #
    model.check_forward_backward_consistency()
    model.check_normalisation_constant()

