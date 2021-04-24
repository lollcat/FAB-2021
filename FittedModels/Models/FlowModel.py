import torch
import torch.nn as nn
import torch.nn.functional as F
from NormalisingFlow.IAF import IAF

class FlowModel(nn.Module):
    def __init__(self, x_dim, flow=IAF, n_flow_steps=3):
        super(FlowModel, self).__init__()
        self.prior = torch.distributions.MultivariateNormal(loc=torch.zeros(x_dim),
                                                            covariance_matrix=torch.eye(x_dim)*1)
        self.flow_blocks = nn.ModuleList([])
        for i in range(n_flow_steps):
            self.flow_blocks.append(flow(x_dim))

    def forward(self, batch_size=1):
        x = self.prior.sample((batch_size,))
        log_prob = self.prior.log_prob(x)
        for flow_step in self.flow_blocks:
            x, log_determinant = flow_step(x)
            log_prob += log_determinant
        return x, log_prob

    def log_prob(self, x):
        #TODO will need to reverse the flow here
        pass

if __name__ == '__main__':
    model = FlowModel(x_dim=5)
    x, log_prob = model(5)
    print(x.shape, log_prob.shape)