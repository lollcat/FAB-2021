import torch
from ImportanceSampling.AnnealedImportanceSampler import AnnealedImportanceSampler

class Debugger(AnnealedImportanceSampler):
    def __init__(self, *args, **kwargs):
        super(Debugger, self).__init__(*args, **kwargs)

    def run_with_checks(self, n_runs):
        # first set flow init zeros to false so we can see what's cracking if our learnt distribution is a flow
        log_w = torch.zeros(n_runs).to(self.device)  # log importance weight
        x_new, log_prob_p0 = self.sampling_distribution(n_runs)
        log_w += self.intermediate_unnormalised_log_prob(x_new, 1) - log_prob_p0
        print(torch.autograd.grad(torch.sum(x_new), self.sampling_distribution.flow_blocks[
            0].AutoregressiveNN.FirstLayer.latent_to_layer.weight, retain_graph=True))
        print(torch.autograd.grad(torch.sum(log_w), self.sampling_distribution.flow_blocks[
            0].AutoregressiveNN.FirstLayer.latent_to_layer.weight, retain_graph=True))
        for j in range(1, self.n_distributions-1):
            x_new = self.Metropolis_transition(x_new, j)
            log_w += self.intermediate_unnormalised_log_prob(x_new, j+1) - \
                     self.intermediate_unnormalised_log_prob(x_new, j)
            if self.save_for_visualisation:
                if (j+1) % self.save_spacing == 0:
                    self.log_w_history.append(log_w.cpu().detach())
                    self.samples_history.append(x_new.cpu().detach())
        print(torch.autograd.grad(torch.sum(x_new), self.sampling_distribution.flow_blocks[
            0].AutoregressiveNN.FirstLayer.latent_to_layer.weight, retain_graph=True))
        print(torch.autograd.grad(torch.sum(log_w), self.sampling_distribution.flow_blocks[
            0].AutoregressiveNN.FirstLayer.latent_to_layer.weight, retain_graph=True))
        return x_new, log_w

    def Metropolis_transition(self, x_original, j):
        x = x_original
        for n in range(self.n_steps_transition_operator):
            x_proposed = x + torch.randn(x.shape).to(x.device) * self.step_size
            x_proposed_log_prob = self.intermediate_unnormalised_log_prob(x_proposed, j)
            x_prev_log_prob = self.intermediate_unnormalised_log_prob(x, j)
            acceptance_probability = torch.exp(x_proposed_log_prob - x_prev_log_prob)
            # not that sometimes this will be greater than one, corresonding to 100% probability of acceptance
            accept = (acceptance_probability > torch.rand(acceptance_probability.shape).to(x.device)).int()
            accept = accept[:, None].repeat(1, x.shape[-1])
            x = accept*x_proposed + (1-accept)*x
        print(torch.autograd.grad(torch.sum(x), x_original, retain_graph=True))
        return x

if __name__ == '__main__':
    from TargetDistributions.MoG import MoG
    from FittedModels.Models.FlowModel import FlowModel
    dim = 2
    n_samples_estimation = int(1e4)
    n_samples_expectation = int(1e6)
    n_samples = int(1e3)
    flow_scaling = 2.0

    target = MoG(dim=dim, n_mixes=10, min_cov=1, loc_scaling=5)
    learnt_sampler = FlowModel(x_dim=dim, scaling_factor=flow_scaling, flow_type="IAF", n_flow_steps=3)
    debug = Debugger(sampling_distribution=learnt_sampler, target_distribution=target,
                 n_distributions=3, n_updates_Metropolis=2, save_for_visualisation=True, save_spacing=20,
                 distribution_spacing="geometric", noise_scaling=1.0)

    debug.run_with_checks(n_runs=10)