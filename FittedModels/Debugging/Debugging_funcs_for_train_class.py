from FittedModels.train import LearntDistributionManager
import torch


class debugger(LearntDistributionManager):
    def __init__(self, *args, **kwargs):
        super(debugger, self).__init__(*args, **kwargs)

    def debugging_check_jacobians(self, loss, log_q_x, log_p_x, x_samples):
        first_level_differentiate = torch.autograd.grad(loss, log_q_x, retain_graph=True)
        last_level_differentiate = \
            torch.autograd.grad(loss,
                                self.learnt_sampling_dist.flow_blocks[
                                    0].AutoregressiveNN.FirstLayer.latent_to_layer.weight,
                                retain_graph=True)  # gives nans
        first_to_mid_level_grad = torch.autograd.grad(torch.sum(log_q_x), self.learnt_sampling_dist.flow_blocks[
            0].AutoregressiveNN.FirstLayer.latent_to_layer.weight,
                                                      retain_graph=True)
        x_sample = x_samples[0, :]
        log_prob_x_sample = self.target_dist.log_prob(x_sample)
        reparam_now = torch.autograd.grad(log_prob_x_sample, x_sample,
                                          retain_graph=True)
        reparam_equiv = torch.autograd.grad(log_p_x[0], x_samples[:, 0],
                                            retain_graph=True)
        return


    def check_DReG_gradients2(self, batch_size=10, max_grad_norm=1, clip_grad=True):
        if (self.loss_type == "DReG" or self.loss_type == "DReG_kl") and self.k is None:
            self.k = batch_size
        for epoch in range(3):
            self.optimizer.zero_grad()
            x_samples, log_q_x = self.learnt_sampling_dist(batch_size)
            log_p_x = self.target_dist.log_prob(x_samples)
            self.update_fixed_version_of_learnt_distribution()
            log_q_x = self.fixed_learnt_sampling_dist.log_prob(x_samples)
            check_1 = torch.autograd.grad(torch.sum(log_q_x), x_samples,
                                          retain_graph=True)
            check_2 = torch.autograd.grad(torch.sum(log_q_x), self.learnt_sampling_dist.flow_blocks[
                0].AutoregressiveNN.FirstLayer.latent_to_layer.weight_g,
                                          retain_graph=True)  # should have a value

            self.optimizer.zero_grad()
            with torch.no_grad():
                x_samples, log_q_x = self.learnt_sampling_dist(batch_size)
            x_samples.requires_grad = True
            log_p_x = self.target_dist.log_prob(x_samples)
            self.update_fixed_version_of_learnt_distribution()
            log_q_x = self.fixed_learnt_sampling_dist.log_prob(x_samples)
            check_1 = torch.autograd.grad(torch.sum(log_q_x), x_samples,
                                          retain_graph=True)
            check_2 = torch.autograd.grad(torch.sum(log_q_x), self.learnt_sampling_dist.flow_blocks[
                0].AutoregressiveNN.FirstLayer.latent_to_layer.weight_g,
                                          retain_graph=True)  # should have no value

            """
            log_w = log_p_x - log_q_x
            outside_dim = log_q_x.shape[0]/self.k  # this is like a batch dimension that we average DReG estimation over
            assert outside_dim % 1 == 0  # always make k & n_samples work together nicely for averaging
            outside_dim = int(outside_dim)
            log_w = log_w.reshape((outside_dim, self.k))
            with torch.no_grad():
                w_alpha_normalised_alpha = F.softmax(self.alpha*log_w, dim=-1)
            DreG_for_each_batch_dim = - self.alpha_one_minus_alpha_sign * \
                        torch.sum(((1 - self.alpha) * w_alpha_normalised_alpha + self.alpha * w_alpha_normalised_alpha**2)
                                  * log_w, dim=-1)
            dreg_loss = torch.mean(DreG_for_each_batch_dim)
            """


    def check_DReG_gradients1(self, batch_size=10, max_grad_norm=1, clip_grad=True):
        if (self.loss_type == "DReG" or self.loss_type == "DReG_kl") and self.k is None:
            self.k = batch_size
        for epoch in range(3):
            self.optimizer.zero_grad()
            x_samples, log_q_x = self.learnt_sampling_dist(batch_size)
            log_p_x = self.target_dist.log_prob(x_samples)
            loss = self.loss(x_samples, log_q_x, log_p_x)
            loss.backward()
            if clip_grad is True:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.learnt_sampling_dist.parameters(), max_grad_norm)
                # torch.nn.utils.clip_grad_value_(self.learnt_sampling_dist.parameters(), 0.001)

            print(torch.sum(
                self.learnt_sampling_dist.flow_blocks[0].AutoregressiveNN.FirstLayer.latent_to_layer.weight_g -
                self.fixed_learnt_sampling_dist.flow_blocks[0].AutoregressiveNN.FirstLayer.latent_to_layer.weight_g))
            self.optimizer.step()
            print(
                torch.sum(self.learnt_sampling_dist.flow_blocks[0].AutoregressiveNN.FirstLayer.latent_to_layer.weight_g -
                          self.fixed_learnt_sampling_dist.flow_blocks[
                              0].AutoregressiveNN.FirstLayer.latent_to_layer.weight_g))


if __name__ == '__main__':
    from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
    from TargetDistributions.MoG import custom_MoG, MoG
    from FittedModels.Models.FlowModel import FlowModel
    from FittedModels.utils import plot_samples
    import matplotlib.pyplot as plt

    dim = 6
    learnt_sampler = FlowModel(x_dim=dim, n_flow_steps=2) #, use_exp=True)
    target = MoG(dim)  # custom_MoG(dim=dim, cov_scaling=0.3)
    tester = debugger(target, learnt_sampler, VanillaImportanceSampling, use_GPU=False,
                                       loss_type="DReG", lr=1e-1)
    samples_fig_before = plot_samples(tester)
    plt.show()
    tester.check_DReG_gradients2()

    #alpha_2_grads_1, alpha_2_grads_2 , kl_DReG_grads_1, kl_DReG_grads_2, kl_grads_1, kl_grads_2 = \
    #    tester.compare_loss_gradients()