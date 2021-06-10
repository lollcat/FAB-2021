from FittedModels.train import LearntDistributionManager
import torch
import torch.nn.functional as F
import numpy as np

class debugger(LearntDistributionManager):
    def __init__(self, *args, **kwargs):
        super(debugger, self).__init__(*args, **kwargs)


    def setup_loss_alternative(self, loss_type, alpha=2, k=None, new_lr=None, annealing=False):
        # this shows that we get the same results if we use a copy model instead of fixing parameters
        if loss_type == "kl":
            self.loss = self.KL_loss
            self.alpha = 1
            self.annealing = annealing
        elif loss_type == "DReG":
            self.loss = self.dreg_alpha_divergence_loss
            self.alpha = alpha  # alpha for alpha-divergence
            self.alpha_one_minus_alpha_sign = torch.sign(torch.tensor(self.alpha * (1 - self.alpha)))
            self.k = k  # number of samples that going inside the log sum, if none then put all of them inside
        elif loss_type == "DReG_kl":
            self.loss = self.dreg_kl_loss
            self.alpha = 1
            self.k = k  # number of samples that going inside the log sum, if none then put all of them inside

        elif loss_type == "alpha_MC":  # this does terribly
            self.loss = self.alpha_MC_loss
            self.alpha = alpha  # alpha for alpha-divergence
            self.alpha_one_minus_alpha_sign = torch.sign(torch.tensor(self.alpha * (1 - self.alpha)))
        else:
            raise Exception("loss_type incorrectly specified")
        if new_lr is not None:
            self.optimizer.param_groups[0]["lr"] = new_lr
        if "DReG" in loss_type:
            self.fixed_learnt_sampling_dist = type(self.learnt_sampling_dist)(
                *self.learnt_sampling_dist.class_args, **self.learnt_sampling_dist.class_kwargs) \
                .to(device=self.device)  # for computing d weights dz

    def compare_loss_gradients(self, n_batches=100, batch_size=100):
        self.k = batch_size  # only used in DReG
        self.alpha = 2  # this is only used in alpha div calc, so doesn't effect other losses
        self.alpha_one_minus_alpha_sign = torch.sign(torch.tensor(self.alpha * (1 - self.alpha)))
        self.annealing = False

        alpha_2_grads_1 = []
        alpha_2_grads_2 = []
        kl_DReG_grads_1 = []
        kl_DReG_grads_2 = []
        kl_grads_1 = []
        kl_grads_2 = []

        param_set_1 = lambda: self.learnt_sampling_dist.flow_blocks[0].AutoregressiveNN.FirstLayer.latent_to_layer.weight
        param_set_2 = lambda: self.learnt_sampling_dist.flow_blocks[0].AutoregressiveNN.FinalLayer.layer_to_m.weight

        for i in range(n_batches):
            self.optimizer.zero_grad()
            x_samples, log_q_x = self.learnt_sampling_dist(batch_size)
            log_p_x = self.target_dist.log_prob(x_samples)

            # alpha 2 divergence
            alpha_2_loss = self.dreg_alpha_divergence_loss(x_samples, log_q_x, log_p_x)
            alpha_2_grads_1.append(torch.autograd.grad(alpha_2_loss, param_set_1(), retain_graph=True)[0].cpu().numpy())
            alpha_2_grads_2.append(torch.autograd.grad(alpha_2_loss, param_set_2(), retain_graph=True)[0].cpu().numpy())

            # kl DReG
            kl_DReG_loss = self.dreg_kl_loss(x_samples, log_q_x, log_p_x)
            kl_DReG_grads_1.append(torch.autograd.grad(kl_DReG_loss, param_set_1(), retain_graph=True)[0].cpu().numpy())
            kl_DReG_grads_2.append(torch.autograd.grad(kl_DReG_loss, param_set_2(), retain_graph=True)[0].cpu().numpy())

            # kl
            kl_loss = self.KL_loss(x_samples, log_q_x, log_p_x)
            kl_grads_1.append(torch.autograd.grad(kl_loss, param_set_1(), retain_graph=True)[0].cpu().numpy())
            kl_grads_2.append(torch.autograd.grad(kl_loss, param_set_2(), retain_graph=True)[0].cpu().numpy())

        alpha_2_grads_1 = np.stack(alpha_2_grads_1)
        alpha_2_grads_2 = np.stack(alpha_2_grads_2)
        kl_DReG_grads_1 = np.stack(kl_DReG_grads_1)
        kl_DReG_grads_2 = np.stack(kl_DReG_grads_2)
        kl_grads_1 = np.stack(kl_grads_1)
        kl_grads_2 = np.stack(kl_grads_2)

        return alpha_2_grads_1, alpha_2_grads_2, kl_DReG_grads_1, kl_DReG_grads_2, kl_grads_1, kl_grads_2


    def dreg_alpha_divergence_loss_old(self, x_samples, log_q_x_not_used, log_p_x):
        self.update_fixed_version_of_learnt_distribution()
        log_q_x = self.fixed_learnt_sampling_dist.log_prob(x_samples)
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
        return dreg_loss


    def check_methods_of_grad_through_z(self, batch_size=1000):
        if "DReG" in self.loss_type and self.k is None:
            self.k = batch_size
        # check that gradients flow nicely in DReG
        for i in range(20):
            print(i)
            self.optimizer.zero_grad()
            # torch.autograd.grad(torch.sum(log_q_x), next(self.learnt_sampling_dist.parameters()), retain_graph=True)
            x_samples, log_q_x = self.learnt_sampling_dist(batch_size)
            log_p_x = self.target_dist.log_prob(x_samples)
            loss1 = self.dreg_alpha_divergence_loss(x_samples, log_q_x, log_p_x)
            grad1 = torch.autograd.grad(loss1,
                                        self.learnt_sampling_dist.flow_blocks[0].AutoregressiveNN.FirstLayer.latent_to_layer.weight_g,
                                        retain_graph=True)

            loss2 = self.dreg_alpha_divergence_loss_old(x_samples, log_q_x, log_p_x)
            grad2 = torch.autograd.grad(loss2,
                                        self.learnt_sampling_dist.flow_blocks[0].AutoregressiveNN.FirstLayer.latent_to_layer.weight_g,
                                        retain_graph=True)

            assert torch.sum(grad1[0] != grad2[0])  == 0

            loss1.backward()
            self.optimizer.step()


    def update_fixed_version_of_learnt_distribution(self):
        self.fixed_learnt_sampling_dist.load_state_dict(self.learnt_sampling_dist.state_dict())

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


    def check_DReG_gradients2(self, batch_size=100, max_grad_norm=1, clip_grad=True):
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
    from TargetDistributions.MoG import MoG
    from FittedModels.Models.FlowModel import FlowModel
    from FittedModels.utils.plotting_utils import plot_samples
    import matplotlib.pyplot as plt

    dim = 6
    learnt_sampler = FlowModel(x_dim=dim, n_flow_steps=3) #, use_exp=True)
    target = MoG(dim)  # custom_MoG(dim=dim, cov_scaling=0.3)
    tester = debugger(target, learnt_sampler, VanillaImportanceSampling, use_GPU=True,
                                       loss_type="DReG", lr=1e-4, annealing=False, alpha=2)
    tester.check_methods_of_grad_through_z()
    samples_fig_before = plot_samples(tester)
    plt.show()
    tester.check_DReG_gradients2()

    #alpha_2_grads_1, alpha_2_grads_2 , kl_DReG_grads_1, kl_DReG_grads_2, kl_grads_1, kl_grads_2 = \
    #    tester.compare_loss_gradients()