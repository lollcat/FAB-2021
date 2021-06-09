import torch


def check_forward_backward_consistency(self, n=100):
    """p(x) generated from forward should be the same as log p(x) for the same samples"""
    """
    log p(x) = log p(z) - log |dx/dz|
    """
    # first let's go forward
    z = self.prior.sample((n,))
    x = z
    prior_prob = self.prior.log_prob(x)
    log_prob = prior_prob.detach().clone()  # use clone otherwise prior prob get's values get changed
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
        print(torch.max(torch.abs(log_det_forward + log_dets_backward[-i - 1])))
