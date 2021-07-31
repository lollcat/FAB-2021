import torch
from FittedModels.utils.model_utils import sample_and_log_prob_big_batch_drop_nans

class MixtureManager(object):
    def __init__(self, list_of_testers):
        self.list_of_testers = list_of_testers
        self.n_mixes = len(list_of_testers)
        self.device = list_of_testers[0].device

    def to(self, device):
        for tester in self.list_of_testers:
            tester.to(device)
        self.device= device

    def log_prob(self, x_samples):
        log_prob_q_x_list = []
        for tester in self.list_of_testers:
            log_probs_q_x = tester.learnt_sampling_dist.log_prob(x_samples)
            log_prob_q_x_list.append(log_probs_q_x)
        log_prob_q_x_stack = torch.stack(log_prob_q_x_list)
        mix_log_probs = torch.logsumexp(log_prob_q_x_stack, dim=0) - torch.log(torch.tensor([self.n_mixes]))
        return mix_log_probs


    @torch.no_grad
    def sample_and_log_prob(self, batch_size, inner_batch_size=None):
        x_sample_list = []
        log_probs_dict = dict([(i, [None] * self.n_mixes) for i in range(len(self.list_of_testers))])
        with torch.no_grad():
            for i, tester in enumerate(self.list_of_testers):
                if inner_batch_size == None:
                    x_samples, log_probs = tester.learnt_sampling_dist(batch_size)
                else:
                    x_samples, log_probs = sample_and_log_prob_big_batch_drop_nans(tester, batch_size,
                                                                               inner_batch_size)
                x_sample_list.append(x_samples)
                log_probs_dict[i][i] = log_probs