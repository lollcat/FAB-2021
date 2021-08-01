import torch
import math

def sample_and_log_prob_big_batch_drop_nans(tester, n_samples, batch_size):
  assert n_samples % batch_size == 0
  n_batches = int(n_samples / batch_size)
  samples = []
  log_q = []
  for i in range(n_batches):
      samples_batch, log_q_batch = tester.learnt_sampling_dist(batch_size)
      nice_indices = (~(torch.isinf(log_q_batch) | torch.isnan(log_q_batch)))
      samples_batch = samples_batch[nice_indices]
      nice_indices = nice_indices.cpu().detach()
      log_q_batch = log_q_batch.cpu().detach()[nice_indices]
      samples.append(samples_batch.cpu().detach())
      log_q.append(log_q_batch)
  samples = torch.cat(samples, dim=0)
  log_q = torch.cat(log_q, dim=0)
  return samples, log_q

def sample_and_log_w_big_batch_drop_nans(AIS_train, n_samples, batch_size, AIS = False):
  assert n_samples % batch_size == 0
  n_batches = int(n_samples / batch_size)
  samples = []
  log_w = []
  for i in range(n_batches):
      if not AIS:
          samples_batch, log_q_batch = AIS_train.learnt_sampling_dist(batch_size)
          log_p_batch = AIS_train.target_dist.log_prob(samples_batch)
          log_w_batch = log_p_batch - log_q_batch
      else:
          samples_batch, log_w_batch = AIS_train.run(batch_size)
      nice_indices = (~(torch.isinf(log_w_batch) | torch.isnan(log_w_batch))).cpu().detach()
      samples_batch = samples_batch.detach().cpu()[nice_indices]
      log_w_batch = log_w_batch.detach().cpu()[nice_indices]
      samples.append(samples_batch.cpu().detach())
      log_w.append(log_w_batch)
  samples = torch.cat(samples, dim=0)
  log_w = torch.cat(log_w, dim=0)
  return samples, log_w

def log_prob_big_batch(log_prob_func, x_samples, batch_size, device="cuda"):
    n_samples = x_samples.shape[0]
    n_batches = int(math.ceil(n_samples / batch_size))
    log_p = []
    for i in range(n_batches):
        if i != n_batches - 1:
            x_sample = x_samples[batch_size*i:batch_size*(i+1)].to(device)
        else:
            x_sample = x_samples[batch_size*i:].to(device)
        log_p_batch = log_prob_func(x_sample).cpu().detach()
        log_p.append(log_p_batch)
    log_p = torch.cat(log_p, dim=0)
    return log_p


if __name__ == '__main__':
    from AIS_train.train_AIS import AIS_trainer
    from FittedModels.Models.FlowModel import FlowModel
    from TargetDistributions.DoubleWell import ManyWellEnergy
    dim = 4
    target = ManyWellEnergy(dim=dim)
    learnt_sampler = FlowModel(x_dim=dim)
    tester = AIS_trainer(target, learnt_sampler)
    print(sample_and_log_w_big_batch_drop_nans(tester, 100, 10, AIS=False))
    print(sample_and_log_w_big_batch_drop_nans(tester, 100, 10, AIS=True))


