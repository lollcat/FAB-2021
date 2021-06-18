

[torch.autograd.grad(loss_1, param, retain_graph=True) for param in self.learnt_sampling_dist.parameters()]
"""
torch.autograd.grad(output, parameter, retain_graph=True)
****************  Parameters ****************************
self.learnt_sampling_dist.scaling_factor
self.learnt_sampling_dist.flow_blocks[0].AutoregressiveNN.FirstLayer.latent_to_layer.weight_g
self.learnt_sampling_dist.flow_blocks[0].MLP.output_layer.weight

****************  Outputs ****************************
torch.sum(x_samples)
torch.sum(log_q)

"""
