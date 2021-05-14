import torch


def check_gradients(named_parameters):
    names = []
    is_nan = []
    for n, p in named_parameters:
        if p.requires_grad:
            grads = p.grad
            names.append(n)
            if grads == None:
                is_nan.append(None)
            else:
                is_nan.append(torch.sum(torch.isnan(grads)) > 0)
    return names, is_nan