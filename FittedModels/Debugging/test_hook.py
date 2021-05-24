import numpy as np
import torch
import torch.nn as nn

def create_hook(name):
    def hook(grad):
        print(name, grad)
    return hook
if __name__ == '__main__':
    x = torch.tensor([1.0, np.nan])
    k = nn.Parameter(0.01*torch.randn(1))

    between = k.repeat(2) # need the extra dimensions in order to fix the NaN gradient
    between.register_hook(create_hook('between'))

    y = between*x

    masked = y[:-1]

    loss = masked.sum()
    print(f'loss: {loss}')

    loss.z_to_x()