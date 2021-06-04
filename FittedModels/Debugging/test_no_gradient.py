import torch

"""
Let's say we have z = g(phi), and then w = func(z, phi)
and we want to calculate the gradient 
dw/dz * dz/dphi

This means in w = func(z, phi), we want to block the gradient at phi, so that it isn't calculated when we backward the 
loss, but we still want to be able to backward through z to phi
"""

def g(phi):
    # return z, which is a function of phi
    z = torch.randn_like(phi)*phi
    return z

def f(z, phi):
    phi.requires_grad = False
    w = phi.T@z
    phi.requires_grad = True
    return w

if __name__ == '__main__':
    phi = torch.randn((3,1))
    z = g(phi)
    w = f(z, phi)
    print(torch.autograd.grad(w, phi))