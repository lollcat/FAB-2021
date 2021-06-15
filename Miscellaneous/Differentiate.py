import torch
from Utils.DebuggingUtils import timer
from TargetDistributions.MoG import MoG
mog = MoG(dim=4)
def y(x):
    #return torch.sum(x**2, dim=-1)
    return mog.log_prob(x)

if __name__ == '__main__':
    x = torch.ones(2000, 4, requires_grad=True)
    out = y(x)
    back = timer("backward")
    torch.autograd.backward(y(x), grad_tensors=torch.ones_like(out))
    back.stop()

    grad = timer("grad")
    grads = torch.autograd.grad(y(x), x, grad_outputs=torch.ones_like(out))
    grad.stop()

    Jac = timer("jac")
    functional_Jac = \
        torch.diagonal(torch.autograd.functional.jacobian(y, x, vectorize=True), dim1=0, dim2=1).T
    Jac.stop()

    #vip = torch.autograd.functional.vjp(y, x, v=torch.ones(50))

    """
    x_short = x[:, 0]
    list_ = timer("list")
    grads_list = torch.autograd.grad(y(x_short), x_short)
    list_.stop()
    """
