import torch

# These custom ops break the usual assumption that gradients given to
# backward have the same shapes as outputs. They are expected to have
# an extra leading dimension, which batches independent reverse mode
# passes.
class BatchedReverseMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return torch.matmul(x, y)

    @staticmethod
    def backward(ctx, batched_grad):
        x, y = ctx.saved_variables
        return torch.matmul(batched_grad, y.t()), torch.matmul(x.t(), batched_grad)

class BatchedReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.mask = x >= 0
        return torch.relu(x)

    @staticmethod
    def backward(ctx, batched_grad):
        return batched_grad * ctx.mask.type_as(batched_grad).expand_as(batched_grad)
if __name__ == '__main__':
    # https://github.com/pytorch/pytorch/issues/7786
    matmul = BatchedReverseMM.apply
    relu = BatchedReLU.apply

    batch_size, input_size, output_size = 20, 100, 10
    x = torch.randn(batch_size, input_size)
    W = torch.randn(input_size, output_size, requires_grad=True)

    # NOTE: need to use the custom ops here
    output = relu(matmul(x, W))
    jac_elems = output_size * batch_size # it's really the size of one dim of the jacobian
    batch_grad_output = torch.eye(jac_elems, jac_elems).view(jac_elems, *output.shape)

    jacobian, = torch.autograd.grad(output, W, batch_grad_output)
    print(jacobian.shape)