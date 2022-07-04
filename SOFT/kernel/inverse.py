import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class _InverseKernel(Function):
    @staticmethod
    def forward(ctx, mat):
        assert mat.dim() == 4 and mat.is_cuda
        if not mat.is_contiguous():
            mat = mat.contiguous()
        with torch.no_grad():
            output = torch.inverse(mat)
        ctx.save_for_backward(mat, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mat, inverse = ctx.saved_tensors
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        with torch.cuda.device_of(mat):
            if ctx.needs_input_grad[0]:
                grad_mat = - (inverse.transpose(-1, -2) @ grad_output @ inverse.transpose(-1, -2))
        return grad_mat, None, None, None, None


def inverse_kernel(input_mat):
    assert input_mat.dim() == 4
    if input_mat.is_cuda:
        out = _InverseKernel.apply(input_mat)
    else:
        raise NotImplementedError
    return out

class _NewtonInverseKernel(Function):
    @staticmethod
    def forward(ctx, mat, iter=20):
        assert mat.dim() == 4 and mat.is_cuda
        if not mat.is_contiguous():
            mat = mat.contiguous()
        with torch.no_grad():
            output = newton_inv(mat, iter)
        ctx.save_for_backward(mat, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mat, inverse = ctx.saved_tensors
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        with torch.cuda.device_of(mat):
            if ctx.needs_input_grad[0]:
                grad_mat = - (inverse.transpose(-1, -2) @ grad_output @ inverse.transpose(-1, -2))
        return grad_mat, None, None, None, None
    
def newton_inv(mat, iter=20):
    P = mat
    I = torch.eye(mat.size(-1), device=mat.device)
    alpha = 0.9 * 2 / torch.max(torch.sum(mat, dim=-1), -1)[0] ** 2
    P = P + 0.01 * I
    V = alpha[..., None, None].broadcast_to(P.shape) * P

    for i in range(iter):
        V = 2 * V - V @ P @ V
    return V


def newton_inverse_kernel(input_mat, iter=20):
    assert input_mat.dim() == 4
    if input_mat.is_cuda:
        out = _NewtonInverseKernel.apply(input_mat, iter)
    else:
        raise NotImplementedError
    return out

class MyInverse(nn.Module):
    def __init__(self):
        super(MyInverse, self).__init__()

    def forward(self, input_mat):
        return inverse_kernel(input_mat)
