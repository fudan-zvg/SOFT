import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from SOFT import _C

class _SubtractionGaussianKernel(Function):
    @staticmethod
    def forward(ctx, query, key):
        assert query.dim() == 4 and query.is_cuda and query.size()[-1] == key.size()[-2]
        if not query.is_contiguous():
            query = query.contiguous()
        if not key.is_contiguous():
            key = key.contiguous()
        batch_size, num_head, q_len, input_channels = query.size()
        batch_size, num_head, input_channels, k_len = key.size()
        output = query.new(batch_size, num_head, q_len, k_len)
        _C.GaussianSubtraction_forward(query, key, output)
        ctx.save_for_backward(query, key)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key = ctx.saved_tensors
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_query, grad_key = None, None
        with torch.cuda.device_of(query):
            if ctx.needs_input_grad[0]:
                grad_query = query.new(query.size())
                _C.GaussianSubtraction_backward_query(grad_output, query, key, grad_query)
            if ctx.needs_input_grad[1]:
                grad_key = key.new(key.size())
                # print("python query", query.shape)
                _C.GaussianSubtraction_backward_key(grad_output, query, key, grad_key)
        return grad_query, grad_key, None, None, None, None


class _SubtractionReduceGaussianKernel(Function):
    @staticmethod
    def forward(ctx, query, key):
        assert query.dim() == 4 and query.is_cuda and query.size()[-1] == key.size()[-2]
        if not query.is_contiguous():
            query = query.contiguous()
        if not key.is_contiguous():
            key = key.contiguous()
        batch_size, num_head, q_len, input_channels = query.size()
        batch_size, num_head, input_channels, k_len = key.size()
        output = query.new(batch_size, num_head, q_len, k_len)
        _C.GaussianSubtraction_reduce_forward(query, key, output)
        ctx.save_for_backward(query, key)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key = ctx.saved_tensors
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_query, grad_key = None, None
        with torch.cuda.device_of(query):
            if ctx.needs_input_grad[0]:
                grad_query = query.new(query.size())
                _C.GaussianSubtraction_backward_query(grad_output, query, key, grad_query)
            if ctx.needs_input_grad[1]:
                grad_key = key.new(key.size())
                # print("python query", query.shape)
                _C.GaussianSubtraction_backward_key(grad_output, query, key, grad_key)
        return grad_query, grad_key, None, None, None, None


def subtraction_gaussian_kernel(input1, input2):
    assert input1.dim() == 4 and input2.dim() == 4 and input1.size()[-1] == input2.size()[-2]
    if input1.is_cuda:
        out = _SubtractionGaussianKernel.apply(input1, input2)
    else:
        raise NotImplementedError
    return out

def subtraction_reduce_gaussian_kernel(input1, input2):
    assert input1.dim() == 4 and input2.dim() == 4 and input1.size()[-1] == input2.size()[-2]
    if input1.is_cuda:
        out = _SubtractionReduceGaussianKernel.apply(input1, input2)
    else:
        raise NotImplementedError
    return out


class SubtractionGaussian(nn.Module):
    def __init__(self):
        super(SubtractionGaussian, self).__init__()

    def forward(self, input1, input2):
        return subtraction_gaussian_kernel(input1, input2)

class SubtractionReduceGaussian(nn.Module):
    def __init__(self):
        super(SubtractionReduceGaussian, self).__init__()

    def forward(self, input1, input2):
        return subtraction_reduce_gaussian_kernel(input1, input2)
