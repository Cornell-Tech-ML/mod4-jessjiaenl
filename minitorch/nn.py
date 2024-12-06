from typing import Tuple

from .tensor import Tensor
from .autodiff import Context
from .tensor_functions import Function
from .fast_ops import FastOps
from .operators import max


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off

# 4.3


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height, new_width = height // kh, width // kw

    # if 1D, do input.view(batch, channel, new_width, kw)
    # idea: add the "to be reduced" dimension on right for contiguous stride to lay out correctly

    out = input.contiguous()
    out = out.view(batch, channel, new_height, kh, width)
    out = out.permute(0, 1, 2, 4, 3)
    out = out.contiguous()
    out = out.view(batch, channel, new_height, new_width, kh * kw)

    return out, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    batch, channel, _, _ = input.shape
    input, new_height, new_width = tile(input, kernel)
    out = input.mean(-1)
    return out.view(batch, channel, new_height, new_width)


# 4.4

max_reduce = FastOps.reduce(max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor."""
    return max_reduce(input, dim) == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Reduce max over dimention dim on t1"""
        ctx.save_for_backward(t1, int(dim.item()))
        return max_reduce(t1, int(dim.item()))

    @staticmethod
    def backward(
        ctx: Context, grad_output: Tensor
    ) -> Tuple[Tensor, float]:  # same return type as View
        """Max's backward is argmax, so multiply by grad output, grad for dim is just 0"""
        (t1, dim) = ctx.saved_values
        return (grad_output * argmax(t1, dim), 0.0)


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    expX = input.exp()
    return expX / expX.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    return softmax(input, dim).log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    batch, channel, _, _ = input.shape
    input, new_height, new_width = tile(input, kernel)
    out = max(input, -1)
    return out.view(batch, channel, new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    return
