from typing import Tuple

from .tensor import Tensor
from numba import prange
from .tensor_data import index_to_position


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


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
    out = Tensor.zeros(batch, channel, new_height, new_width, kh*kw)
    for b in prange(batch):
        for c in prange(channel):
            for h in prange(new_height):
                for w in prange(new_width):
                    for i in prange(kh*kw):
                        # calculate out[b, c, h, w, i]
                        o = index_to_position(b, c, h, w, i, out.strides)
                        # decompose i to row, col in kernel
                        row = i // kw
                        col = i % kw
                        # place kernel onto correct pos in input to see what kenel[row, col] corresponds to
                        global_row = h * kh + row
                        global_col = w * kw + col
                        out[o] = input[b, c, global_row, global_col]

    return out, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    return

def max(input: Tensor, dim: int) -> Tensor:
    return

def softmax(input: Tensor, dim: int) -> Tensor:
    return

def logsoftmax(input: Tensor, dim: int) -> Tensor:
    return

def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    return

def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    return