# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Set jit decice optional arg to True"""
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """Jit the function fn"""
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Call our implementation of CUDA zip on fn"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Call our implementation of CUDA reduce on fn"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Call our implementation of CUDA matmul on a and b"""
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # do we have to deal with stride align?
        # if i < out_size:
        #     if np.array_equal(in_strides, out_strides) and np.array_equal(in_shape, out_shape):
        #         out[i] = fn(in_storage[i])
        #     else:
        #         to_index(i, out_shape, out_index)
        #         broadcast_index(out_index, out_shape, in_shape, in_index)
        #         o = index_to_position(out_index, out_strides)
        #         j = index_to_position(in_index, in_strides)
        #         out[o] = fn(in_storage[j])

        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            out[o] = fn(in_storage[j])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # if i < out_size:
        #     if np.array_equal(a_strides, out_strides) and np.array_equal(a_shape, out_shape) and np.array_equal(a_strides, b_strides) and np.array_equal(a_shape, b_shape):
        #         out[i] = fn(a_storage[i], b_storage[i])
        #     else:
        #         to_index(i, out_shape, out_index)
        #         o = index_to_position(out_index, out_strides)
        #         broadcast_index(out_index, out_shape, a_shape, a_index)
        #         j = index_to_position(a_index, a_strides)
        #         broadcast_index(out_index, out_shape, b_shape, b_index)
        #         k = index_to_position(b_index, b_strides)
        #         out[o] = fn(a_storage[j], b_storage[k])

        if i < out_size:
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            j = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            k = index_to_position(b_index, b_strides)
            out[o] = fn(a_storage[j], b_storage[k])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""A practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # Implement for Task 3.3.
    block_i = cuda.blockIdx.x
    cache[pos] = 0.0
    if i < size:
        cache[pos] = a[i]
    cuda.syncthreads()

    if i < size:  # check this so that block_i < out_size is implied
        pow = 1
        while pow < BLOCK_DIM:
            if pos % (2 * pow) == 0 and pos + pow < BLOCK_DIM:
                # 1st round: pos 0 += pos 1, pos 2 += pos 3, ...
                # 2nd round: pos 0 += pos 2, pos 4 += pos 6
                # 3rd round: pos 0 += pos 4
                cache[pos] += cache[pos + pow]
            cuda.syncthreads()
            pow *= 2

        # final reduce val is stored in first cache block
        if pos == 0:
            out[block_i] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Calls our implementation of CUDA sum"""
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024  # how many threads in a block
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        """
        Imagine doing reduce row on a 2D matrix:
        each out cell represent a col in a
        for each col in a, make the threads do something like sum practice
        we use one block for each col => out[block_id] = out[col] = reduced val for that col
        """

        # init cache vals
        cache[pos] = reduce_value
        # one cache block for each thread

        if out_pos < out_size:  # for each cell in out i.e. each col in a
            to_index(
                out_pos, out_shape, out_index
            )  # out_storage[out_pos] = out_storage[o] = out[col]
            o = index_to_position(out_index, out_strides)

            # increase the index at the dim to be reduced to get an imaginary index to index into a
            out_index[reduce_dim] = (
                out_index[reduce_dim] * BLOCK_DIM + pos
            )  # now out_index[reduce_dim] = row index
            j = index_to_position(out_index, a_strides)  # a_storage[j] = a[row][col]

            # copy the rows of this col into cache
            if out_index[reduce_dim] < a_shape[reduce_dim]:  # row < len(matrix)
                cache[pos] = a_storage[j]
            cuda.syncthreads()

            # do reduce over rows of that col
            if out_index[reduce_dim] < a_shape[reduce_dim]:  # row < len(matrix)
                pow = 1
                while pow < BLOCK_DIM:
                    if pos % (2 * pow) == 0 and pos + pow < BLOCK_DIM:
                        cache[pos] = fn(cache[pos], cache[pos + pow])
                    cuda.syncthreads()
                    pow *= 2
            # store reduced value at out[out_pos] = out[block_id] = out[col]
            if pos == 0:
                out[o] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """A practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y

    acc = 0  # initialize local var to store out[i,j]
    # we want to loop through the shared dimension k (inner loop in the above comment)
    # but since we compute a block at a time, we loop through that by stepping an offset k, k+BLOCK_DIM, k+2*BLOCK_DIM and then loop through each block by k + local_i or local_j
    for k in range(0, size, BLOCK_DIM):
        # for this k, we have multiple blocks, collectively the shared memories across these blocks hold:
        # a[i, k:k+BLOCK_DIM] and b[k:k+BLOCK_DIM, j]
        # note that each block holds a a_shared of shape (BLOCK_DIM, BLOCK_DIM) and a b_shared of shape (BLOCK_DIM, BLOCK_DIM)
        # just that for all shared memories for this k they cover the values a[i, k:k+BLOCK_DIM] and b[k:k+BLOCK_DIM, j]
        if i < size and k + local_j < size:
            a_shared[local_i, local_j] = a[i * size + k + local_j]
        if j < size and k + local_i < size:
            b_shared[local_i, local_j] = b[(k + local_i) * size + j]
        cuda.syncthreads()

        # now we are able to accumulate to out[i, j] by computing a dot product between BLOCK_DIM len vectors
        # we essentially partitioned a whole row of a and a whole col of b into BLOCK_DIM size sub vectors, dot each subvector, and add to accumulation
        for local_k in range(BLOCK_DIM):
            if k + local_k < size:
                acc += a_shared[local_i, local_k] * b_shared[local_k, local_j]

    # after we completely go through the whole row of a and whole col of b (i.e. all "k"), we finish the accumulation of out[i,j]
    if i < size and j < size:
        out[i * size + j] = acc


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Calls our implementation of CUDA mm practice"""
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]

    acc = 0  # for computing the dot produce for position c[i, j]
    for k in range(
        0, a_shape[-1], BLOCK_DIM
    ):  # Move across shared dimension by block dim size of steps, and take smaller steps within using pi pj in order to move a block of a/b into shared memeory at a time
        if (
            i < a_shape[-2] and k + pj < a_shape[-1]
        ):  # Copy into shared memory for a matrix
            a_shared[pi, pj] = a_storage[
                batch * a_batch_stride + i * a_strides[-2] + (k + pj) * a_strides[-1]
            ]
        if (
            j < b_shape[-1] and k + pi < b_shape[-2]
        ):  # Copy into shared memory for b matrix
            b_shared[pi, pj] = b_storage[
                batch * b_batch_stride + (k + pi) * b_strides[-2] + j * b_strides[-1]
            ]
        cuda.syncthreads()

        # Accumulate to c[i, j] by computing a partial dot product on a subvector of a of len BLOCK_DIM and a subvector of b of len BLOCK_DIM
        for pk in range(BLOCK_DIM):
            if k + pk < a_shape[-1]:  # a_shape[-1] == b_shape[-2]
                acc += a_shared[pi, pk] * b_shared[pk, pj]

    # Now we've accumulated all partitions of a row in a and a col in b, can store the computed dot produce for position c[i, j] (skipping previous dims with batch * out_strides[0])
    if i < a_shape[-2] and j < b_shape[-1]:
        out[batch * out_strides[0] + i * out_strides[-2] + j * out_strides[-1]] = acc


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
