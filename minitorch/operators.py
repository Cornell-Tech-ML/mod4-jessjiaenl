"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiplies x by y"""
    return x * y


def id(x: float) -> float:
    """Returns x"""
    return x


def add(x: float, y: float) -> float:
    """Add x and y"""
    return x + y


def neg(x: float) -> float:
    """Return -x"""
    return float(-x)


def lt(x: float, y: float) -> float:
    """Returns 1 iff x less than y"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Returns 1 iff x is equal to y"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the maximum of x and y"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Returns 1 iff x and y have a difference within 1e-2"""
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    r"""Return sigmoid of x as: $\frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$"""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Returns x if x >= 0 otherwise return 0"""
    return x if x >= 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Returns the natural log of x"""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Returns e^x"""
    return math.exp(x)


def inv(x: float) -> float:
    """Returns x's reciprocol"""
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log(x) times y"""
    return y / (x + EPS)


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of 1/x times y"""
    return -(1.0 / x**2) * y


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU(x) times y"""
    return y if x >= 0 else 0.0


def sigmoid_back(x: float, y: float) -> float:
    """Computes the derivative of sigmoid times y"""
    return mul(mul(sigmoid(x), add(1, neg(sigmoid(x)))), y)
    # y * (sigmoid(x) * (1-sigmoid(x)))


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(f: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order function that applies f to each element of an iterable"""

    def mapf(L: Iterable[float]) -> Iterable[float]:
        return [f(x) for x in L]

    return mapf


def zipWith(
    f: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order function that combines elements from two iterables using function f"""

    def zipWithf(L1: Iterable[float], L2: Iterable[float]) -> Iterable[float]:
        return [f(x1, x2) for (x1, x2) in zip(L1, L2)]

    return zipWithf


# def reduce(
#     f: Callable[[float, float], float],
# ) -> Callable[[float], Callable[[Iterable[float]], float]]:
#     """Higher-order function that reduces an iterable to a value using function f"""

#     def reducef(id: float) -> Callable[[Iterable[float]], float]:
#         def reducefid(L: Iterable[float]) -> float:
#             acc = id
#             for x in L:
#                 acc = f(acc, x)
#             return acc

#         return reducefid

#     return reducef


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order function that reduces an iterable to a value using function f"""

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


def negList(L: Iterable[float]) -> Iterable[float]:
    """Negate all elements in L using map"""
    return map(neg)(L)


def addLists(L1: Iterable[float], L2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from L1 and L2 using zipWith"""
    return zipWith(add)(L1, L2)


def sum(L: Iterable[float]) -> float:
    """Sum all elements in a list using reduce"""
    # reduceAdd = reduce(lambda x, y: x + y)(0)
    # return reduceAdd(L)
    return reduce(add, 0.0)(L)


def prod(L: Iterable[float]) -> float:
    """Calculate the product of all elements in L using reduce"""
    # reduceMult = reduce(lambda x, y: x * y)(1)
    # return reduceMult(L)
    return reduce(mul, 1.0)(L)
