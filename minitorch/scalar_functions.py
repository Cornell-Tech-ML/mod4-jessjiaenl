from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the forward function specified by cls to vals"""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Return the sum of a and b"""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Return the gradient of sum multiplied by accumulated gradient d_output"""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Apply log function to a"""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Return the derivative of log evaluated at value saved in ctx multiplied by accumulated gradient d_output"""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Mul function $f(x,y) = x*y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Return the product of a and b"""
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Return the gradient of product evaluated at inputs stored in ctx multiplied by accumulated gradient d_output"""
        (a, b) = ctx.saved_values
        return operators.mul(b, d_output), operators.mul(a, d_output)


class Inv(ScalarFunction):
    """Inv function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Apply inverse function to a"""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Return the derivative of inverse evaluated at value saved in ctx multiplied by accumulated gradient d_output"""
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Neg function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Apply negation function to a"""
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Return the derivative of negation multiplied by accumulated gradient d_output"""
        return operators.mul(-1, d_output)


class Sigmoid(ScalarFunction):
    r"""Sigmoid function $\frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Apply sigmoid function to a"""
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Return the derivative of sigmoid evaluated at value saved in ctx multiplied by accumulated gradient d_output"""
        (a,) = ctx.saved_values
        return operators.mul(
            operators.mul(
                operators.sigmoid(a),
                operators.add(1, operators.neg(operators.sigmoid(a))),
            ),
            d_output,
        )


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Apply ReLU function to a"""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Return the derivative of ReLU evaluated at value saved in ctx multiplied by accumulated gradient d_output"""
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exp function $f(x) = exp(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Apply exponential function to a"""
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Return the derivative of exponential evaluated at value saved in ctx multiplied by accumulated gradient d_output"""
        (a,) = ctx.saved_values
        return operators.mul(operators.exp(a), d_output)


class LT(ScalarFunction):
    """LT function $f(x,y) = 1 if x < y else 0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Apply LT function to a and b"""
        ctx.save_for_backward(a, b)
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Return the gradient of LT which is 0"""
        return 0.0, 0.0


class EQ(ScalarFunction):
    """EQ function $f(x,y) = 1 if x == y else 0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Apply EQ function to a and b"""
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Return the gradient of EQ which is 0"""
        return 0.0, 0.0
