from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    val_plus = [vals[i] for i in range(len(vals))]
    val_plus[arg] += epsilon
    val_minus = [vals[i] for i in range(len(vals))]
    val_minus[arg] -= epsilon
    return (f(*val_plus) - f(*val_minus)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Stores x as the derivative for this variable (leaf)"""
        ...

    @property
    def unique_id(self) -> int:
        """Return unique ID for this variable"""
        ...

    def is_leaf(self) -> bool:
        """Return true iff this variable is a leaf"""
        ...

    def is_constant(self) -> bool:
        """Return true iff this variable is constant"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Return the list of parent variables (immediate left boxes) of this variable"""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to the previously applied function

        Args:
        ----
            d_output: gradient that was passed on

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # right most, then vars left of it in any order, then vars left of those in any order,...
    # DFS: finish order = reverse topological order
    # acyclic: won't visit node between first time visiting it and finishing it

    finish_order = []
    visited = set()

    def dfs(var: Variable) -> None:
        if var.unique_id in visited or var.is_constant():
            return

        if not var.is_leaf():
            for parent in var.parents:
                if not parent.is_constant():
                    dfs(parent)

        visited.add(var.unique_id)
        finish_order.append(var)
        return

    dfs(variable)
    return finish_order[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph to compute derivatives for the leaf nodes.

    Args:
    ----
        variable: The right-most variable.
        deriv: The derivative we want to propagate backward to the leaves.

    Returns:
    -------
        None: Updates the derivative values of each leaf through accumulate_derivative`.

    """
    topo_order = topological_sort(variable)  # step 0: topological order
    deriv_dict = {}  # step 1: store intermedeate derivs of vars (i.e. d_outputs)

    # initialize d_output for var
    deriv_dict[variable.unique_id] = deriv

    for var in topo_order:
        deriv = deriv_dict[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                deriv_dict.setdefault(v.unique_id, 0.0)
                deriv_dict[v.unique_id] = deriv_dict[v.unique_id] + d

        # # do chain rule passing d_output (deriv) through var's parents, add to intermediate derivs
        # # since vars are processed in topological order, by the time we need to do chain rule on it
        # # we would have already have d_outputs = accumulated derivs from all of its parents
        # if not var.is_leaf():
        #     for parent, parent_deriv in var.chain_rule(deriv_dict[var.unique_id]):
        #         # note that multivariate chain rule: h'(f(x), g(x)) = h'(f)f'(x) + h'(g)g'(x)
        #         deriv_dict[parent.unique_id] = (
        #             deriv_dict.get(parent.unique_id, 0) + parent_deriv
        #         )

        # # at leaf, we've aready calculated the derivative wrt of it, so store it with 'accumulate_derivative'
        # else:
        #     var.accumulate_derivative(deriv_dict[var.unique_id])


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the saved tensors from the forward pass."""
        return self.saved_values
