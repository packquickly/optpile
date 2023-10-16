import abc
from enum import Enum
from typing import Generic, Optional

import equinox as eqx
from equinox import AbstractClassVar, AbstractVar
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar

from ..custom_types import Args, Out, Y
from ..random_generators import RandomGenerator


class Minimum(eqx.Module):
    min: Optional[float]
    argmin: Optional[PyTree]


class Difficulty(Enum):
    EASY = "easy"
    HARD = "hard"


class AbstractTestProblem(eqx.Module, Generic[Out, Y, Args]):
    name: AbstractClassVar[str]
    difficulty: AbstractClassVar[Optional[Difficulty]]
    minimum: AbstractVar[Minimum]
    in_dim: AbstractVar[int]

    @abc.abstractmethod
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray]
    ) -> Y:
        """Returns a reasonable default initialisation for the problem.

        Accepts a `random_generator`, a function which takes a PyTree of
        `jax.ShapeDtypeStruct` and `PRNGKey` and returns a random PyTree.
        If a random generator is passed, then the output random pytree
        is used to perturb the initialisation. This is useful for testing
        robustness of an optimiser with respect to its init.

        **Arguments:**

        - `random_generator`: A function with argument `struct` taking a
            `jax.ShapeDtypeStruct` and a keyword argument `key` taking a JAX PRNGKey
            and returning a random array with the same shape and dtype as `struct`.
            Optional.
        - `options`: Any other problem-dependent options.
        - `key`: A JAX PRNGKey. Optional, keyword only.
        """
        ...

    @abc.abstractmethod
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray]
    ) -> Optional[Args]:
        """Returns a list of any arguments the problem may use. This can include
        specificwparameters used in test functions. Each set of args in the list
        defines a separate test problem.

        Accepts a `random_generator`, a function which takes a PyTree of
        `jax.ShapeDtypeStruct` and `PRNGKey` and returns a random PyTree.
        If a random generator is passed, then the output random pytree
        is used to perturb the initialisation. This is useful for generating
        similar problems, but is not applicable to every problem.

        **Arguments:**

        - `random_generator`: A function with argument `struct` taking a
            `jax.ShapeDtypeStruct` and a keyword argument `key` taking a JAX PRNGKey
            and returning a random array with the same shape and dtype as `struct`.
            Optional.
        - `options`: Any other problem-dependent options.
        - `key`: A JAX PRNGKey. Optional, keyword only.
        """
        ...

    @abc.abstractmethod
    def fn(self, y: Y, args: Args) -> Out:
        """The actual test function. Accepts a pytree of arrays `x`
        and a pytree of parameters `args`.
        """
        ...


class AbstractMinimisationProblem(AbstractTestProblem[Scalar, PyTree[Array], PyTree]):
    ...


class AbstractLeastSquaresProblem(
    AbstractTestProblem[PyTree[Array], PyTree[Array], PyTree]
):
    out_dim: AbstractVar[int]
