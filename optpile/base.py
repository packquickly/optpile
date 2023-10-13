import abc
from enum import Enum
from typing import Generic, Optional
from typing_extensions import TypeVar

import equinox as eqx
from equinox import AbstractClassVar, AbstractVar
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar

from .custom_types import RandomGenerator


Out = TypeVar("Out")


class Minimum(eqx.Module):
    min: Optional[float]
    argmin: Optional[PyTree]


class Difficulty(Enum):
    EASY = "easy"
    HARD = "hard"


class AbstractTestProblem(eqx.Module, Generic[Out]):
    in_dim: AbstractVar[int]
    out_dim: AbstractVar[int]
    name: AbstractClassVar[str]
    difficulty: AbstractClassVar[Optional[Difficulty]]

    @abc.abstractmethod
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray]
    ) -> PyTree[Array]:
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
        - `options`: Any other options the user may want to pass. Many solvers support
            passing a specific dimension with `options["dimension"]`.
            Optional.
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
    ) -> list[Optional[PyTree[Array]]]:
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
        - `options`: Any other options the user may want to pass. Many solvers support
            passing a specific dimension with `options["dimension"]`.
            A custom init can be passed through options using `options["init"]`.
            Optional.
        - `key`: A JAX PRNGKey. Optional, keyword only.
        """
        ...

    @abc.abstractmethod
    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> Out:
        """The actual test function. Accepts a pytree of arrays `x`
        and a pytree of parameters `args`.
        """
        ...


class AbstractMinimisationProblem(AbstractTestProblem[Scalar]):
    minimum: AbstractVar[Minimum]


class AbstractLeastSquaresProblem(AbstractTestProblem[PyTree[Array]]):
    minimum: AbstractVar[Minimum]
