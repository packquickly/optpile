from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, PRNGKeyArray, PyTree, ScalarLike

from .base import AbstractLeastSquaresProblem, Difficulty
from .custom_types import RandomGenerator
from .misc import additive_perturbation, get_dim


class SimpleBowl(AbstractLeastSquaresProblem):
    name = "Simple Bowl"
    difficulty = Difficulty.EASY
    minimum: ScalarLike = eqx.static_field(converter=jnp.asarray, default=0.0)
    default_dim: int = 120  # mutable!

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None
    ) -> PyTree[Array]:
        dim = get_dim(options, default=self.default_dim)
        return jnp.ones(dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None
    ) -> PyTree[Array]:
        dim = get_dim(options, default=self.default_dim)
        return jnp.ones(dim)

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        return (args**ω * ω(x).call(jnp.square)).ω
