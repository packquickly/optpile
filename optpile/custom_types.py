from typing import Protocol

import jax
from jaxtyping import Array, PRNGKeyArray, PyTree


class RandomGenerator(Protocol):
    # Similar to `Callable`, but with properly typed keyword argument and
    # specific argument names. I may change this to an abstract class later,
    # but for now prefer structural typing for this.
    def __call__(
        self, struct: jax.ShapeDtypeStruct, *, key: PRNGKeyArray
    ) -> PyTree[Array]:
        ...
