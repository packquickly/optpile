from typing import Protocol, Union
from typing_extensions import TypeVar

import jax
from jaxtyping import Array, PRNGKeyArray, PyTree


Out = TypeVar("Out", bound=Union[PyTree[Array], Array])
Y = TypeVar("Y", bound=Union[PyTree[Array], Array])
Args = TypeVar("Args", bound=PyTree[Array])


class RandomGenerator(Protocol):
    # Similar to `Callable`, but with properly typed keyword argument and
    # specific argument names. I may change this to an abstract class later,
    # but for now prefer structural typing for this.
    def __call__(
        self, struct: jax.ShapeDtypeStruct, *, key: PRNGKeyArray
    ) -> PyTree[Array]:
        ...
