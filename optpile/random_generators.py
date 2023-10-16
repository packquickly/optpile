from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray, PyTree

from .misc import default_floating_dtype


class RandomGenerator(Protocol):
    # Similar to `Callable`, but with properly typed keyword argument and
    # specific argument names. I may change this to an abstract class later,
    # but for now prefer structural typing for this.
    def __call__(
        self, struct: jax.ShapeDtypeStruct, *, key: PRNGKeyArray
    ) -> PyTree[Array]:
        ...


class NormalRandomGenerator(eqx.Module):
    variance: float

    def __call__(self, struct: jax.ShapeDtypeStruct, *, key: PRNGKeyArray):
        leaves, treedef = jtu.tree_flatten(struct)
        rand_leaves = []
        for leaf in leaves:
            key, _ = jr.split(key)
            # TODO(packquickly): support type conversion when
            # `leaf.dtype` is a jax integer type.
            std_normal_leaf = jr.normal(key, leaf.shape, leaf.dtype)
            rand_leaves.append(jnp.sqrt(self.variance) * std_normal_leaf)

        if len(rand_leaves) == 0:
            return jnp.array(0, default_floating_dtype())
        else:
            return treedef.unflatten(rand_leaves)
