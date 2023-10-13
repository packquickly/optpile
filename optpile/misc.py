from typing import Literal, Optional, overload, Union

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, PyTree


#
# A number of misc helper functions, mostly for manipulating PyTrees easily.
# Most of these are taken from Optimistix:
# (https://github.com/patrick-kidger/optimistix)
#


def additive_perturbation(fn):
    """This decorator wraps either an `init` or `args` method and returns
    an additve perturbation as specified by the `random_generator`.

    This removes some boilerplate the occurs every time `init`/`args` wants
    to add a random perturbation to the init. Instead they can just worry
    about returning the deterministic values
    """

    def wrapper(random_generator, options, *, key):
        tree = fn(random_generator, options, key=key)
        if random_generator is not None:
            if key is None:
                raise ValueError(
                    "Please pass a JAX `PRNGKey` via the keyword"
                    "argument `key`when using a `random_genrator`"
                )
            perturbation = random_generator(tree, key=key)
        else:
            perturbation = tree_full_like(tree, 0.0)
        return tree + perturbation

    return wrapper


def get_dim(options: Optional[dict], *, default: int):
    if options is None:
        dim = default
    else:
        try:
            breakpoint()
            dim = options["dimension"]
        except KeyError:
            dim = default
    return dim


def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32


@overload
def tree_full_like(
    struct: PyTree[Union[Array, jax.ShapeDtypeStruct]],
    fill_value: ArrayLike,
    allow_static: Literal[False] = False,
):
    ...


@overload
def tree_full_like(
    struct: PyTree, fill_value: ArrayLike, allow_static: Literal[True] = True
):

    ...


def tree_full_like(struct: PyTree, fill_value: ArrayLike, allow_static: bool = False):
    """Return a pytree with the same type and shape as the input with values
    `fill_value`.

    If `allow_static=True`, then any non-{array, struct}s are ignored and left alone.
    If `allow_static=False` then any non-{array, struct}s will result in an error.
    """
    fn = lambda x: jnp.full(x.shape, fill_value, x.dtype)
    if isinstance(fill_value, (int, float)):
        if fill_value == 0:
            fn = lambda x: jnp.zeros(x.shape, x.dtype)
        elif fill_value == 1:
            fn = lambda x: jnp.ones(x.shape, x.dtype)
    if allow_static:
        _fn = fn
        fn = (
            lambda x: _fn(x)
            if eqx.is_array(x) or isinstance(x, jax.ShapeDtypeStruct)
            else x
        )
    return jtu.tree_map(fn, struct)