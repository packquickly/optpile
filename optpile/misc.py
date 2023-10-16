import functools as ft
import inspect
from typing import cast, Literal, overload, Union

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, Bool, PyTree, Scalar


#
# A number of misc helper functions, mostly for manipulating PyTrees easily.
# Most of these are taken from Optimistix:
# (https://github.com/patrick-kidger/optimistix)
#


def sum_squares(tree: PyTree[Array]):
    mapped = jtu.tree_map(lambda x: jnp.sum(jnp.square(x)), tree)
    return jtu.tree_reduce(lambda x, y: x + y, mapped)


def array_tuple(in_list: list[Union[float, int, bool, Scalar]]) -> tuple[Array]:
    return cast(tuple[Array], tuple(jnp.asarray(x) for x in in_list))


def additive_perturbation(fn):
    """This decorator wraps either an `init` or `args` method and returns
    an additve perturbation as specified by the `random_generator`.

    This removes some boilerplate the occurs every time `init`/`args` wants
    to add a random perturbation to the init. Instead they can just worry
    about returning the deterministic values
    """

    @ft.wraps(fn)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(fn)
        ba = sig.bind(*args, **kwargs)
        ba.apply_defaults()

        random_generator = ba.arguments["random_generator"]
        key = ba.arguments["key"]

        tree = fn(*args, **kwargs)
        if random_generator is not None:
            if key is None:
                raise ValueError(
                    "Please pass a JAX `PRNGKey` via the keyword"
                    "argument `key`when using a `random_genrator`"
                )
            perturbation = random_generator(tree, key=key)
        else:
            perturbation = tree_full_like(tree, jnp.array(0.0))
        return (tree**ω + perturbation**ω).ω

    return wrapper


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


def tree_where(
    pred: Bool[ArrayLike, ""], true: PyTree[ArrayLike], false: PyTree[ArrayLike]
) -> PyTree[Array]:
    """Return the `true` or `false` pytree depending on `pred`."""
    keep = lambda a, b: jnp.where(pred, a, b)
    return jtu.tree_map(keep, true, false)
