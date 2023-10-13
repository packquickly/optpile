import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree

import optpile as op


def sum_squares(tree: PyTree[Array]):
    mapped = jtu.tree_map(lambda x: jnp.sum(jnp.square(x)), tree)
    return jtu.tree_reduce(lambda x, y: x + y, mapped)


tuos_problems = [
    op.Bard(),
    op.Beale(),
    op.BiggsEXP6(),
    op.BoxThreeDim(),
    op.BrownAlmostLinear(),
    op.BrownBadlyScaled(),
    op.BrownDennis(),
    op.BroydenBanded(),
    op.BroydenTridiagonal(),
    op.Chebyquad(),
    op.CoupledRosenbrock(),
    op.DecoupledRosenbrock(),
    op.DiscreteBoundary(),
    op.DiscreteIntegral(),
    op.ExtendedPowellSingular(),
    op.FreudensteinRoth(),
    op.Gaussian(),
    op.GULFRnD(),
    op.Gaussian(),
    op.HelicalValley(),
    op.JennrichSampson(),
    op.KowalikOsborne(),
    op.LinearFullRank(),
    op.LinearRank1(),
    op.LinearRank1Zero(),
    op.Meyer(),
    op.Osborne1(),
    op.Osborne2(),
    op.PenaltyFunction1(),
    op.PenaltyFunction2(),
    op.PowellBadlyScaled(),
    op.PowellSingular(),
    op.SimpleBowl(),
    op.Trigonometric(),
    op.VariablyDimensioned(),
    op.Watson(),
    op.Wood(),
]


hard_problems = [op.CoatingThickness()]
