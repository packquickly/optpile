from typing import ClassVar, Optional

import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.random as jr
from equinox.internal import ω
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar

from .base import AbstractLeastSquaresProblem, Difficulty, Minimum
from .custom_types import RandomGenerator
from .misc import (
    additive_perturbation,
    array_tuple,
    default_floating_dtype,
    fixed_dim,
    get_dim,
)


#
# These problems are pulled from a handful of test problem banks,
# which are abbreviated with the shorthands:
# TUOS: "Testing Unconstrained Optimization Software" by Moré, Garbow, and Hillstrom
# MINPACK2: "The MINPACK-2 Test Problem Collection" by Averick, Carter, Moré, and Xue.
# WIKI: Wikipedia
# UOTF: "An Unconstrained Optimization Test Functions Collection" by Andrei
# LSBF: "A Literature Survey of Benchmark Functions for Global Optimization Problems"
# by Jamil and Yan
# NESS: "A Numerical Evaluation of Several Stochastic Algorithms on Selected
# Continuous Global Optimization Test Problems."
#
# TUOS is included in its entirety, this is because TUOS provides both initialisation
# and minima for each of its test problems. UOTF provides only initialisations while
# LSBF and NESS provide only minima. Not having access to the true minima makes it
# difficult to implement accurate tests (in the programming sense) for each problem.
# However, not having good inits requires either an arbitrary init (making the problem
# potentially easier or harder) or an requires a more intensive procedure and
# heuristics to choose a good init for local optimisation.
#
# I (packquickly) implement a mix of both of these, understanding their flaws. While
# I do not intend to implement anything incorrectly, in the case where it happens
# using UOTF, we accept that we are testing some optimisation problem, it just may
# not be the original one laid out in the paper if there is a mistake (which hopefully
# the community will catch and report!) When implmenting from LSBF or NESS, the current
# approch is to choose a randomized point near the minimum, with the intention to
# do a proper search for reasonably difficult inits at a later point.
#
# Lastly, the problems in MINPACK2 tend to compare mathematical models to actual
# data, which we do not have. The methodology for implementing these is to
# generate data from the mathematical model using some set of parameters, and
# treat this as the real data. This gives two methods of evaluating the result
# of the optimisation procedure:
# 1. Compare the output of the fit model to the output of the data-generating model.
# 2. Compare the parameters in the fit model to the true parameters in the data-
#    generting model.
# I (packquickly) opt for the first approach to remain consistent with MINPACK2.
#
# The MINPACK2 problems are typically more involved, more niche, and more difficult
# to implement than the others, so there aren't many of them. However, they each add
# a lot of value for testing optimisers in more realistic use-cases.
#


class SimpleBowl(AbstractLeastSquaresProblem):
    in_dim: int
    out_dim: int
    name: ClassVar[str] = "Simple Bowl"
    difficulty: ClassVar[Difficulty] = Difficulty.EASY
    minimum: Minimum

    def __init__(self, in_dim: int = 120):
        # 120 is an arbitrary default
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.minimum = Minimum(0.0, jnp.zeros(self.in_dim))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        dim = get_dim(options, default=self.in_dim)
        return jnp.ones(dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        dim = get_dim(options, default=self.in_dim)
        return [jnp.ones(dim)]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        return (args**ω * ω(x).call(jnp.square)).ω


# TUOS (21)
# WIKI
class DecoupledRosenbrock(AbstractLeastSquaresProblem):
    in_dim: int
    out_dim: int
    name: ClassVar[str] = "Decoupled Rosenbrock"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum

    def __init__(self, in_dim: int = 2):
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.minimum = Minimum(0.0, jnp.ones(in_dim))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        dim = get_dim(options, default=self.in_dim)
        if dim % 2 != 0:
            raise ValueError(
                "Dimension must be divisible by 2 for decoupled" "Rosenbrock."
            )
        index = jnp.arange(1, dim + 1)
        return jnp.where(index % 2 == 0, 1, -1.2)  # from TUOS

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [None]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        # Rosenbrock doesn't translate well to PyTrees
        y, _ = jfu.ravel_pytree(x)
        index = jnp.arange(1, y.size + 1)
        jnp.where(index % 2 == 0, y, y**2)
        # Remember that all these values will be squared, hence why
        # this looks different than what is on Wikipedia.
        diffs_y = 10 * (y[:-1] - y[1:])
        diffs_1 = jnp.where(index % 2 != 0, (1 - y), 0.0)
        return (diffs_y, diffs_1)


class CoupledRosenbrock(AbstractLeastSquaresProblem):
    in_dim: int
    out_dim: int
    name: ClassVar[str] = "Coupled Rosenbrock"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum

    def __init__(self, in_dim: int = 99):
        # arbitrary default `in_dim`
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.minimum = Minimum(0.0, jnp.ones(in_dim))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        dim = get_dim(options, default=self.in_dim)
        index = jnp.arange(1, dim + 1)
        return jnp.where(index % 2 == 0, 1, -1.2)

    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [None]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y, _ = jfu.ravel_pytree(x)
        diffs_y = 10 * (y[1:] - y[:-1] ** 2)
        diffs_1 = 1 - y
        return (diffs_y, diffs_1)


# TUOS (2)
class FreudensteinRoth(AbstractLeastSquaresProblem):
    in_dim: ClassVar[int] = 2
    out_dim: ClassVar[int] = 2
    name: ClassVar[str] = "Freudenstein and Roth function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, (5.0, 4.0))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        fixed_dim(options, self.name)
        return array_tuple([0.5, -2])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([13.0, 5.0, 2.0, 29.0, 1.0, 14.0])]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        x1, x2 = x
        c1, c2, c3, c4, c5, c6 = args
        f1 = -c1 + x1 + ((c2 - x2) * x2 - c3) * x2
        f2 = -c4 + x1 + ((x2 + c5) * x2 - c6) * x2
        return (f1, f2)


# TUOS (3)
class PowellBadlyScaled(AbstractLeastSquaresProblem):
    in_dim: ClassVar[int] = 2
    out_dim: ClassVar[int] = 2
    name: ClassVar[str] = "Powell's badly scaled function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0, (1.098 * 1e-5, 9.106))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        fixed_dim(options, self.name)
        return array_tuple([0.0, 1.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([1e4, 1.0, 1.0001])]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        x1, x2 = x
        c1, c2, c3 = args
        f1 = c1 * x1 * x2 - c2
        f2 = jnp.exp(-x1) + jnp.exp(-x2) - c3
        return (f1, f2)


# TUOS (4)
class BrownBadlyScaled(AbstractLeastSquaresProblem):
    in_dim: ClassVar[int] = 2
    out_dim: ClassVar[int] = 3
    name: ClassVar[str] = "Brown's badly scaled function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, (1e6, 2e-6))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        fixed_dim(options, self.name)
        return array_tuple([1.0, 1.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([1e6, 2, 1e-6, 2])]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        x1, x2 = x
        c1, c2, c3, c4 = args
        f1 = x1 - c1
        f2 = x2 - c2 * c3
        f3 = x1 * x2 - c4
        return (f1, f2, f3)


# TUOS (5)
class Beale(AbstractLeastSquaresProblem):
    in_dim: ClassVar[int] = 2
    out_dim: ClassVar[int] = 3
    name: ClassVar[str] = "Beale's function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, (3, 0.5))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        fixed_dim(options, self.name)
        return array_tuple([1.0, 1.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        i = jnp.arange(1, 4)
        y_i = jnp.array([1.5, 2.25, 2.625])
        return [(i, y_i)]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        x1, x2 = x
        i, y_i = args
        return y_i - x1 * (1 - x2**i)


# TUOS (6)
class JennrichSampson(AbstractLeastSquaresProblem):
    in_dim: ClassVar[int] = 2
    out_dim: int
    name: ClassVar[str] = "Jennrich and Sampson function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum

    def __init__(self, out_dim: int = 10):
        self.out_dim = out_dim
        if self.out_dim == 10:
            self.minimum = Minimum(124.362, (0.2578, 0.2578))
        else:
            self.minimum = Minimum(None, None)

    def __check_init__(self):
        if self.out_dim < self.in_dim:
            raise ValueError("`out_dim` must be greater than `in_dim`")

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        fixed_dim(options, self.name)
        return array_tuple([0.3, 0.4])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        c1 = jnp.array(2.0)
        c2 = jnp.array(2.0)
        i = jnp.arange(1, self.out_dim + 1)
        return [(c1, c2, i)]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        x1, x2 = x
        c1, c2, i = args
        return 2 + 2 * i - (jnp.exp(i * x1) + jnp.exp(i * x2))


# TUOS (7)
class HelicalValley(AbstractLeastSquaresProblem):
    in_dim: ClassVar[int] = 3
    out_dim: ClassVar[int] = 3
    name: ClassVar[str] = "Helical valley function (decoupled)"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(
        0.0,
        (
            1.0,
            0.0,
            0.0,
        ),
    )

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return array_tuple([-1.0, 0.0, 0.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([10.0, 1.0, 0.5])]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        x1, x2, x3 = x
        c1, c2, c3 = args

        info = jnp.finfo(jnp.asarray(x1).dtype)
        x1_nonzero = jnp.abs(x1) > info.eps
        jnp.where(x1_nonzero, x1, 1)
        x2x1ratio = jnp.where(x1_nonzero, x2 / x1, info.max)
        arctan_val = 1 / (2 * jnp.pi) * jnp.arctan(x2x1ratio)
        theta = jnp.where(x1 > 0, arctan_val, arctan_val + c3)

        f1 = c1 * (x3 - c1 * theta)
        f2 = c1 * (jnp.sqrt(x1**2 + x2**2) - c2)
        f3 = x3
        return (f1, f2, f3)


# TUOS (8)
class Bard(AbstractLeastSquaresProblem):
    in_dim: ClassVar[int] = 3
    out_dim: ClassVar[int] = 15
    name: ClassVar[str] = "Bard function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(8.21487 * 1e-3, None)
    # Another minimum at 17.4286. Not sure if it's wise to
    # incorporate both or not.

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        fixed_dim(options, self.name)
        return array_tuple([1.0, 1.0, 1.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        u_i = jnp.arange(1, 16)
        v_i = 16 - u_i
        w_i = jnp.minimum(u_i, v_i)
        # fmt: off
        y_i = jnp.array([
            0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39,
            0.37, 0.58, 0.73, 0.96, 1.34, 2.10, 4.39
        ])
        # fmt: on
        return [(u_i, v_i, w_i, y_i)]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        x1, x2, x3 = x
        u_i, v_i, w_i, y_i = args
        return y_i - (x1 + u_i / (v_i * x2 + w_i * x3))


# TUOS (9)
class Gaussian(AbstractLeastSquaresProblem):
    in_dim: ClassVar[int] = 3
    out_dim: ClassVar[int] = 15
    name: ClassVar[str] = "Gaussian function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(1.12798 * 1e-8, None)

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        fixed_dim(options, self.name)
        return array_tuple([0.4, 1.0, 0.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        i = jnp.arange(1, 16)
        t_i = (8 - i) / 2
        # fmt: off
        y_i = jnp.array([
                0.0009, 0.0044, 0.0175, 0.0540, 0.1295,
                0.2420, 0.3521, 0.3989, 0.3521, 0.2420,
                0.1295, 0.0540, 0.0175, 0.0044, 0.0009,
            ])
        # fmt: on
        return [(t_i, y_i)]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        x1, x2, x3 = x
        t_i, y_i = args
        return x1 * jnp.exp((-x2 * (t_i - x3) ** 2) / 2) - y_i


# TUOS (10)
class Meyer(AbstractLeastSquaresProblem):
    in_dim: ClassVar[int] = 3
    out_dim: ClassVar[int] = 16
    name: ClassVar[str] = "Meyer function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(87.9458, None)

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        fixed_dim(options, self.name)
        return array_tuple([0.02, 4000, 250])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        i = jnp.arange(1, self.out_dim + 1)
        t_i = 45 + 5 * i
        # fmt: off
        y_i = jnp.array([
            34780, 28610, 23650, 19630, 16370, 13720, 11540, 9744, 
            8261,  7030,  6005,  5147,  4427,  3820,  3307,  2872
        ])
        # fmt: on
        return [(t_i, y_i)]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        x1, x2, x3 = x
        t_i, y_i = args
        return x1 * jnp.exp(x2 / (t_i + x3)) - y_i


# TUOS (11)
# This function has a known typo in TUOS, the symbol `mi` represents
# the minus sign.
class GULFRnD(AbstractLeastSquaresProblem):
    in_dim: ClassVar[int] = 3
    out_dim: int = 99
    name: ClassVar[str] = "Gulf research and development function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, (50, 25, 1.5))

    def __check_init__(self):
        if self.out_dim > 100 or self.out_dim < self.in_dim:
            raise ValueError("`out_dim` must be between 3 and 100")

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        fixed_dim(options, self.name)
        return array_tuple([5.0, 2.5, 0.15])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        i = jnp.arange(1, self.out_dim + 1)
        t_i = i / 100
        y_i = 25 + (-50 * jnp.log(t_i)) ** (2 / 3)
        return [(i, t_i, y_i)]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        x1, x2, x3 = x
        i, t_i, y_i = args
        return jnp.exp(-(jnp.abs(y_i - x2) ** x3) / x1) - t_i


# TUOS (12)
class BoxThreeDim(AbstractLeastSquaresProblem):
    in_dim: ClassVar[int] = 3
    out_dim: int = 100
    name: ClassVar[str] = "Box three-dimensional function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, (1, 10, 1))

    def __check_init__(self):
        if self.out_dim < self.in_dim:
            raise ValueError(f"`out_dim` must be greater than {self.in_dim}")

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        fixed_dim(options, self.name)
        return array_tuple([0.0, 10.0, 20.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        t_i = (0.1) * jnp.arange(1, self.out_dim + 1)
        c = jnp.array(10)
        return [(t_i, c)]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        x1, x2, x3 = x
        t_i, c = args
        exp_t_diff = x3 * (jnp.exp(-t_i) - jnp.exp(-c * t_i))
        return jnp.exp(-t_i * x1) - jnp.exp(-t_i * x2) - exp_t_diff


# TUOS (13)
class PowellSingular(AbstractLeastSquaresProblem):
    in_dim: ClassVar[int] = 4
    out_dim: ClassVar[int] = 4
    name: ClassVar[str] = "Powell's singular function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, (0.0, 0.0, 0.0, 0.0))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        fixed_dim(options, self.name)
        return array_tuple([3.0, -1.0, 0.0, 1.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([10.0, jnp.sqrt(5), 2.0, jnp.sqrt(10)])]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        x1, x2, x3, x4 = x
        c1, c2, c3, c4 = args
        f1 = x1 + c1 * x2
        f2 = c2 * (x3 - x4)
        f3 = (x2 - c3 * x4) ** 2
        f4 = c4 * (x1 - x4) ** 2
        return (f1, f2, f3, f4)


# TUOS (14)
class Wood(AbstractLeastSquaresProblem):
    in_dim: ClassVar[int] = 4
    out_dim: ClassVar[int] = 6
    name: ClassVar[str] = "Wood function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, (1.0, 1.0, 1.0, 1.0))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        fixed_dim(options, self.name)
        return array_tuple([-3.0, -1.0, -3.0, -1.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        coeffs = array_tuple(
            [10.0, 1.0, jnp.sqrt(90.0), 1.0, jnp.sqrt(10), jax.lax.rsqrt(10.0)]
        )
        return [coeffs]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        x1, x2, x3, x4 = x
        c1, c2, c3, c4, c5, c6 = args
        f1 = c1 * (x2 - x1**2)
        f2 = c2 - x1
        f3 = c3 * (x4 - x3**2)
        f4 = c4 - x3
        f5 = c5 * (x2 + x4 - 2)
        f6 = c6 * (x2 - x4)
        return (f1, f2, f3, f4, f5, f6)


# TUOS (15)
class KowalikOsborne(AbstractLeastSquaresProblem):
    in_dim: ClassVar[int] = 4
    out_dim: ClassVar[int] = 11
    name: ClassVar[str] = "Kowalik and Osborne's singular function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(3.07505 * 1e-4, None)
    # Another local min of 1.02734*1e-3.

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        fixed_dim(options, self.name)
        return array_tuple([0.25, 0.39, 0.415, 0.39])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        jnp.arange(1, 12)
        y_i = jnp.array(
            [
                0.1957,
                0.1947,
                0.1735,
                0.1600,
                0.0844,
                0.0627,
                0.0456,
                0.0342,
                0.0323,
                0.0235,
                0.0246,
            ]
        )
        u_i = jnp.array(
            [4.0, 2.0, 1.0, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625]
        )
        return [(y_i, u_i)]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        x1, x2, x3, x4 = x
        y_i, u_i = args
        return y_i - (x1 * (u_i**2 + u_i * x2)) / (u_i**2 + u_i * x3 + x4)


# TUOS (16)
class BrownDennis(AbstractLeastSquaresProblem):
    in_dim: ClassVar[int] = 4
    out_dim: int
    name: ClassVar[str] = "Brown and Dennis function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum

    def __init__(self, out_dim=20):
        self.out_dim = 20
        if self.out_dim == 20:
            self.minimum = Minimum(85822.2, None)
        else:
            self.minimum = Minimum(None, None)

    def __check_init__(self):
        if self.out_dim < self.in_dim:
            raise ValueError(f"`out_dim` must be greater than {self.in_dim}")

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        fixed_dim(options, self.name)
        return array_tuple([25.0, 5.0, -5.0, -1.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        t_i = jnp.arange(1, self.out_dim + 1) / 5
        return [t_i]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        x1, x2, x3, x4 = x
        t_i = args
        tmp = (x3 + x4 * jnp.sin(t_i) - jnp.cos(t_i)) ** 2
        return (x1 + t_i * x2 - jnp.exp(t_i)) ** 2 + tmp


# TUOS (17)
class Osborne1(AbstractLeastSquaresProblem):
    in_dim: ClassVar[int] = 5
    out_dim: ClassVar[int] = 33
    name: ClassVar[str] = "Osborne 1 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(5.46489 * 1e-5, None)

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        fixed_dim(options, self.name)
        return array_tuple([0.5, 1.5, -1.0, 0.01, 0.02])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        # This is (i - 1) in TUOS since they use 1-indexing.
        t_i = 10 * jnp.arange(33)
        # fmt: off
        y_i = jnp.array([
            0.844, 0.908, 0.932, 0.936, 0.925, 0.908, 0.881, 0.850, 0.818,
            0.784, 0.751, 0.718, 0.685, 0.658, 0.628, 0.603, 0.580, 0.558,
            0.538, 0.522, 0.506, 0.490, 0.478, 0.467, 0.457, 0.448, 0.438,
            0.431, 0.424, 0.420, 0.414, 0.411, 0.406
        ])
        # fmt: on
        return [(t_i, y_i)]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        x1, x2, x3, x4, x5 = x
        t_i, y_i = args
        return y_i - (x1 + x2 * jnp.exp(-t_i * x4) + x3 * jnp.exp(-t_i * x5))


# TUOS (18)
class BiggsEXP6(AbstractLeastSquaresProblem):
    in_dim: ClassVar[int] = 6
    out_dim: int
    name: ClassVar[str] = "Biggs EXP6 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum

    def __init__(self, out_dim: int = 42):
        self.out_dim = out_dim
        if self.out_dim == 13:
            # so silly!
            self.minimum = Minimum(5.65565 * 1e-3, None)
        else:
            self.minimum = Minimum(0.0, (1.0, 10.0, 1.0, 5.0, 4.0, 3.0))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        fixed_dim(options, self.name)
        return array_tuple([1.0, 2.0, 1.0, 1.0, 1.0, 1.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        i = jnp.arange(1, self.out_dim + 1)
        t_i = (0.1) * i
        y_i = jnp.exp(-t_i) - 5 * jnp.exp(-10 * t_i) + 3 * jnp.exp(-4 * t_i)
        return [(t_i, y_i)]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        x1, x2, x3, x4, x5, x6 = x
        t_i, y_i = args
        tmp = x3 * jnp.exp(-t_i * x1) - x4 * jnp.exp(-t_i * x2)
        return tmp + x6 * jnp.exp(-t_i * x5) - y_i


# TUOS (19)
class Osborne2(AbstractLeastSquaresProblem):
    in_dim: ClassVar[int] = 11
    out_dim: ClassVar[int] = 65
    name: ClassVar[str] = "Osborne 2 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(4.01377 * 1e-2, None)

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        fixed_dim(options, self.name)
        return array_tuple([1.3, 0.65, 0.65, 0.7, 0.6, 3, 5, 7, 2, 4.5, 5.5])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        t_i = jnp.arange(1, self.out_dim + 1) / 10
        # fmt: off
        y_i = jnp.array([
            1.366, 1.191, 1.112, 1.013, 0.991, 0.885, 0.831, 0.847, 0.786,
            0.725, 0.746, 0.679, 0.608, 0.655, 0.616, 0.606, 0.602, 0.626,
            0.651, 0.724, 0.649, 0.649, 0.694, 0.644, 0.624, 0.661, 0.612,
            0.558, 0.533, 0.495, 0.500, 0.423, 0.395, 0.375, 0.372, 0.391,
            0.396, 0.405, 0.428, 0.429, 0.523, 0.562, 0.607, 0.653, 0.672,
            0.708, 0.633, 0.668, 0.645, 0.632, 0.591, 0.559, 0.597, 0.625,
            0.739, 0.710, 0.729, 0.720, 0.636, 0.581, 0.428, 0.292, 0.162,
            0.098, 0.054
        ])
        # fmt: on
        return [(t_i, y_i)]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = x
        t_i, y_i = args
        tmp1 = x1 * jnp.exp(-t_i * x5) + x2 * jnp.exp(-((t_i - x9) ** 2) * x6)
        tmp2 = x3 * jnp.exp(-((t_i - x10) ** 2) * x7) + x4 * jnp.exp(
            -((t_i - x11) ** 2) * x8
        )
        return y_i - tmp1 + tmp2


# TUOS (20)
class Watson(AbstractLeastSquaresProblem):
    in_dim: int
    out_dim: ClassVar[int] = 31
    name: ClassVar[str] = "Watson function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum

    def __check_init__(self):
        if self.in_dim < 2 or self.in_dim > 31:
            raise ValueError(f"`in_dim` must be between 2 and 31 for {self.name}")

    def __init__(self, in_dim: int = 12):
        self.in_dim = in_dim

        if self.in_dim == 6:
            self.minimum = Minimum(2.28767 * 1e-3, None)
        elif self.in_dim == 9:
            self.minimum = Minimum(1.39976 * 1e-6, None)
        elif self.in_dim == 12:
            self.minimum = Minimum(4.72238 * 1e-10, None)
        else:
            self.minimum = Minimum(None, None)

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        fixed_dim(options, self.name)
        return jnp.zeros(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        t_i = jnp.arange(1, 30) / 30
        return [t_i]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        xs = x
        t_i = args

        x1 = xs[0]
        x2 = xs[1]
        f30 = x1
        f31 = x2 - x1**2 - 1

        index = jnp.arange(1, x.size + 1)
        x_2nd = xs[1:]
        index_2nd = index[1:]
        # broadcasted power
        t_pow1 = t_i ** (index_2nd - 2)[:, None]
        t_pow2 = t_i ** (index - 1)[:, None]
        sum1 = jnp.einsum("i, ij -> j", (index_2nd - 1) * x_2nd, t_pow1)
        sum2 = jnp.einsum("i, ij -> j", xs, t_pow2) ** 2
        return (sum1 - sum2 - 1, f30, f31)


# TUOS (22)
class ExtendedPowellSingular(AbstractLeastSquaresProblem):
    in_dim: int
    out_dim: int
    name: ClassVar[str] = "Extended Powell singular function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum

    def __check_init__(self):
        if self.in_dim % 4 != 0:
            raise ValueError(f"{self.name} only supports `in_dim`s divisible by 4.")

    def __init__(self, in_dim: int = 40):
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.minimum = Minimum(0.0, jnp.zeros(self.in_dim))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        fixed_dim(options, self.name)
        index = jnp.arange(1, self.in_dim + 1)
        init = jnp.ones(self.in_dim)
        init = jnp.where(index % 4 == 1, 3 * init, init)
        init = jnp.where(index % 4 == 2, -1 * init, init)
        init = jnp.where(index % 4 == 3, 0 * init, init)
        return init

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([10.0, jnp.sqrt(5.0), 2.0, jnp.sqrt(10.0)])]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2, c3, c4 = args
        index = jnp.arange(self.in_dim)

        x_tmp1 = x[jnp.minimum(index + 1, self.in_dim)]
        x_tmp2 = x[jnp.minimum(index + 2, self.in_dim)]
        x_tmp3 = x[jnp.maximum(index - 1, 0)]
        x_tmp4 = x[jnp.maximum(index - 3, 0)]

        f1 = jnp.where((index + 1) % 4 == 1, x + c1 * (x + c1 * x_tmp1), 0)
        f2 = jnp.where((index + 1) % 4 == 2, c2 * (x_tmp1 - x_tmp2), 0)
        f3 = jnp.where((index + 1) % 4 == 3, (x_tmp3 - c3 * x) ** 2, 0)
        f4 = jnp.where((index + 1) % 4 == 0, c4 * (x_tmp4 - x), 0)
        return (f1, f2, f3, f4)


# TUOS (23)
class PenaltyFunction1(AbstractLeastSquaresProblem):
    in_dim: int
    out_dim: int
    name: ClassVar[str] = "Penalty function 1"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum

    def __init__(self, in_dim: int = 10):
        self.in_dim = in_dim
        self.out_dim = self.in_dim + 1
        if self.in_dim == 4:
            self.minimum = Minimum(2.24997 * 1e-5, None)
        elif self.in_dim == 10:
            self.minimum = Minimum(7.08765 * 1e-5, None)
        else:
            self.minimum = Minimum(None, None)

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        fixed_dim(options, self.name)
        return jnp.arange(1, self.in_dim + 1)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([jnp.sqrt(1e-5), 1, 0.25])]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2, c3 = args
        fi = c1 * (x - c2)
        f_m = jnp.sum(x**2) - c3
        return (fi, f_m)


# TUOS (24)
class PenaltyFunction2(AbstractLeastSquaresProblem):
    in_dim: int
    out_dim: int
    name: ClassVar[str] = "Penalty function 2"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum

    def __init__(self, in_dim: int = 10):
        self.in_dim = in_dim
        self.out_dim = 2 * self.in_dim
        if self.in_dim == 4:
            self.minimum = Minimum(9.37629 * 1e-6, None)
        elif self.in_dim == 10:
            self.minimum = Minimum(2.93660 * 1e-4, None)
        else:
            self.minimum = Minimum(None, None)

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        fixed_dim(options, self.name)
        return 0.5 * jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        i = jnp.arange(2, self.in_dim + 1)
        y_i = jnp.exp(i / 10) + jnp.exp((i - 1) / 10)
        return [(y_i, jnp.asarray(jnp.sqrt(1e-5)), jnp.asarray(1.0))]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y_i, c1, c2 = args
        x_mid = x[1:-1]

        index = self.in_dim - jnp.arange(1, self.in_dim + 1) + 1

        f1 = c1 * (jnp.exp(x[1:] / 10) + jnp.exp(x[:-1] / 10) - y_i)
        f2 = c1 * (jnp.exp(x_mid / 10) - jnp.exp(-1 / 10))
        f3 = jnp.sum(index * x**2) - c2
        return f1, f2, f3


## TUOS (25)
class VariablyDimensioned(AbstractLeastSquaresProblem):
    in_dim: int
    out_dim: int
    name: ClassVar[str] = "Variably dimensioned function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum

    def __init__(self, in_dim: int = 10):
        self.in_dim = in_dim
        self.out_dim = self.in_dim + 2
        self.minimum = Minimum(0.0, jnp.ones(self.in_dim))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        fixed_dim(options, self.name)
        return 1 - jnp.arange(1, self.in_dim + 1) / self.in_dim

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [(jnp.array(1.0))]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c = args
        index = jnp.arange(1, self.in_dim + 1)

        f1 = x - c
        f2 = jnp.sum(index * (x - 1))
        f3 = jnp.sum(index * (x - 1)) ** 2
        return (f1, f2, f3)


## TUOS (26)
class Trigonometric(AbstractLeastSquaresProblem):
    in_dim: int
    out_dim: int
    name: ClassVar[str] = "Trigonometric function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, None)

    def __init__(self, in_dim: int = 24):
        # arbitrary default
        self.in_dim = in_dim
        self.out_dim = self.in_dim

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return jnp.ones(self.in_dim) / self.in_dim

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [None]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        index = jnp.arange(1, self.in_dim + 1)
        return self.in_dim - jnp.sum(jnp.cos(x)) + index * (1 - jnp.cos(x)) - jnp.sin(x)


## TUOS (27)
class BrownAlmostLinear(AbstractLeastSquaresProblem):
    in_dim: int
    out_dim: int
    name: ClassVar[str] = "Brown's almost-linear function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum

    def __init__(self, in_dim: int = 24):
        # arbitrary default
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.minimum = Minimum(0.0, jnp.ones(in_dim))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return 0.5 * jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        # The constants are closely tied in a way that doesn't make sense to
        # include as args here.
        return [None]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args

        f1 = x + jnp.sum(x) - self.in_dim - 1
        f2 = jnp.prod(x) - 1
        return (f1, f2)


## TUOS (28)
class DiscreteBoundary(AbstractLeastSquaresProblem):
    in_dim: int
    out_dim: int
    name: ClassVar[str] = "Discrete boundary value function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, None)

    def __init__(self, in_dim: int = 24):
        # arbitrary default
        self.in_dim = in_dim
        self.out_dim = self.in_dim

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        t_i = jnp.arange(1, self.in_dim + 1) / (self.in_dim + 1)
        return t_i * (t_i - 1)

    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        # h and t_i are natural candidates for `args` but don't work easily
        # in practice. May change this in the future but it's not high priority.
        return [None]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        h = 1 / (self.in_dim + 1)
        t_i = jnp.arange(1, self.in_dim + 1) / (self.in_dim + 1)

        x_prev = jnp.insert(x[:-1], 0, jnp.array(0.0), axis=0)
        x_next = jnp.insert(x[1:], -1, jnp.array(0.0), axis=0)
        return 2 * x - x_prev - x_next + (h**3 * (x + t_i + 1) ** 3) / 2


## TUOS (29)
class DiscreteIntegral(AbstractLeastSquaresProblem):
    in_dim: int
    out_dim: int
    name: ClassVar[str] = "Discrete integral equation function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, None)

    def __init__(self, in_dim: int = 24):
        # arbitrary default
        self.in_dim = in_dim
        self.out_dim = self.in_dim

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        t_i = jnp.arange(1, self.in_dim + 1) / (self.in_dim + 1)
        return t_i * (t_i - 1)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        # h and t_i are natural candidates for `args` but don't work easily
        # in practice. May change this in the future but it's not high priority.
        return [None]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        h = 1 / (self.in_dim + 1)
        t_i = jnp.arange(1, self.in_dim + 1) / (self.in_dim + 1)
        out1 = jnp.outer((1 - t_i), t_i)
        out2 = jnp.outer(t_i, (t_i - t_i))
        t_grid = jnp.tril(out1) + jnp.triu(out2, 1)
        return x + h * (t_grid @ (t_i * (x + t_i + 1) ** 3)) / 2


## TUOS (30)
class BroydenTridiagonal(AbstractLeastSquaresProblem):
    in_dim: int
    out_dim: int
    name: ClassVar[str] = "Broyden tridiagonal function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, None)

    def __init__(self, in_dim: int = 99):
        # arbitrary default
        self.in_dim = in_dim
        self.out_dim = self.in_dim

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return -jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([3.0, 2.0, 2.0, 1.0])]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2, c3, c4 = args
        x_prev = jnp.insert(x[:-1], 0, jnp.array(0.0), axis=0)
        x_next = jnp.insert(x[1:], -1, jnp.array(0.0), axis=0)
        return (c1 - c2 * x) * x - x_prev - c3 * x_next + c4


## TUOS (31)
class BroydenBanded(AbstractLeastSquaresProblem):
    in_dim: int
    out_dim: int
    name: ClassVar[str] = "Broyden banded function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, None)

    def __check_init__(self):
        if self.in_dim <= 5:
            raise ValueError(f"{self.name} requires `in_dim > 5`.")

    def __init__(self, in_dim: int = 99):
        # arbitrary default
        self.in_dim = in_dim
        self.out_dim = self.in_dim

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return -jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([2.0, 5.0, 1.0, 1.0])]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2, c3, c4 = args
        l5 = jnp.diag(jnp.ones(self.in_dim - 5), -5)
        l4 = jnp.diag(jnp.ones(self.in_dim - 4), -4)
        l3 = jnp.diag(jnp.ones(self.in_dim - 3), -3)
        l2 = jnp.diag(jnp.ones(self.in_dim - 2), -2)
        l1 = jnp.diag(jnp.ones(self.in_dim - 1), -1)
        u1 = jnp.diag(jnp.ones(self.in_dim - 1), 1)
        banded = u1 + l1 + l2 + l3 + l4 + l5
        return x * (c1 + c2 * x**2) + c3 - banded @ (x * (c4 + x))


## TUOS (32)
class LinearFullRank(AbstractLeastSquaresProblem):
    in_dim: int
    out_dim: int
    name: ClassVar[str] = "Linear function - full rank"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum

    def __check_init__(self):
        if self.in_dim > self.out_dim:
            raise ValueError(f"{self.name} requires `in_dim <= out_dim`.")

    def __init__(self, in_dim: int = 99, out_dim: int = 99):
        # arbitrary default
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.minimum = Minimum(self.out_dim - self.in_dim, -jnp.ones(self.in_dim))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [None]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        shared_sum = -2 / self.out_dim * jnp.sum(x) - 1
        f1 = jnp.ones(self.out_dim - self.in_dim) * shared_sum
        f2 = x + shared_sum
        return (f1, f2)


## TUOS (33)
class LinearRank1(AbstractLeastSquaresProblem):
    in_dim: int
    out_dim: int
    name: ClassVar[str] = "Linear function - rank 1"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum

    def __check_init__(self):
        if self.in_dim > self.out_dim:
            raise ValueError(f"{self.name} requires `in_dim <= out_dim`.")

    def __init__(self, in_dim: int = 99, out_dim: int = 99):
        # arbitrary default
        self.in_dim = in_dim
        self.out_dim = out_dim
        test_min = jnp.zeros(in_dim)
        # magic val makes it satisfy the correct equation to reach min
        magicval = 3 / (2 * (2 * self.out_dim + 1))
        minval = (self.out_dim * (self.out_dim - 1)) / (2 * (2 * self.out_dim + 1))
        self.minimum = Minimum(minval, test_min.at[1].set(magicval))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [None]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        j = jnp.arange(1, 1 + self.in_dim)
        i = jnp.arange(1, 1 + self.out_dim)
        return i * jnp.sum(j * x) - 1


## TUOS (34)
class LinearRank1Zero(AbstractLeastSquaresProblem):
    in_dim: int
    out_dim: int
    name: ClassVar[str] = "Linear function - rank 1 with zero columns and rows"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum

    def __check_init__(self):
        if self.in_dim > self.out_dim:
            raise ValueError(f"{self.name} requires `in_dim <= out_dim`.")

    def __init__(self, in_dim: int = 99, out_dim: int = 99):
        # arbitrary default
        self.in_dim = in_dim
        self.out_dim = out_dim

        test_min = jnp.zeros(in_dim)
        numerator = self.out_dim**2 + 3 * self.out_dim - 6
        denom = 2 * (2 * self.out_dim - 3)

        magicval = 1 / (2 * self.out_dim - 3)

        self.minimum = Minimum(numerator / denom, test_min.at[2].set(magicval))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [None]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        i = jnp.arange(2, self.out_dim)
        j = jnp.arange(2, self.in_dim)
        x_mid = x[1:-1]
        f1 = -1
        fm = -1
        return (f1, (i - 1) * jnp.sum(j * x_mid) - 1, fm)


## TUOS (35)
class Chebyquad(AbstractLeastSquaresProblem):
    in_dim: int
    out_dim: int
    name: ClassVar[str] = "Chebyquad function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum

    def __check_init__(self):
        if self.in_dim > self.out_dim:
            raise ValueError(f"{self.name} requires `in_dim <= out_dim`.")

    def __init__(self, in_dim: int = 10, out_dim: int = 10):
        # arbitrary default
        self.in_dim = in_dim
        self.out_dim = out_dim

        if (self.in_dim == self.out_dim) and ((self.in_dim < 7) or (self.in_dim == 9)):
            self.minimum = Minimum(0, None)
        elif (self.in_dim == self.out_dim) and (self.in_dim == 8):
            self.minimum = Minimum(3.51687 * 1e-3, None)
        elif (self.in_dim == self.out_dim) and (self.in_dim == 10):
            self.minimum = Minimum(6.50395 * 1e-3, None)

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return jnp.arange(1, self.in_dim + 1) / (self.in_dim + 1)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [None]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        index = jnp.arange(1, self.out_dim + 1)
        cheb_integrals = jnp.where(index % 2 == 0, -1 / (index**2 - 1), 0)
        cheb_broadcast = lambda y: jax.vmap(self._simple_cheb, in_axes=(0, None))(
            index, y
        )
        # Returns the array who's ith value is `sum_j (T_i(x_j))`.
        broadcast_sum = jnp.sum(jax.vmap(cheb_broadcast)(x), axis=0)
        return 1 / self.out_dim * broadcast_sum - cheb_integrals

    def _simple_cheb(self, n: Scalar, x: Scalar):
        """A simple approximation to the chebyshev polynomial of the first
        kind shifted to [0, 1]. Only valid in this range.
        """
        return jnp.cos(n * jnp.arccos(2 * x - 1))


#
# MINPACK2
# This problem models the nonuniformity of a lead-tin coating on a sheet.
# `_z1` is a model for the coating thickness and `_z2` is a model for the relative
# abundance of lead to tin. In MINPACK, the fit model are compared to measured values
# `y_1, y_2`, and thus subject to noise. To simulate this, we generate data `y_1`,
# `y_2` from the model with a hidden set of parameters, and use the user's
# `random_generator` to determine the amount of random noise to add. We then fit
# fit the model to the hidden parameters.
#
class CoatingThickness(AbstractLeastSquaresProblem):
    in_dim: int
    out_dim: int
    name: ClassVar[str] = "Coating Thickness Standardization"
    difficulty: ClassVar[Optional[Difficulty]] = Difficulty.HARD
    minimum: ClassVar[Optional[float]] = None

    def __check_init__(self):
        if (self.in_dim - 8) / 2 == 0 or self.in_dim < 8:
            raise ValueError(
                f"{self.name} requires `in_dim` - 8 to be" "an even nonnegative number"
            )

    def __init__(self, in_dim: int = 48):
        # arbitrary default
        self.in_dim = in_dim
        self.out_dim = self.in_dim - 8

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        # params is arbitrary, perturbations is not.
        grid_dim = (self.in_dim - 8) // 2
        init_parameters = 0.5 * jnp.ones(8)
        init_perturbations = jnp.zeros(grid_dim, 2)
        return (init_parameters, init_perturbations)

    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: PRNGKeyArray,
    ) -> list[Optional[PyTree[Array]]]:
        # TODO(packquickly): document this well. The problem requires a PRNGKey
        # always, but only a random generator to incorporate measurement error.
        dim2 = self.out_dim // 2
        grid_key, param_key, noise_key = jr.split(key, 3)
        gridpoints = jr.uniform(
            jr.PRNGKey(0), (dim2, 2), dtype=default_floating_dtype()
        )
        hidden_params = jr.normal(jr.PRNGKey(0), (8,), dtype=default_floating_dtype())
        y_1 = jax.vmap(self._z1, in_axes=(None, 0))(hidden_params, gridpoints)
        y_2 = jax.vmap(self._z2, in_axes=(None, 0))(hidden_params, gridpoints)

        # Add noise to the measurements
        if random_generator is not None:
            grid_key, y_1_key, y_2_key = jr.split(noise_key, 3)
            gridpoints = gridpoints + random_generator(
                jax.eval_shape(lambda: gridpoints), key=grid_key
            )
            y_1 = y_1 + random_generator(jax.eval_shape(lambda: y_1), key=y_1_key)
            y_2 = y_2 + random_generator(jax.eval_shape(lambda: y_2), key=y_2_key)

        return [(gridpoints, y_1, y_2)]

    def fn(self, x: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        params, perturbations = x
        gridpoints, y_1, y_2 = args

        perturbed_gridpoints = gridpoints + perturbations
        f1 = jax.vmap(self._z1)(params, perturbed_gridpoints) - y_1
        f2 = jax.vmap(self._z2)(params, perturbed_gridpoints) - y_2
        return (f1, f2)

    def _z1(self, x, gridpoint):
        z1 = gridpoint[0]
        z2 = gridpoint[1]
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = [3]
        return x1 + x2 * z1 + x3 * z2 + x4 * z1 * z2

    def _z2(self, x, gridpoint):
        z1 = gridpoint[0]
        z2 = gridpoint[1]
        x5 = x[4]
        x6 = x[5]
        x7 = x[6]
        x8 = x[7]
        return x5 + x6 * z1 + x7 * z2 + x8 * z2 * z2
