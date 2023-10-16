from typing import ClassVar, Optional

import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.random as jr
from equinox.internal import ω
from jaxtyping import Array, PRNGKeyArray, PyTree

from ..misc import (
    additive_perturbation,
    array_tuple,
    default_floating_dtype,
)
from ..random_generators import RandomGenerator
from .base import AbstractLeastSquaresProblem, Difficulty, Minimum


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
# Continuous Global Optimization Test Problems." by Ali, et al.
# CUTE: "Constrained and Unconstrained Testing Environment" by Gould, Orban, and Toint.
# The CUTE functions are taken from the description in UOTF, not from the FORTRAN
# itself.
#
# LSBF is riddled with errors, if problems implemented from LSBF do not match the actual
# test functions, please open a PR to fix them. Replacing LSBF with
# the POWER/Al-Roomi implementations https://www.al-roomi.org/benchmarks/unconstrained
# in future problems
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
# UOTF includes some "extended" versions of CUTE problems, which consists of a number
# of decoupled versions of a CUTE problem. Rosenbrock is included in Optpile, as it's
# shown in WIKI, but the rest are not. It's not obvious how to interpret these
# extended problems, as for a true Hessian they should be solved in the same number
# of steps. For an approximate Hessian, maybe it shows how well the approximation can
# decouple the problems? Further, randominzing the init over such a problem will
# return the worst-case time over the random inits. These aren't necessarily bad,
# but given they're more difficult to interpret, less standard, and harder to
# implement, just the non-extended versions are implemented.
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
    name: ClassVar[str] = "Simple Bowl"
    difficulty: ClassVar[Difficulty] = Difficulty.EASY
    minimum: Minimum
    in_dim: int
    out_dim: int

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
        return jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return jnp.ones(self.in_dim)

    def fn(self, y: PyTree[Array], args: PyTree) -> PyTree[Array]:
        return (args**ω * ω(y).call(jnp.square)).ω


# TUOS (21)
# WIKI
class DecoupledRosenbrock(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Decoupled Rosenbrock"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int = 2):
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.minimum = Minimum(0.0, jnp.ones(in_dim))

    def __check_ini__(self):
        if self.in_dim % 2 != 0:
            raise ValueError(f"{self.name} requires an even `in_dim`.")

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        index = jnp.arange(1, self.in_dim + 1)
        return jnp.where(index % 2 == 0, 1, -1.2)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return None

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        # Rosenbrock doesn't translate well to PyTrees
        flat_y, _ = jfu.ravel_pytree(y)
        index = jnp.arange(1, flat_y.size + 1, dtype=default_floating_dtype())
        jnp.where(index % 2 == 0, flat_y, flat_y**2)
        # Remember that all these values will be squared, hence why
        # this looks different than what is on Wikipedia.
        diffs_y = 10 * (flat_y[:-1] - flat_y[1:])
        diffs_1 = jnp.where(index % 2 != 0, (1 - flat_y), 0.0)
        return (diffs_y, diffs_1)


class CoupledRosenbrock(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Coupled Rosenbrock"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: int

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
        index = jnp.arange(1, self.in_dim + 1)
        return jnp.where(index % 2 == 0, 1, -1.2)

    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return None

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        flat_y, _ = jfu.ravel_pytree(y)
        diffs_y = 10 * (flat_y[1:] - flat_y[:-1] ** 2)
        diffs_1 = 1 - flat_y
        return (diffs_y, diffs_1)


# TUOS (2)
class FreudensteinRoth(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Freudenstein and Roth function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.5 * 48.9842, (5.0, 4.0))
    in_dim: ClassVar[int] = 2
    out_dim: ClassVar[int] = 2

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return array_tuple([0.5, -2.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return array_tuple([13.0, 5.0, 2.0, 29.0, 1.0, 14.0])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        c1, c2, c3, c4, c5, c6 = args
        f1 = -c1 + y1 + ((c2 - y2) * y2 - c3) * y2
        f2 = -c4 + y1 + ((y2 + c5) * y2 - c6) * y2
        return (f1, f2)


# TUOS (3)
class PowellBadlyScaled(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Powell's badly scaled function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0, (1.098, 9.1 - 6))
    in_dim: ClassVar[int] = 2
    out_dim: ClassVar[int] = 2

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return array_tuple([0.0, 1.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return array_tuple([1e4, 1.0, 1.0001])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        c1, c2, c3 = args
        f1 = c1 * y1 * y2 - c2
        f2 = jnp.exp(-y1) + jnp.exp(-y2) - c3
        return (f1, f2)


# TUOS (4)
class BrownBadlyScaled(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Brown's badly scaled function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, (1e6, 2e-6))
    in_dim: ClassVar[int] = 2
    out_dim: ClassVar[int] = 3

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return array_tuple([1.0, 1.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return array_tuple([1e6, 2.0, 1e-6, 2.0])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        c1, c2, c3, c4 = args
        f1 = y1 - c1
        f2 = y2 - c2 * c3
        f3 = y1 * y2 - c4
        return (f1, f2, f3)


# TUOS (5)
class Beale(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Beale's function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, (3, 0.5))
    in_dim: ClassVar[int] = 2
    out_dim: ClassVar[int] = 3

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return array_tuple([1.0, 1.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        i = jnp.arange(1, 4, dtype=default_floating_dtype())
        z_i = jnp.array([1.5, 2.25, 2.625])
        return (i, z_i)

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        i, z_i = args
        return z_i - y1 * (1 - y2**i)


# TUOS (6)
class JennrichSampson(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Jennrich and Sampson function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: ClassVar[int] = 2
    out_dim: int

    def __init__(self, out_dim: int = 10):
        self.out_dim = out_dim
        if self.out_dim == 10:
            self.minimum = Minimum(0.5 * 124.362, (0.2578, 0.2578))
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
        return array_tuple([0.3, 0.4])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        c1 = jnp.array(2.0)
        c2 = jnp.array(2.0)
        i = jnp.arange(1, self.out_dim + 1, dtype=default_floating_dtype())
        return (c1, c2, i)

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        c1, c2, i = args
        return 2 + 2 * i - (jnp.exp(i * y1) + jnp.exp(i * y2))


# TUOS (7)
class HelicalValley(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Helical valley function (decoupled)"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, (1.0, 0.0, 0.0))
    in_dim: ClassVar[int] = 3
    out_dim: ClassVar[int] = 3

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
    ) -> Optional[PyTree]:
        return array_tuple([10.0, 1.0, 0.5])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2, y3 = y
        c1, c2, c3 = args

        info = jnp.finfo(jnp.asarray(y1).dtype)
        y1_nonzero = jnp.abs(y1) > info.eps
        jnp.where(y1_nonzero, y1, 1)
        y2y1ratio = jnp.where(y1_nonzero, y2 / y1, info.max)
        arctan_val = 1 / (2 * jnp.pi) * jnp.arctan(y2y1ratio)
        theta = jnp.where(y1 > 0, arctan_val, arctan_val + c3)

        f1 = c1 * (y3 - c1 * theta)
        f2 = c1 * (jnp.sqrt(y1**2 + y2**2) - c2)
        f3 = y3
        return (f1, f2, f3)


# TUOS (8)
class Bard(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Bard function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.5 * 8.21487 * 1e-3, None)
    in_dim: ClassVar[int] = 3
    out_dim: ClassVar[int] = 15
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
        return array_tuple([1.0, 1.0, 1.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        u_i = jnp.arange(1, 16, dtype=default_floating_dtype())
        v_i = 16 - u_i
        w_i = jnp.minimum(u_i, v_i)
        # fmt: off
        z_i = jnp.array([
            0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39,
            0.37, 0.58, 0.73, 0.96, 1.34, 2.10, 4.39
        ])
        # fmt: on
        return (u_i, v_i, w_i, z_i)

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2, y3 = y
        u_i, v_i, w_i, z_i = args
        return z_i - (y1 + u_i / (v_i * y2 + w_i * y3))


# TUOS (9)
class Gaussian(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Gaussian function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.5 * 1.12798 * 1e-8, None)
    in_dim: ClassVar[int] = 3
    out_dim: ClassVar[int] = 15

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return array_tuple([0.4, 1.0, 0.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        i = jnp.arange(1, 16, dtype=default_floating_dtype())
        t_i = (8 - i) / 2
        # fmt: off
        z_i = jnp.array([
                0.0009, 0.0044, 0.0175, 0.0540, 0.1295,
                0.2420, 0.3521, 0.3989, 0.3521, 0.2420,
                0.1295, 0.0540, 0.0175, 0.0044, 0.0009,
            ])
        # fmt: on
        return (t_i, z_i)

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2, y3 = y
        t_i, z_i = args
        return y1 * jnp.exp((-y2 * (t_i - y3) ** 2) / 2) - z_i


# TUOS (10)
class Meyer(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Meyer function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.5 * 87.9458, None)
    in_dim: ClassVar[int] = 3
    out_dim: ClassVar[int] = 16

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return array_tuple([0.02, 4000.0, 250.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        i = jnp.arange(1, self.out_dim + 1, dtype=default_floating_dtype())
        t_i = 45.0 + 5.0 * i
        # fmt: off
        z_i = jnp.array([
            34780, 28610, 23650, 19630, 16370, 13720, 11540, 9744, 
            8261,  7030,  6005,  5147,  4427,  3820,  3307,  2872
        ])
        # fmt: on
        return (t_i, z_i)

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2, y3 = y
        t_i, z_i = args
        return y1 * jnp.exp(y2 / (t_i + y3)) - z_i


# TUOS (11)
# This function has a known typo in TUOS, the symbol `mi` represents
# the minus sign.
class GULFRnD(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Gulf research and development function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, (50, 25, 1.5))
    in_dim: ClassVar[int] = 3
    out_dim: int = 99

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
        return array_tuple([5.0, 2.5, 0.15])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        i = jnp.arange(1, self.out_dim + 1, dtype=default_floating_dtype())
        t_i = i / 100
        z_i = 25.0 + (-50.0 * jnp.log(t_i)) ** (2 / 3)
        return (i, t_i, z_i)

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2, y3 = y
        i, t_i, z_i = args
        return jnp.exp(-(jnp.abs(z_i - y2) ** y3) / y1) - t_i


# TUOS (12)
class BoxThreeDim(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Box three-dimensional function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, (1, 10, 1))
    in_dim: ClassVar[int] = 3
    out_dim: int = 100

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
        return array_tuple([0.0, 10.0, 20.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        t_i = (0.1) * jnp.arange(1, self.out_dim + 1, dtype=default_floating_dtype())
        c = jnp.array(10.0)
        return (t_i, c)

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2, y3 = y
        t_i, c = args
        exp_t_diff = y3 * (jnp.exp(-t_i) - jnp.exp(-c * t_i))
        return jnp.exp(-t_i * y1) - jnp.exp(-t_i * y2) - exp_t_diff


# TUOS (13)
class PowellSingular(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Powell's singular function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, (0.0, 0.0, 0.0, 0.0))
    in_dim: ClassVar[int] = 4
    out_dim: ClassVar[int] = 4

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return array_tuple([3.0, -1.0, 0.0, 1.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return array_tuple([10.0, jnp.sqrt(5), 2.0, jnp.sqrt(10)])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2, y3, y4 = y
        c1, c2, c3, c4 = args
        f1 = y1 + c1 * y2
        f2 = c2 * (y3 - y4)
        f3 = (y2 - c3 * y4) ** 2
        f4 = c4 * (y1 - y4) ** 2
        return (f1, f2, f3, f4)


# TUOS (14)
class Wood(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Wood function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, (1.0, 1.0, 1.0, 1.0))
    in_dim: ClassVar[int] = 4
    out_dim: ClassVar[int] = 6

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return array_tuple([-3.0, -1.0, -3.0, -1.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        coeffs = array_tuple(
            [10.0, 1.0, jnp.sqrt(90.0), 1.0, jnp.sqrt(10), jax.lax.rsqrt(10.0)]
        )
        return coeffs

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2, y3, y4 = y
        c1, c2, c3, c4, c5, c6 = args
        f1 = c1 * (y2 - y1**2)
        f2 = c2 - y1
        f3 = c3 * (y4 - y3**2)
        f4 = c4 - y3
        f5 = c5 * (y2 + y4 - 2)
        f6 = c6 * (y2 - y4)
        return (f1, f2, f3, f4, f5, f6)


# TUOS (15)
class KowalikOsborne(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Kowalik and Osborne's singular function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.5 * 3.07505 * 1e-4, None)
    in_dim: ClassVar[int] = 4
    out_dim: ClassVar[int] = 11
    # Another local min of 1.02734*1e-3.

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return array_tuple([0.25, 0.39, 0.415, 0.39])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        z_i = jnp.array(
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
        return (z_i, u_i)

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2, y3, y4 = y
        z_i, u_i = args
        return z_i - (y1 * (u_i**2 + u_i * y2)) / (u_i**2 + u_i * y3 + y4)


# TUOS (16)
class BrownDennis(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Brown and Dennis function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: ClassVar[int] = 4
    out_dim: int

    def __init__(self, out_dim=20):
        self.out_dim = 20
        if self.out_dim == 20:
            self.minimum = Minimum(0.5 * 85822.2, None)
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
        return array_tuple([25.0, 5.0, -5.0, -1.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        t_i = jnp.arange(1, self.out_dim + 1) / 5
        return t_i

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2, y3, y4 = y
        t_i = args
        tmp = (y3 + y4 * jnp.sin(t_i) - jnp.cos(t_i)) ** 2
        return (y1 + t_i * y2 - jnp.exp(t_i)) ** 2 + tmp


# # TUOS (17)
# class Osborne1(AbstractLeastSquaresProblem):
#     name: ClassVar[str] = "Osborne 1 function"
#     difficulty: ClassVar[Optional[Difficulty]] = None
#     minimum: ClassVar[Minimum] = Minimum(5.46489 * 1e-5, None)
#     in_dim: ClassVar[int] = 5
#     out_dim: ClassVar[int] = 33

#     @additive_perturbation
#     def init(
#         self,
#         random_generator: Optional[RandomGenerator] = None,
#         options: Optional[dict] = None,
#         *,
#         key: Optional[PRNGKeyArray] = None,
#     ) -> PyTree[Array]:
#         return array_tuple([0.5, 1.5, -1.0, 0.01, 0.02])

#     @additive_perturbation
#     def args(
#         self,
#         random_generator: Optional[RandomGenerator] = None,
#         options: Optional[dict] = None,
#         *,
#         key: Optional[PRNGKeyArray] = None,
#     ) -> Optional[PyTree]:
#         # This is (i - 1) in TUOS since they use 1-indexing.
#         t_i = 10.0 * jnp.arange(33, dtype=default_floating_dtype())
#         # fmt: off
#         z_i = jnp.array([
#             0.844, 0.908, 0.932, 0.936, 0.925, 0.908, 0.881, 0.850, 0.818,
#             0.784, 0.751, 0.718, 0.685, 0.658, 0.628, 0.603, 0.580, 0.558,
#             0.538, 0.522, 0.506, 0.490, 0.478, 0.467, 0.457, 0.448, 0.438,
#             0.431, 0.424, 0.420, 0.414, 0.411, 0.406
#         ])
#         # fmt: on
#         return (t_i, z_i)

#     def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
#         y1, y2, y3, y4, y5 = y
#         t_i, z_i = args
#         return z_i - (y1 + y2 * jnp.exp(-t_i * y4) + y3 * jnp.exp(-t_i * y5))


# TUOS (18)
class BiggsEXP6(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Biggs EXP6 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: ClassVar[int] = 6
    out_dim: int

    def __init__(self, out_dim: int = 42):
        self.out_dim = out_dim
        if self.out_dim == 13:
            # so silly!
            self.minimum = Minimum(0.5 * 5.65565 * 1e-3, None)
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
        return array_tuple([1.0, 2.0, 1.0, 1.0, 1.0, 1.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        i = jnp.arange(1, self.out_dim + 1)
        t_i = (0.1) * i
        z_i = jnp.exp(-t_i) - 5.0 * jnp.exp(-10.0 * t_i) + 3.0 * jnp.exp(-4.0 * t_i)
        return (t_i, z_i)

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2, y3, y4, y5, y6 = y
        t_i, z_i = args
        tmp = y3 * jnp.exp(-t_i * y1) - y4 * jnp.exp(-t_i * y2)
        return tmp + y6 * jnp.exp(-t_i * y5) - z_i


# TUOS (19)
class Osborne2(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Osborne 2 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.5 * 4.01377 * 1e-2, None)
    in_dim: ClassVar[int] = 11
    out_dim: ClassVar[int] = 65

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return array_tuple([1.3, 0.65, 0.65, 0.7, 0.6, 3.0, 5.0, 7.0, 2.0, 4.5, 5.5])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        t_i = jnp.arange(1, self.out_dim + 1) / 10
        # fmt: off
        z_i = jnp.array([
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
        return (t_i, z_i)

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11 = y
        t_i, z_i = args
        tmp1 = y1 * jnp.exp(-t_i * y5) + y2 * jnp.exp(-((t_i - y9) ** 2) * y6)
        tmp2 = y3 * jnp.exp(-((t_i - y10) ** 2) * y7) + y4 * jnp.exp(
            -((t_i - y11) ** 2) * y8
        )
        return z_i - tmp1 + tmp2


# TUOS (20)
class Watson(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Watson function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: ClassVar[int] = 31

    def __check_init__(self):
        if self.in_dim < 2 or self.in_dim > 31:
            raise ValueError(f"`in_dim` must be between 2 and 31 for {self.name}")

    def __init__(self, in_dim: int = 12):
        self.in_dim = in_dim

        if self.in_dim == 6:
            self.minimum = Minimum(0.5 * 2.28767 * 1e-3, None)
        elif self.in_dim == 9:
            self.minimum = Minimum(0.5 * 1.39976 * 1e-6, None)
        elif self.in_dim == 12:
            self.minimum = Minimum(0.5 * 4.72238 * 1e-10, None)
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
        return jnp.zeros(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        t_i = jnp.arange(1, 30) / 30
        return t_i

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        t_i = args

        y1 = y[0]
        y2 = y[1]
        f30 = y1
        f31 = y2 - y1**2 - 1

        index = jnp.arange(1, y.size + 1)
        y_2nd = y[1:]
        index_2nd = index[1:]
        # broadcasted power
        t_pow1 = t_i ** (index_2nd - 2)[:, None]
        t_pow2 = t_i ** (index - 1)[:, None]
        sum1 = jnp.einsum("i, ij -> j", (index_2nd - 1) * y_2nd, t_pow1)
        sum2 = jnp.einsum("i, ij -> j", y, t_pow2) ** 2
        return (sum1 - sum2 - 1, f30, f31)


# TUOS (22)
class ExtendedPowellSingular(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Extended Powell singular function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: int

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
        index = jnp.arange(1, self.in_dim + 1)
        init = jnp.ones(self.in_dim)
        init = jnp.where(index % 4 == 1, 3.0 * init, init)
        init = jnp.where(index % 4 == 2, -1.0 * init, init)
        init = jnp.where(index % 4 == 3, 0.0 * init, init)
        return init

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return array_tuple([10.0, jnp.sqrt(5.0), 2.0, jnp.sqrt(10.0)])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2, c3, c4 = args
        index = jnp.arange(self.in_dim)

        y_tmp1 = y[jnp.minimum(index + 1, self.in_dim)]
        y_tmp2 = y[jnp.minimum(index + 2, self.in_dim)]
        y_tmp3 = y[jnp.maximum(index - 1, 0)]
        y_tmp4 = y[jnp.maximum(index - 3, 0)]

        f1 = jnp.where((index + 1) % 4 == 1, y + c1 * (y + c1 * y_tmp1), 0)
        f2 = jnp.where((index + 1) % 4 == 2, c2 * (y_tmp1 - y_tmp2), 0)
        f3 = jnp.where((index + 1) % 4 == 3, (y_tmp3 - c3 * y) ** 2, 0)
        f4 = jnp.where((index + 1) % 4 == 0, c4 * (y_tmp4 - y), 0)
        return (f1, f2, f3, f4)


# TUOS (23)
class PenaltyFunction1(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Penalty function 1"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int = 10):
        self.in_dim = in_dim
        self.out_dim = self.in_dim + 1
        if self.in_dim == 4:
            self.minimum = Minimum(0.5 * 2.24997 * 1e-5, None)
        elif self.in_dim == 10:
            self.minimum = Minimum(0.5 * 7.08765 * 1e-5, None)
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
        return jnp.arange(1, self.in_dim + 1, dtype=default_floating_dtype())

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return array_tuple([jnp.sqrt(1e-5), 1.0, 0.25])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2, c3 = args
        fi = c1 * (y - c2)
        f_m = jnp.sum(y**2) - c3
        return (fi, f_m)


# TUOS (24)
class PenaltyFunction2(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Penalty function 2"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int = 10):
        self.in_dim = in_dim
        self.out_dim = 2 * self.in_dim
        if self.in_dim == 4:
            self.minimum = Minimum(0.5 * 9.37629 * 1e-6, None)
        elif self.in_dim == 10:
            self.minimum = Minimum(0.5 * 2.93660 * 1e-4, None)
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
        return 0.5 * jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        i = jnp.arange(2, self.in_dim + 1)
        z_i = jnp.exp(i / 10) + jnp.exp((i - 1) / 10)
        return (z_i, jnp.asarray(jnp.sqrt(1e-5)), jnp.asarray(1.0))

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        z_i, c1, c2 = args
        y_mid = y[1:-1]

        index = self.in_dim - jnp.arange(1, self.in_dim + 1) + 1

        f1 = c1 * (jnp.exp(y[1:] / 10) + jnp.exp(y[:-1] / 10) - z_i)
        f2 = c1 * (jnp.exp(y_mid / 10) - jnp.exp(-1 / 10))
        f3 = jnp.sum(index * y**2) - c2
        return f1, f2, f3


# TUOS (25)
class VariablyDimensioned(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Variably dimensioned function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: int

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
        return 1 - jnp.arange(1, self.in_dim + 1) / self.in_dim

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return jnp.array(1.0)

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c = args
        index = jnp.arange(1, self.in_dim + 1)

        f1 = y - c
        f2 = jnp.sum(index * (y - 1))
        f3 = jnp.sum(index * (y - 1)) ** 2
        return (f1, f2, f3)


# TUOS (26)
class Trigonometric(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Trigonometric function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, None)
    in_dim: int
    out_dim: int

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
    ) -> Optional[PyTree]:
        return None

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        index = jnp.arange(1, self.in_dim + 1)
        return self.in_dim - jnp.sum(jnp.cos(y)) + index * (1 - jnp.cos(y)) - jnp.sin(y)


# TUOS (27)
class BrownAlmostLinear(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Brown's almost-linear function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: int

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
    ) -> Optional[PyTree]:
        # The constants are closely tied in a way that doesn't make sense to
        # include as args here.
        return None

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args

        f1 = y + jnp.sum(y) - self.in_dim - 1
        f2 = jnp.prod(y) - 1
        return (f1, f2)


# TUOS (28)
class DiscreteBoundary(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Discrete boundary value function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, None)
    in_dim: int
    out_dim: int

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
    ) -> Optional[PyTree]:
        # h and t_i are natural candidates for `args` but don't work easily
        # in practice. May change this in the future but it's not high priority.
        return None

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        h = 1 / (self.in_dim + 1)
        t_i = jnp.arange(1, self.in_dim + 1) / (self.in_dim + 1)

        x_prev = jnp.insert(y[:-1], 0, jnp.array(0.0), axis=0)
        x_next = jnp.insert(y[1:], -1, jnp.array(0.0), axis=0)
        return 2 * y - x_prev - x_next + (h**3 * (y + t_i + 1) ** 3) / 2


# TUOS (29)
class DiscreteIntegral(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Discrete integral equation function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, None)
    in_dim: int
    out_dim: int

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
    ) -> Optional[PyTree]:
        # h and t_i are natural candidates for `args` but don't work easily
        # in practice. May change this in the future but it's not high priority.
        return None

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        h = 1 / (self.in_dim + 1)
        t_i = jnp.arange(1, self.in_dim + 1) / (self.in_dim + 1)
        out1 = jnp.outer((1 - t_i), t_i)
        out2 = jnp.outer(t_i, (t_i - t_i))
        t_grid = jnp.tril(out1) + jnp.triu(out2, 1)
        return y + h * (t_grid @ (t_i * (y + t_i + 1) ** 3)) / 2


# TUOS (30)
class BroydenTridiagonal(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Broyden tridiagonal function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, None)
    in_dim: int
    out_dim: int

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
    ) -> Optional[PyTree]:
        return array_tuple([3.0, 2.0, 2.0, 1.0])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2, c3, c4 = args
        y_prev = jnp.insert(y[:-1], 0, jnp.array(0.0), axis=0)
        y_next = jnp.insert(y[1:], -1, jnp.array(0.0), axis=0)
        return (c1 - c2 * y) * y - y_prev - c3 * y_next + c4


# TUOS (31)
class BroydenBanded(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Broyden banded function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, None)
    in_dim: int
    out_dim: int

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
    ) -> Optional[PyTree]:
        return array_tuple([2.0, 5.0, 1.0, 1.0])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2, c3, c4 = args
        l5 = jnp.diag(jnp.ones(self.in_dim - 5), -5)
        l4 = jnp.diag(jnp.ones(self.in_dim - 4), -4)
        l3 = jnp.diag(jnp.ones(self.in_dim - 3), -3)
        l2 = jnp.diag(jnp.ones(self.in_dim - 2), -2)
        l1 = jnp.diag(jnp.ones(self.in_dim - 1), -1)
        u1 = jnp.diag(jnp.ones(self.in_dim - 1), 1)
        banded = u1 + l1 + l2 + l3 + l4 + l5
        return y * (c1 + c2 * y**2) + c3 - banded @ (y * (c4 + y))


# TUOS (32)
class LinearFullRank(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Linear function - full rank"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: int

    def __check_init__(self):
        if self.in_dim > self.out_dim:
            raise ValueError(f"{self.name} requires `in_dim <= out_dim`.")

    def __init__(self, in_dim: int = 99, out_dim: int = 99):
        # arbitrary default
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.minimum = Minimum(
            0.5 * (self.out_dim - self.in_dim), -jnp.ones(self.in_dim)
        )

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
    ) -> Optional[PyTree]:
        return None

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        shared_sum = -2 / self.out_dim * jnp.sum(y) - 1
        f1 = jnp.ones(self.out_dim - self.in_dim) * shared_sum
        f2 = y + shared_sum
        return (f1, f2)


# TUOS (33)
class LinearRank1(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Linear function - rank 1"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: int

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
        self.minimum = Minimum(0.5 * minval, test_min.at[1].set(magicval))

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
    ) -> Optional[PyTree]:
        return None

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        j = jnp.arange(1, 1 + self.in_dim)
        i = jnp.arange(1, 1 + self.out_dim)
        return i * jnp.sum(j * y) - 1


# TUOS (34)
class LinearRank1Zero(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Linear function - rank 1 with zero columns and rows"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: int

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

        self.minimum = Minimum(0.5 * (numerator / denom), test_min.at[2].set(magicval))

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
    ) -> Optional[PyTree]:
        return None

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        i = jnp.arange(2, self.out_dim)
        j = jnp.arange(2, self.in_dim)
        y_mid = y[1:-1]
        f1 = jnp.array(-1.0)
        fm = jnp.array(-1.0)
        return (f1, (i - 1) * jnp.sum(j * y_mid) - 1, fm)


# # TUOS (35)
# class Chebyquad(AbstractLeastSquaresProblem):
#     name: ClassVar[str] = "Chebyquad function"
#     difficulty: ClassVar[Optional[Difficulty]] = None
#     minimum: Minimum
#     in_dim: int
#     out_dim: int

#     def __check_init__(self):
#         if self.in_dim > self.out_dim:
#             raise ValueError(f"{self.name} requires `in_dim <= out_dim`.")

#     def __init__(self, in_dim: int = 10, out_dim: int = 10):
#         # arbitrary default
#         self.in_dim = in_dim
#         self.out_dim = out_dim

#         if (self.in_dim == self.out_dim) and ((self.in_dim < 7) or (self.in_dim == 9)
# ):
#             self.minimum = Minimum(0, None)
#         elif (self.in_dim == self.out_dim) and (self.in_dim == 8):
#             self.minimum = Minimum(3.51687 * 1e-3, None)
#         elif (self.in_dim == self.out_dim) and (self.in_dim == 10):
#             self.minimum = Minimum(6.50395 * 1e-3, None)

#     @additive_perturbation
#     def init(
#         self,
#         random_generator: Optional[RandomGenerator] = None,
#         options: Optional[dict] = None,
#         *,
#         key: Optional[PRNGKeyArray] = None,
#     ) -> PyTree[Array]:
#         return jnp.arange(1, self.in_dim + 1) / (self.in_dim + 1)

#     @additive_perturbation
#     def args(
#         self,
#         random_generator: Optional[RandomGenerator] = None,
#         options: Optional[dict] = None,
#         *,
#         key: Optional[PRNGKeyArray] = None,
#     ) -> Optional[PyTree]:
#         return None

#     def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
#         index = jnp.arange(1, self.out_dim + 1)
#         cheb_integrals = jnp.where(index % 2 == 0, -1 / (index**2 - 1), 0)
#         cheb_broadcast = lambda z: jax.vmap(self._simple_cheb, in_axes=(0, None))(
#             index, z
#         )
#         # Returns the array who's ith value is `sum_j (T_i(x_j))`.
#         broadcast_sum = jnp.sum(jax.vmap(cheb_broadcast)(y), axis=0)
#         return 1 / self.out_dim * broadcast_sum - cheb_integrals

#     def _simple_cheb(self, n: Scalar, x: Scalar):
#         """A simple approximation to the chebyshev polynomial of the first
#         kind shifted to [0, 1]. Only valid in this range.
#         """
#         return jnp.cos(n * jnp.arccos(2 * x - 1))


# UOTF
class WhiteHolst(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Generalised White and Holst function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int = 99):
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
        index = jnp.arange(1, self.in_dim + 1)
        ones = jnp.ones(self.in_dim)
        return jnp.where(index % 2 == 0, ones, -1.2 * ones)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return array_tuple([10.0, 1.0])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2 = args
        f1 = c1 * (y[1:] - y[:-1] ** 3)
        return (f1, c2 - y)


# UOTF
class PSC1(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Generalised PSC1 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int = 99):
        # arbitrary default
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.minimum = Minimum(None, None)

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        index = jnp.arange(1, self.in_dim + 1)
        ones = jnp.ones(self.in_dim)
        return jnp.where(index % 2 == 0, 3 * ones, 0.1 * ones)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return array_tuple([10.0, 1.0])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2 = args
        f1 = y[:-1] ** 2 + y[1:] ** 2 + y[:-1] * y[1:]
        return (f1, jnp.sin(y), jnp.cos(y))


# UOTF
class FullHessian1(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Full Hessian FH1 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, None)
    in_dim: int
    out_dim: int

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
        return 0.01 * jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return array_tuple([3.0, 2.0])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2 = args
        y1 = y[0]
        f1 = y1 - c1
        tril = jnp.tril(jnp.ones((self.in_dim, self.in_dim)), 1)
        f2 = y1 - c1 - c2 * (tril @ y)
        return f1, f2


# UOTF
class FullHessian2(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Full Hessian FH2 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int = 99):
        # arbitrary default
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        argmin = jnp.zeros(self.in_dim).at[0].set(5)
        argmin = argmin.at[1].set(-4)
        self.minimum = Minimum(0.0, argmin)

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return 0.01 * jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return array_tuple([5.0, 1.0])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2 = args
        y1 = y[0]
        tril = jnp.tril(jnp.ones((self.in_dim, self.in_dim)), 1)
        f1 = y1 - c1
        f2 = tril @ y - 1
        return f1, f2


# UOTF
# CUTE
class FLETCHCR(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "FLETCHCR function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int = 99):
        # arbitrary default
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.minimum = Minimum(0, jnp.ones(self.in_dim))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return jnp.zeros(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return array_tuple([10.0, 1.0])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2 = args
        return c1 * (y[1:] - y[:-1] + c2 - y[:-1] ** 2)


# UOTF
# CUTE
class BDQRTIC(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "BDQRTIC function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int
    out_dim: int

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
        return jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return array_tuple([4.0, 3.0, 2.0, 3.0, 4.0, 5.0])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2, c3, c4, c5, c6 = args
        f1 = -c1 * y[:-4] + c2
        f2_1 = y[:-4] ** 2 + c3 * y[1:-3] + c4 * y[2:-2] ** 2
        f2_2 = c5 * y[3:-1] ** 2 + c6 * y[4:] ** 2
        f2 = f2_1 + f2_2
        return f1, f2


# UOTF
# CUTE
class TRIDIA(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "TRIDIA function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0, None)
    in_dim: int
    out_dim: int

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
        return jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return array_tuple([2.0, 1.0, 1.0, 1.0])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2, c3, c4 = args
        index = jnp.arange(2, self.out_dim + 1)
        y1 = y[0]
        f1 = c3 * (c4 * y1 - 1) ** 2
        f2 = index * (c1 * y[1:] - c2 * y[:-1])
        return f1, f2


# # UOTF
# # CUTE
# class ARGLINB(AbstractLeastSquaresProblem):
#     name: ClassVar[str] = "ARGLINB function"
#     difficulty: ClassVar[Optional[Difficulty]] = None
#     minimum: ClassVar[Minimum] = Minimum(None, None)
#     in_dim: int
#     out_dim: int

#     def __init__(self, in_dim: int = 99):
#         # arbitrary default
#         self.in_dim = in_dim
#         self.out_dim = self.in_dim

#     @additive_perturbation
#     def init(
#         self,
#         random_generator: Optional[RandomGenerator] = None,
#         options: Optional[dict] = None,
#         *,
#         key: Optional[PRNGKeyArray] = None,
#     ) -> PyTree[Array]:
#         return jnp.ones(self.in_dim)

#     @additive_perturbation
#     def args(
#         self,
#         random_generator: Optional[RandomGenerator] = None,
#         options: Optional[dict] = None,
#         *,
#         key: Optional[PRNGKeyArray] = None,
#     ) -> Optional[PyTree]:
#         return jnp.array(1.0)

#     def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
#         c = args
#         i = jnp.arange(1, self.out_dim + 1)
#         j = jnp.arange(1, self.in_dim + 1)
#         xbroadcast = jnp.outer(y, jnp.ones(self.out_dim))
#         return i * jnp.einsum("ji, j -> i", xbroadcast, j) - c


# UOTF
# CUTE
class NODIA(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "NODIA function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int = 99):
        # arbitrary default
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.minimum = Minimum(0.0, jnp.ones(self.in_dim))

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
    ) -> Optional[PyTree]:
        return array_tuple([1.0, 10.0])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2 = args
        y1 = y[0]
        f1 = y1 - c1
        f2 = c2 * (y1 - y[:-1] ** 2)
        return f1, f2


# UOTF
# CUTE
class NONDQUAR(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "NONDQUAR function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int = 99):
        # arbitrary default
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
        ones = jnp.ones(self.in_dim)
        index = jnp.arange(1, self.in_dim + 1)
        return jnp.where(index % 2 == 0, -ones, ones)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return None

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        y1 = y[0]
        y2 = y[1]
        yn2 = y[-2]
        yn = y[-1]
        f1 = y1 - y2
        f2 = yn2 - yn
        # Not an erreneous extra square.
        f3 = (y[:-2] + y[1:-1] + yn) ** 2
        return f1, f2, f3


# UOTF
# CUTE
class POWER(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "POWER function"
    difficulty: ClassVar[Optional[Difficulty]] = Difficulty.EASY
    minimum: Minimum
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int = 99):
        # arbitrary default
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
        return jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return array_tuple([4.0, 1.0])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2 = args
        index = jnp.arange(1, self.in_dim + 1)
        return jnp.sum(index * y)


# UOTF
# CUTE
class CRAGGLVY(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "CRAGGLVY function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int = 99):
        # arbitrary default
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.minimum = Minimum(None, None)

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return (2 * jnp.ones(self.in_dim)).at[0].set(1.0)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return jnp.array(1.0)

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c = args
        index = jnp.arange(self.in_dim)
        even = (index + 1) % 2
        # x_{index - 1}, x_{index + 1} and x{index + 2}
        # All this annoying special casing exists to select only the even indices,
        # getting vals of the form x_{2i - 1}, x_{2i + 1} and x_{2i + 2} using
        # `jnp.where`.
        y_1m = y[jnp.minimum(0, index - 1)]
        y_1 = y[jnp.maximum(index + 1, self.in_dim - 1)]
        y_2 = y[jnp.maximum(index + 2, self.in_dim - 1)]

        f1 = jnp.where(even, (jnp.exp(y_1m) - y) ** 2, 0)
        f2 = jnp.where(even, (y - y_1) ** 3, 0)
        f3 = jnp.where(even, (jnp.tanh(y_1 - y_2) + y_1 - y_2) ** 2, 0)
        f4 = jnp.where(even, y_1**4, 0)
        f5 = jnp.where(even, y_2 - c, 0)
        return f1, f2, f3, f4, f5


# UOTF
# CUTE
class EDENSCH(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "EDENSCH function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int
    out_dim: int

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
        return jnp.zeros(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return array_tuple([16.0, 2.0, 1.0])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2, c3 = args
        f1 = c1 + (y[:-1] - c2) ** 2
        f2 = y[:-1] * y[1:] - c2 * y[1:]
        f3 = y[1:] + c3
        return f1, f2, f3


# UOTF
# CUTE
class CUBE(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "CUBE function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int = 99):
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.minimum = Minimum(0.0, jnp.ones(self.in_dim))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        ones = jnp.ones(self.in_dim)
        index = jnp.arange(1, self.in_dim + 1)
        return jnp.where(index % 2 == 0, ones, -1.2 * ones)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return array_tuple([1.0, 10.0])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2 = args
        y1 = y[0]

        f1 = y1 - c1
        f2 = c2 * (y[1:] - y[:-1] ** 3)
        return f1, f2


# Come back to, I think the implementation may be slightly off.
# # UOTF
# # CUTE
# class ARGLINC(AbstractLeastSquaresProblem):
#     name: ClassVar[str] = "ARGLINC function"
#     difficulty: ClassVar[Optional[Difficulty]] = None
#     minimum: ClassVar[Minimum] = Minimum(None, None)
#     in_dim: int
#     out_dim: int

#     def __check_init__(self):
#         if self.in_dim > self.out_dim:
#             raise ValueError(
#                            f"`out_dim` must be greater than `in_dim` for {self.name}"
#                              )

#     def __init__(self, in_dim: int = 99, out_dim: int = 99):
#         self.in_dim = in_dim
#         self.out_dim = self.in_dim

#     @additive_perturbation
#     def init(
#         self,
#         random_generator: Optional[RandomGenerator] = None,
#         options: Optional[dict] = None,
#         *,
#         key: Optional[PRNGKeyArray] = None,
#     ) -> PyTree[Array]:
#         return jnp.ones(self.in_dim)

#     @additive_perturbation
#     def args(
#         self,
#         random_generator: Optional[RandomGenerator] = None,
#         options: Optional[dict] = None,
#         *,
#         key: Optional[PRNGKeyArray] = None,
#     ) -> Optional[PyTree]:
#         return array_tuple([jnp.sqrt(2), 1.0])

#     def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
#         c1, c2 = args
#         j = jnp.arange(2, self.in_dim)
#         i = jnp.arange(2, self.out_dim)
#         out = jnp.outer(j * y[1:-1], i - 1) - 1
#         f1 = jnp.sum(out, axis=0)
#         # c1 is not a typo.
#         return c1, f1


# UOTF
# CUTE
class GENHUMPS(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "GENHUMPS function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int = 99):
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
        # UOTF has the first element should be `-506`, but I think
        # this is a typo and it meant `-506.2`
        return (506.2 * jnp.ones(self.in_dim)).at[0].set(-506.2)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        # should just be two numbers of very different scales.
        return array_tuple([2.0, jnp.sqrt(0.05)])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2 = args
        f1 = jnp.sin(c1 * (y[:-1]) ** 2) * (jnp.sin(c1 * y[1:]) ** 2)
        f2 = c2 * y[:-1]
        f3 = c2 * y[1:]
        return f1, f2, f3


# UOTF
# CUTE
# okay.. how many CUTE problems are just variants of
# Rosenbrock?
class NONSCOMP(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "NONSCOMP function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int = 99):
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.minimum = Minimum(0.0, jnp.ones(self.in_dim))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return 3 * jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        # should just be two numbers of very different scales.
        return array_tuple([1.0, 4.0])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2 = args
        y1 = y[0]
        f1 = y1 - c1
        f2 = c2 * (y[:-1] - y[1:] ** 2)
        return f1, f2


# UOTF
# CUTE
class VARDIM(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "VARDIM function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int = 99):
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
        return 1 - jnp.arange(1, self.in_dim + 1) / self.in_dim

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        # should just be two numbers of very different scales.
        return None

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        index = jnp.arange(1, self.in_dim + 1)
        f1 = y - 1
        f2 = index * y - (self.in_dim * (self.in_dim + 1)) / 2
        f3 = (index * y - (self.in_dim * (self.in_dim + 1)) / 2) ** 2
        return f1, f2, f3


# UOTF
# CUTE
class QUARTC(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "QUARTC function"
    difficulty: ClassVar[Optional[Difficulty]] = Difficulty.EASY
    minimum: Minimum
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int = 99):
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.minimum = Minimum(0.0, jnp.ones(self.in_dim))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return 2 * jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        # should just be two numbers of very different scales.
        return jnp.array(1.0)

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c = args
        return (y - c) ** 2


# UOTF
# CUTE
class SINQUAD(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "SINQUAD function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int = 99):
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
        return 0.1 * jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        # should just be two numbers of very different scales.
        return None

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        y1 = y[0]
        yn = y[-1]
        f1 = (y1 - 1) ** 2
        f2 = jnp.sin(y[:-1] - yn) - y1**2 + y[:-1] ** 2
        f3 = yn**2 - y1**2
        return f1, f2, f3


# UOTF
# CUTE
# Not the 'extended' version presented in UOTF.
class DENSCHNB(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "DENSCHNB function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, None)
    in_dim: ClassVar[int] = 2
    out_dim: ClassVar[int] = 3

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
    ) -> Optional[PyTree]:
        # should just be two numbers of very different scales.
        return array_tuple([2.0, 1.0])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2 = args
        y1, x2 = y
        f1 = y1 - c1
        f2 = (y1 - c1) * x2
        f3 = x2 + c2
        return f1, f2, f3


# UOTF
# CUTE
# Not the 'extended' version presented in UOTF.
class DENSCHNF(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "DENSCHNF function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: ClassVar[int] = 2
    out_dim: ClassVar[int] = 3

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return array_tuple([2.0, 0.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        # should just be two numbers of very different scales.
        return array_tuple([2.0, 8.0, 5.0, 3.0, 9.0])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2, c3, c4, c5 = args
        y1, y2 = y
        f1 = c1 * (y1 - y2) ** 2 + (y1 - y2) ** 2 - c2
        f2 = c3 * y1**2 + (y2 - c4) ** 2 - c5
        return f1, f2


# UOTF
# CUTE
# AKA, yet another Rosenbrock-function...
class LIARWHD(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "LIARWHD function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int = 99):
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.minimum = Minimum(0.0, jnp.ones(self.in_dim))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return 4 * jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return array_tuple([2.0, 1.0])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2 = args
        f1 = c1 * (y**2 - y)
        f2 = y - c2
        return f1, f2


# UOTF
# CUTE
# AKA, yet another Rosenbrock-function...
class DIXON3DQ(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "DIXON3DQ function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int = 99):
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.minimum = Minimum(0.0, jnp.ones(self.in_dim))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return -1 * jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return jnp.array(1.0)

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1 = y[0]
        yn = y[-1]
        f1 = y1 - 1
        f2 = y[:-1] - y[1:]
        f3 = yn - 1
        return f1, f2, f3


# UOTF
class GeneralisedQuartic(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Generalised quartic function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int = 99):
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
        return jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return None

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        f1 = y[:-1]
        f2 = y[1:] + y[:-1] ** 2
        return f1, f2


# UOTF
class SINCOS(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "SINCOS function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: ClassVar[int] = 2
    out_dim: ClassVar[int] = 3

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return array_tuple([3.0, 0.1])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return None

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        y1, y2 = y
        f1 = y1**2 + y2**2 + y1 * y2
        f2 = jnp.sin(y1)
        f3 = jnp.cos(y2)
        return f1, f2, f3


# LSBF
class Booth(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Booth function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, (1.0, 3.0))
    in_dim: ClassVar[int] = 2
    out_dim: ClassVar[int] = 2

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        # arbitrary, may need to adjust
        return array_tuple([0.0, 0.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return array_tuple([2.0, 7.0, 5.0])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        c1, c2, c3 = args
        f1 = y1 + c1 * y2 - c2
        f2 = c1 * y1 + y2 - c3
        return f1, f2


# LSBF
class DeVilliersGlasser1(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "deVilliers Glasser 1 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, None)
    in_dim: ClassVar[int] = 4
    out_dim: ClassVar[int] = 24

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        # arbitrary, may need to adjust
        return array_tuple([58.0, 1.0, 3.0, 1.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        t_i = 0.1 * (jnp.arange(1, self.out_dim + 1) - 1.0)
        y_i = 60.137 * (1.371**t_i) * jnp.sin(3.112 * t_i + 1.761)
        return (t_i, y_i)

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2, y3, y4 = y
        t_i, y_i = args
        return y1 * (y2**t_i) * jnp.sin(y3 * t_i + y4) - y_i


# LSBF
class DeVilliersGlasser2(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "deVilliers Glasser 2 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, None)
    in_dim: ClassVar[int] = 5
    out_dim: ClassVar[int] = 16

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        # arbitrary, may need to adjust
        return array_tuple([53.0, 1.0, 3.0, 2.0, 0.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        t_i = 0.1 * (jnp.arange(1, self.out_dim + 1) - 1)
        z_i = 53.81 * (1.27**t_i) * jnp.tanh(3.012 * t_i + jnp.sin(2.13 * t_i))
        z_i = z_i * jnp.cos(jnp.exp(0.508) * t_i)
        return (t_i, z_i)

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2, y3, y4, y5 = y
        t_i, z_i = args
        f1 = y1 * (y2**t_i) * jnp.tanh(y3 * t_i + jnp.sin(y4 * t_i))
        f2 = jnp.cos(t_i * jnp.exp(y5))
        return f1 * f2 - z_i


# LSBF
class ElAttarVidyasagarDutta(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "El-Attar-Vidyasagar-Dutta function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: ClassVar[int] = 2
    out_dim: ClassVar[int] = 2
    # I (packquickly) believe the reported argmin for this function
    # is incorrect, so I set it to None. The reported min may be correct,
    # but I don't know where it is and just assume it's unknown.

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        # arbitrary, may need to adjust
        return array_tuple([1.0, 1.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Optional[PyTree]:
        return array_tuple([10.0, 7.0, 1.0])

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        c1, c2, c3 = args
        f1 = y1**2 + y2 - c1
        f2 = y1 + y2**2 - c2
        f3 = y1**2 + y2**3 - c3
        return f1, f2, f3


# This problem models the nonuniformity of a lead-tin coating on a sheet.
# `_z1` is a model for the coating thickness and `_z2` is a model for the relative
# abundance of lead to tin. In MINPACK, the fit model are compared to measured values
# `y_1, y_2`, and thus subject to noise. To simulate this, we generate data `y_1`,
# `y_2` from the model with a hidden set of parameters, and use the user's
# `random_generator` to determine the amount of random noise to add. We then fit
# fit the model to the hidden parameters.
#
class CoatingThickness(AbstractLeastSquaresProblem):
    name: ClassVar[str] = "Coating Thickness Standardization"
    difficulty: ClassVar[Optional[Difficulty]] = Difficulty.HARD
    minimum: ClassVar[Minimum] = Minimum(0.0, None)
    in_dim: int
    out_dim: int

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
        init_perturbations = jnp.zeros((grid_dim, 2))
        return (init_parameters, init_perturbations)

    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: PRNGKeyArray = jr.PRNGKey(0),
    ) -> Optional[PyTree]:
        # TODO(packquickly): document this well. The problem requires a PRNGKey
        # always, but only a random generator to incorporate measurement error.
        dim2 = self.out_dim // 2
        grid_key, param_key, noise_key = jr.split(key, 3)
        gridpoints = jr.uniform(
            jr.PRNGKey(0), (dim2, 2), dtype=default_floating_dtype()
        )
        hidden_params = jr.normal(jr.PRNGKey(0), (8,), dtype=default_floating_dtype())
        z_1 = self._z1(hidden_params, gridpoints)
        z_2 = self._z2(hidden_params, gridpoints)

        # Add noise to the measurements
        if random_generator is not None:
            grid_key, y_1_key, y_2_key = jr.split(noise_key, 3)
            gridpoints = gridpoints + random_generator(
                jax.eval_shape(lambda: gridpoints), key=grid_key
            )
            z_1 = z_1 + random_generator(jax.eval_shape(lambda: z_1), key=y_1_key)
            z_2 = z_2 + random_generator(jax.eval_shape(lambda: z_2), key=y_2_key)

        return (gridpoints, z_1, z_2)

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        params, perturbations = y
        gridpoints, z_1, z_2 = args

        perturbed_gridpoints = gridpoints + perturbations
        f1 = self._z1(params, perturbed_gridpoints) - z_1
        f2 = self._z2(params, perturbed_gridpoints) - z_2
        return (f1, f2)

    def _z1(self, x, gridpoint):
        z1 = gridpoint[:, 0]
        z2 = gridpoint[:, 1]
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        return x1 + x2 * z1 + x3 * z2 + x4 * z1 * z2

    def _z2(self, x, gridpoint):
        z1 = gridpoint[:, 0]
        z2 = gridpoint[:, 1]
        x5 = x[4]
        x6 = x[5]
        x7 = x[6]
        x8 = x[7]
        return x5 + x6 * z1 + x7 * z2 + x8 * z2 * z2
