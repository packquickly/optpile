from typing import cast, ClassVar, Optional

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, PyTree

from .base import AbstractMinimisationProblem, Difficulty, Minimum
from .custom_types import RandomGenerator
from .misc import (
    additive_perturbation,
    array_tuple,
)


#
# See the long comment at the start of lstsq_problems.
#
# Most of the test problems are least-squares problems.
#


# UOTF
class Raydan1(AbstractMinimisationProblem):
    name: ClassVar[str] = "Raydan 1 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int = 99

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
        t_i = jnp.arange(1, self.in_dim + 1) / 10
        return [t_i]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        t_i = args
        return jnp.sum(t_i * (jnp.exp(y) - y))


# UOTF
class Diagonal2(AbstractMinimisationProblem):
    name: ClassVar[str] = "Diagonal 2 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int = 99

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return 1 / jnp.arange(1, self.in_dim + 1)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [None]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        i = jnp.arange(1, self.in_dim + 1)
        return jnp.sum(jnp.exp(y) - y / i)


# UOTF
class Diagonal3(AbstractMinimisationProblem):
    name: ClassVar[str] = "Diagonal 3 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int = 99

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

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        i = jnp.arange(1, self.in_dim + 1)
        return jnp.sum(jnp.exp(y) - i * jnp.sin(y))


# UOTF
class Hager(AbstractMinimisationProblem):
    name: ClassVar[str] = "Hager function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int = 99

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

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        i = jnp.arange(1, self.in_dim + 1)
        return jnp.sum(jnp.exp(y) - jnp.sqrt(i) * y)


# UOTF
class Diagonal5(AbstractMinimisationProblem):
    name: ClassVar[str] = "Diagonal 5 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int = 99

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return 1.1 * jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [None]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        return jnp.sum(jnp.log(jnp.exp(y) + jnp.exp(-y)))


# UOTF
class QuadraticQF1(AbstractMinimisationProblem):
    name: ClassVar[str] = "Quadratic QF1 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int

    def __init__(self, in_dim: int = 99):
        # arbitrary default
        self.in_dim = in_dim

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
        return [None]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        i = jnp.arange(1, self.in_dim + 1)
        ym = y[-1]
        return 0.5 * jnp.sum(i * y**2) - ym


# UOTF
class QuadraticQF2(AbstractMinimisationProblem):
    name: ClassVar[str] = "Quadratic QF2 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int = 99

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
        return [jnp.array(1.0)]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        i = jnp.arange(1, self.in_dim + 1)
        ym = y[-1]
        return 0.5 * jnp.sum(i * (y**2 - 1)) - ym


# UOTF
# CUTE
class FLETCBV3(AbstractMinimisationProblem):
    name: ClassVar[str] = "FLETCBV 3 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int = 99

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
        return [array_tuple([1e-8, 1.0])]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        p, c = args
        y1 = y[0]
        yn = y[-1]
        h = 1 / (self.in_dim + 1)
        f1 = 0.5 * p * (y1**2 + yn**2)
        f2 = 0.5 * p * jnp.sum((y[:-1] - y[1:]) ** 2)
        f3 = jnp.sum(
            (p * (h**2 + 2) / (h**2)) * y + (c * p) / (h**2) * jnp.cos(y)
        )
        return f1 + f2 - f3


# UOTF
# CUTE
class ARWHEAD(AbstractMinimisationProblem):
    name: ClassVar[str] = "ARWHEAD function"
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
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([4.0, 3.0])]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2 = args
        yn = y[-1]
        f1 = jnp.sum(-c1 * y[:-1] + c2)
        f2 = jnp.sum((y[:-1] ** 2 + yn**2) ** 2)
        return f1 + f2


# UOTF
# CUTE
class EG2(AbstractMinimisationProblem):
    name: ClassVar[str] = "EG2 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int = 99

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
        return [jnp.array(1.0)]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1 = y[0]
        yn = y[-1]
        f1 = jnp.sum(jnp.sin(y1 + y[:-1] ** 2 - 1))
        return f1 + 0.5 * jnp.sin(yn**2)


# UOTF
# CUTE
class ENGVAL1(AbstractMinimisationProblem):
    name: ClassVar[str] = "ENGVA 1 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int = 99

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
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([4.0, 3.0])]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2 = args
        f1 = jnp.sum((y[:-1] ** 2 + y[1:] ** 2) ** 2)
        f2 = jnp.sum(-c1 * y[:-1] + 3)
        return f1 + f2


# UOTF
# CUTE
class CURLY20(AbstractMinimisationProblem):
    name: ClassVar[str] = "CURLY 20 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int

    def __check_init__(self):
        if self.in_dim < 20:
            raise ValueError(f"{self.name} requires `in_dim>20`")

    def __init__(self, in_dim: int = 99):
        self.in_dim = in_dim
        self.minimum = Minimum(0.0, jnp.zeros(self.in_dim))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return jnp.ones(self.in_dim) / (1000 * (self.in_dim + 1))

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        # TODO(packquickly): this one is a natural candidate for multiple
        # `args`, but it requires an integer `random_generator` to support
        # randomness.
        return [None]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        until_k = jnp.triu(jnp.ones((20, 20))) @ y[:20]
        after_k = jnp.tril(jnp.ones((self.in_dim - 20, self.in_dim - 20))) @ y[20:]
        q = jnp.concatenate((until_k, after_k), axis=0)
        return jnp.sum(q**4 - 20 * q**2 - 0.1 * q)


# UOTF
class DIXMAANA(AbstractMinimisationProblem):
    name: ClassVar[str] = "DIXMAANA-DIXMANNL functions"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int

    def __check_init__(self):
        if self.in_dim % 3 != 0:
            raise ValueError(f"{self.name} requires `in_dim` divisible by 3.")

    def __init__(self, in_dim: int = 99):
        self.in_dim = in_dim

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
    ) -> list[Optional[PyTree[Array]]]:
        A = array_tuple([1.0, 0.0, 0.125, 0.125, 0, 0, 0, 0])
        B = array_tuple([1.0, 0.0625, 0.0625, 0.0625, 0, 0, 0, 1])
        C = array_tuple([1.0, 0.125, 0.125, 0.125, 0, 0, 0, 0])
        D = array_tuple([1.0, 0.26, 0.26, 0.26, 0, 0, 0, 0])
        E = array_tuple([1.0, 0.0, 0.126, 0.125, 1, 0, 0, 1])
        F = array_tuple([1.0, 0.0625, 0.0625, 0.625, 1, 0, 0, 1])
        G = array_tuple([1.0, 0.125, 0.125, 0.125, 1, 0, 0, 1])
        H = array_tuple([1.0, 0.26, 0.26, 0.26, 1, 0, 0, 1])
        I = array_tuple([1.0, 0.0, 0.125, 0.125, 2, 0, 0, 2])
        J = array_tuple([1.0, 0.0625, 0.0625, 0.625, 2, 0, 0, 2])
        K = array_tuple([1.0, 0.125, 0.125, 0.125, 2, 0, 0, 2])
        L = array_tuple([1.0, 0.26, 0.26, 0.26, 2, 0, 0, 2])
        return [A, B, C, D, E, F, G, H, I, J, K, L]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2, c3, c4, c5, c6, c7, c8 = args
        m = self.in_dim // 3
        index = jnp.arange(1, self.in_dim + 1)

        f1 = 1 + jnp.sum(c1 * y**2 * (index / self.in_dim) ** c5)
        f2 = jnp.sum(
            c2 * (y[:-1] ** 2 * (y[1:] + y[1:] ** 2)) * (index[:-1] / self.in_dim) ** c6
        )
        f3 = c3 * jnp.sum(
            (y[:-m] ** 2) * (y[m:] ** 4) * (index[:-m] / self.in_dim) ** c7
        )
        f4 = c4 * jnp.sum(
            y[: -2 * m] * y[2 * m :] * (index[: -2 * m] / self.in_dim) ** c8
        )
        return f1 + f2 + f3 + f4


# UOTF
# CUTE
class INDEF(AbstractMinimisationProblem):
    name: ClassVar[str] = "INDEF function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int = 99

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
        return [array_tuple([2.0])]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1 = y[0]
        yn = y[-1]

        f1 = jnp.sum(y)
        f2 = 0.5 * jnp.sum(2 * y[1:-1] - yn - y1)
        return f1 + f2


# UOTF
# CUTE
class EXPLIN1(AbstractMinimisationProblem):
    name: ClassVar[str] = "EXPLIN1 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int = 99

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
    ) -> list[Optional[PyTree[Array]]]:
        # should just be two numbers of very different scales.
        return [array_tuple([0.1, 10])]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2 = args
        jnp.sum(jnp.exp(c1 * y[:-1] * y[1:]))


# UOTF
# CUTE
class HARKERP2(AbstractMinimisationProblem):
    name: ClassVar[str] = "HARKERP2 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int = 99

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return jnp.arange(1, self.in_dim + 1)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [jnp.array(2.0)]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c = args
        f1 = jnp.sum(y**2)
        f2 = jnp.sum(y + (1 / c) * y**2)
        f3 = c * (jnp.triu(jnp.ones((self.in_dim, self.in_dim)), 1) @ y) ** 2
        return f1 - f2 + f3


# UOTF
# CUTE
class MCCORMCK(AbstractMinimisationProblem):
    name: ClassVar[str] = "MCCORMCK function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int = 99

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
        return [array_tuple([1.5, 2.5, 1.0])]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2, c3 = args
        tmp = -c1 * y[:-1] + c2 * y[1:] + c3 + (y[:-1] - y[1:]) ** 2
        return jnp.sum(tmp + jnp.sin(y[:-1] + y[1:]))


# UOTF
# CUTE
class Diagonal6(AbstractMinimisationProblem):
    name: ClassVar[str] = "Diagonal 6 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int = 99

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
        return [jnp.array(1.0)]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        return jnp.sum(jnp.exp(y) - (1 - y))


# UOTF
# CUTE
class COSINE(AbstractMinimisationProblem):
    name: ClassVar[str] = "COSINE function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int = 99

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
        return [jnp.array(0.5)]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c = args
        return jnp.sum(jnp.cos(-c * y[1:] + y[:-1] ** 2))


# UOTF
# CUTE
class SINE(AbstractMinimisationProblem):
    name: ClassVar[str] = "SINE function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int = 99

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
        return [jnp.array(0.5)]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c = args
        return jnp.sum(jnp.sin(-c * y[1:] + y[:-1] ** 2))


# UOTF
# CUTE
class HIMMELBG(AbstractMinimisationProblem):
    name: ClassVar[str] = "HIMMELBG function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: ClassVar[int] = 2

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return array_tuple([1.5, 1.5])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([2.0, 3.0])]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        c1, c2 = args
        return (c1 * y1**2 + c2 * y2**2) * jnp.exp(-y1 - y2)


# UOTF
# CUTE
class HIMMELH(AbstractMinimisationProblem):
    name: ClassVar[str] = "HIMMELH function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: ClassVar[int] = 2

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        return array_tuple([1.5, 1.5])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([3.0, 2.0])]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        c1, c2 = args
        return -c1 * y1 - c2 * y2 + c2 + y1**3 + y2**2


# UOTF
class Diagonal7(AbstractMinimisationProblem):
    name: ClassVar[str] = "Diagonal 8 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int = 99

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
        return [jnp.array(2.0)]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c = args
        return jnp.sum(jnp.exp(y) - c * y - y**2)


# UOTF
class Diagonal8(AbstractMinimisationProblem):
    name: ClassVar[str] = "Diagonal 8 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int = 99

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
        return [jnp.array(2.0)]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c = args
        return jnp.sum(y * jnp.exp(y) - c * y - y**2)


# UOTF
class Diagonal9(AbstractMinimisationProblem):
    name: ClassVar[str] = "Diagonal 9 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int = 99

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
        return [jnp.array(1e4)]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c = args
        index = jnp.arange(1, self.in_dim)
        yn = y[-1]
        return jnp.sum(jnp.exp(y[:-1]) - index * y[:-1]) + c * yn


# UOTF
class FullHessian3(AbstractMinimisationProblem):
    name: ClassVar[str] = "Full Hessian FH3 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: int = 99

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
        return [jnp.array(2.0)]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c = args
        f1 = jnp.sum(y) ** 2
        f2 = jnp.sum(y * jnp.exp(y) - c * y - y**2)
        return f1 + f2


# LSBF
class Ackley1(AbstractMinimisationProblem):
    name: ClassVar[str] = "Ackley 1 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int

    def __init__(self, in_dim: int = 99):
        self.in_dim = in_dim
        self.minimum = Minimum(0, jnp.zeros(self.in_dim))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        # arbitrary, may need to adjust
        return jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [jnp.array(20.0)]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c = args
        invdim = 1 / self.in_dim
        f1 = -c * jnp.exp(-(1 / (10 * c)) * jnp.sqrt(invdim * jnp.sum(y**2)))
        f2 = jnp.exp(invdim * jnp.sum(jnp.cos(2 * jnp.pi * y)))
        return f1 - f2 + 20 + jnp.exp(1)


# LSBF
class Ackley2(AbstractMinimisationProblem):
    name: ClassVar[str] = "Ackley 2 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(-200.0, (0.0, 0.0))
    in_dim: ClassVar[int] = 2

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
    ) -> list[Optional[PyTree[Array]]]:
        return [jnp.array(200.0)]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        c = args
        return -c * jnp.exp(-(1 / c) * jnp.sqrt(y1**2 + y2**2))


# LSBF
class Ackley3(AbstractMinimisationProblem):
    name: ClassVar[str] = "Ackley 3 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: ClassVar[int] = 2
    # The listed function and minimum in LSBF is incorrect.
    # The function is adjusted, but the init is just `None`.

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
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([200.0, 5.0, 3.0])]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        c1, c2, c3 = args
        f1 = -c1 * jnp.exp(-(1 / c1) * jnp.sqrt(y1**2 + y2**2))
        f2 = -c2 * jnp.exp(jnp.cos(c3 * y1) + jnp.sin(c3 * y2))
        return f1 + f2


# LSBF
class Adjiman(AbstractMinimisationProblem):
    name: ClassVar[str] = "Adjiman function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(-2.02181, (2.0, 0.10578))
    in_dim: ClassVar[int] = 2

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
    ) -> list[Optional[PyTree[Array]]]:
        return [None]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        y1, y2 = y
        return jnp.cos(y1) * jnp.sin(y2) - y1 / (y2**2 + 1)


# LSBF
class Alpine2(AbstractMinimisationProblem):
    name: ClassVar[str] = "Alpine 2 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int

    def __init__(self, in_dim: int = 99):
        self.in_dim = in_dim
        self.minimum = Minimum(2.808**self.in_dim, 7.917 * jnp.ones(in_dim))

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
    ) -> list[Optional[PyTree[Array]]]:
        return [None]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        return jnp.prod(jnp.sqrt(y) * jnp.sin(y))


# LSBF
class Bird(AbstractMinimisationProblem):
    name: ClassVar[str] = "Bird function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(-106.764537, (4.70104, 3.15294))
    in_dim: ClassVar[int] = 2

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
    ) -> list[Optional[PyTree[Array]]]:
        return [None]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        y1, y2 = y
        f1 = jnp.sin(y1) * jnp.exp((1 - jnp.cos(y2)) ** 2)
        f2 = jnp.cos(y2) * jnp.exp((1 - jnp.sin(y1)) ** 2)
        f3 = (y1 - y2) ** 2
        return f1 + f2 + f3


# LSBF
# The function listed in LSBF is incorrect, this is the
# corrected version.
class Bohachevsky2(AbstractMinimisationProblem):
    name: ClassVar[str] = "Bohachevsky 2 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0, (0.0, 0.0))
    in_dim: ClassVar[int] = 2

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        # arbitrary, may need to adjust
        return array_tuple([10.0, -10.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([2.0, 3.0, 4.0])]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        c1, c2, c3 = args
        f1 = y1**2 + c1 * y2**2
        f2 = (c2 / 10) * jnp.cos(c2 * jnp.pi * y1) * jnp.cos(c3 * jnp.pi * y2)
        return f1 - f2 + (c2 / 10)


# LSBF
class Bohachevsky3(AbstractMinimisationProblem):
    name: ClassVar[str] = "Bohachevsky 3 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0, (0.0, 0.0))
    in_dim: ClassVar[int] = 2

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        # arbitrary, may need to adjust
        return array_tuple([10.0, -10.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([2.0, 3.0, 4.0])]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        c1, c2, c3 = args
        f1 = y1**2 + c1 * y2**2
        f2 = (c1 / 10) * jnp.cos(c2 * jnp.pi * y1 + c3 * jnp.pi * y2)
        return f1 - f2 + (c1 / 10)


# LSBF
class BraninRCOS(AbstractMinimisationProblem):
    name: ClassVar[str] = "Branin RCOS function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.3978873, (jnp.pi, 2.275))
    in_dim: ClassVar[int] = 2

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        # arbitrary, may need to adjust
        return array_tuple([0.0, 5.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([5.1, 4.0, 5.0, 6.0, 10.0, 8.0])]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        c1, c2, c3, c4, c5, c6 = args
        f1 = (y2 - (c1 * y1**2) / (c2 * jnp.pi**2) + (c3 * y1) / jnp.pi - c4) ** 2
        f2 = c5 * (1 - (1 / (c6 * jnp.pi))) * jnp.cos(y1) + c5
        return f1 + f2


# LSBF
# The function called `BraninRCOS2` is not consistent, and the implementation
# in LSBF does not match their min/argmin. Keeping this implementation as
# a test function with the name from LSBF, but it be a footgun to users
# who have a different understanding of what the Branin RCOS 2 function is.
class BraninRCOS2(AbstractMinimisationProblem):
    name: ClassVar[str] = "Branin RCOS 2 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: ClassVar[int] = 2

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        # arbitrary, may need to adjust
        return array_tuple([5, 5.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([5.1, 4.0, 5.0, 6.0, 10.0, 8.0])]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        c1, c2, c3, c4, c5, c6 = args
        f1 = (y2 - (c1 * y1**2) / (c2 * jnp.pi**2) + (c3 * y1) / jnp.pi - c4) ** 2
        f2 = c5 * (1 - 1 / (c6 * jnp.pi)) * jnp.cos(y1) * jnp.cos(y2)
        f3 = jnp.log(y1**2 + y2**2 + 1)
        return f1 + f2 * f3 + c5


# LSBF
class CamelThree(AbstractMinimisationProblem):
    name: ClassVar[str] = "Camel three hump function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, (0.0, 0.0))
    in_dim: ClassVar[int] = 2

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        # arbitrary, may need to adjust
        return array_tuple([2.5, 2.5])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([2.0, 1.05, 6])]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        c1, c2, c3 = args
        return c1 * y1**2 - c2 * y1**4 + (y1**6) / c3 + y1 * y2 + y2**2


# LSBF
class CamelSix(AbstractMinimisationProblem):
    name: ClassVar[str] = "Camel six hump function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(-1.0316284, (0.0898, -0.7126))
    in_dim: ClassVar[int] = 2

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
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([4.0, 2.1, 3.0])]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        c1, c2, c3 = args
        f1 = (c1 - c2 * y1**2 + (y1**4) / c3) * (y1**2)
        f2 = y1 * y2 + (c1 * y2**2 - c1) * y2**2
        return f1 + f2


# LSBF
# A variant not actually in LSBF, as their implementation has a min
# at `-inf` when they claim it is at -2000. This is a variant from
# http://al-roomi.org/benchmarks/unconstrained/2-dimensions/111-chen-s-bird-function
# ... which also marks the minimum incorrectly.
class ChenBird(AbstractMinimisationProblem):
    name: ClassVar[str] = "Chen bird function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: ClassVar[int] = 2

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        # arbitrary, may need to adjust
        return array_tuple([-4.0, 1.5])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([1e-3, 1.0, 0.5])]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        c1, c2, c3 = args
        f1 = -(c1**2) / (c1**2 + (y1**2 + y2**2 - c2))
        f2 = -(c1**2) / (c1**2 + (y1**2 + y2**2 - c3))
        f3 = -(c1**2) / (c1**2 + (y1 - y2) ** 2)
        return f1 + f2 + f3


# LSBF
class Chichinadze(AbstractMinimisationProblem):
    name: ClassVar[str] = "Chichinadze function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(-42.944387, (6.189866, 0.5))
    in_dim: ClassVar[int] = 2

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
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([12.0, 11.0, 10.0, 8.0, 5.0, 0.5])]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        c1, c2, c3, c4, c5, c6 = args
        f1 = y1**2 - c1 * y1 + c2
        f2 = c3 * jnp.cos(jnp.pi * y1 * 0.5) + c4 * jnp.sin(c5 * jnp.pi * y1 * 0.5)
        f3 = (1 / c5) * (c5**c6) * (1 / jnp.exp(-c6 * (y2 - c6) ** 2))
        return f1 + f2 - f3


# LSBF
class Colville(AbstractMinimisationProblem):
    name: ClassVar[str] = "Colville function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, (1.0, 1.0, 1.0, 1.0))
    in_dim: ClassVar[int] = 4

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array]:
        # arbitrary, may need to adjust
        return array_tuple([0.0, 0.0, 0.0, 0.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([100.0, 90.0, 10.1, 19.8])]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2, y3, y4 = y
        c1, c2, c3, c4 = args

        f1 = c1 * (y1 - y2**2) ** 2 + (1 - y1) ** 2
        f2 = c2 * (y4 - y3**2) ** 2 + (1 - y3) ** 2
        f3 = c3 * ((y2 - 1) ** 2 + (y4 - 1) ** 2)
        f4 = c4 * (y2 - 1) * (y4 - 1)
        return f1 + f2 + f3 + f4


# LSBF
# Typo in LSBF, this function usually has the degenerate
# case specialised to 0.
class Csendes(AbstractMinimisationProblem):
    name: ClassVar[str] = "Csendes function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int

    def __init__(self, in_dim: int = 99):
        self.in_dim = in_dim
        self.minimum = Minimum(0.0, jnp.zeros(self.in_dim))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: PRNGKeyArray = jr.PRNGKey(0),
    ) -> PyTree[Array]:
        # arbitrary, may need to adjust
        return jr.uniform(key, (self.in_dim,), minval=-1, maxval=1)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [jnp.array(2.0)]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c = args
        y_prod_nonzero = jnp.abs(jnp.prod(y)) > jnp.finfo(y.dtype).eps
        # Make Pyright happy
        safe_y = cast(Array, jnp.where(y_prod_nonzero, y, 1))
        f1 = jnp.where(y_prod_nonzero, jnp.sum((y**6) * (c + jnp.sin(1 / safe_y))), 0)
        return f1


# LSBF
# Typo in LSBF, this function usually has the degenerate
# case specialised to 0.
class Damavandi(AbstractMinimisationProblem):
    name: ClassVar[str] = "Damavandi function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, (2.0, 2.0))
    in_dim: ClassVar[int] = 2

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: PRNGKeyArray,
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
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([2.0, 7.0])]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        c1, c2 = args
        numerator = jnp.sin(jnp.pi * (y1 - c1) * jnp.sin(jnp.pi * (y2 - c1)))
        denominator = (jnp.pi**2) * (y1 - c1) * (y2 - c1)
        denom_nonzero = jnp.abs(denominator) > jnp.finfo(denominator.dtype).eps
        denom_safe = jnp.where(denom_nonzero, denominator, 1)
        f1 = jnp.where(denom_nonzero, numerator / denom_safe, 0.0)
        f2 = c1 + (y1 - c2) ** 2 + c1 * (y2 - c2) ** 2
        return f1 * f2


# LSBF
# LSBF marked the minimum incorrectly.
class DeckkersAarts(AbstractMinimisationProblem):
    name: ClassVar[str] = "Decckers-Aarts function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(-24771.094, (0.0, -15.0))
    in_dim: ClassVar[int] = 2

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: PRNGKeyArray,
    ) -> PyTree[Array]:
        # arbitrary, may need to adjust
        return array_tuple([5.0, 0.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [jnp.array(10.0)]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        c = args
        f1 = (c**5) * (y1**2) + y2**2 - (y1**2 + y2**2) ** 2
        f2 = (1 / (c**5)) * (y1**2 + y2**2) ** 4
        return f1 + f2


# LSBF
class Easom(AbstractMinimisationProblem):
    name: ClassVar[str] = "Easom function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(-1.0, (jnp.pi, jnp.pi))
    in_dim: ClassVar[int] = 2

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: PRNGKeyArray = None,
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
    ) -> list[Optional[PyTree[Array]]]:
        return [None]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        y1, y2 = y
        f1 = -jnp.cos(y1) * jnp.cos(y2)
        f2 = jnp.exp(-((y1 - jnp.pi) ** 2) - (y2 - jnp.pi) ** 2)
        return f1 * f2


# LSBF
# Typo in minimum in LSBF
class EggHolder(AbstractMinimisationProblem):
    name: ClassVar[str] = "Egg Holder function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(-959.64, jnp.array([512.0, 404.2319]))
    in_dim: int = 99
    # TODO(packquickly): decide if supporting a minimum with an argmin
    # of a specific dimension is okay. This is particularly awkward
    # becase this is a 2-dimensional minimum, but a 99 dimensional default.

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: PRNGKeyArray = None,
    ) -> PyTree[Array]:
        # arbitrary, may need to adjust
        return 300 * jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [jnp.array(47.0)]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c = args
        f1 = -(y[1:] + c) * jnp.sin(jnp.sqrt(jnp.abs(y[1:] + 0.5 * y[:-1] + c)))
        f2 = y[:-1] * jnp.sin(jnp.sqrt(jnp.abs(y[:-1] - (y[1:] + c))))
        return jnp.sum(f1 - f2)


# LSBF
class Griewank(AbstractMinimisationProblem):
    name: ClassVar[str] = "Griewank function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int

    def __init__(self, in_dim: int = 99):
        self.in_dim = 99
        self.minimum = Minimum(0.0, jnp.zeros(self.in_dim))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: PRNGKeyArray = None,
    ) -> PyTree[Array]:
        # arbitrary, may need to adjust
        return jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [jnp.array(4000.0)]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c = args
        index = jnp.arange(1, self.in_dim + 1)
        return jnp.sum(y**2 / c) - jnp.prod(jnp.cos(y / jnp.sqrt(index))) + 1


# LSBF
class Hosaki(AbstractMinimisationProblem):
    name: ClassVar[str] = "Hosaki function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(-2.3458, (4.0, 2.0))
    in_dim: ClassVar[int] = 2

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: PRNGKeyArray = None,
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
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([1.0, 8.0, 7.0, 7 / 3, 0.25])]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        c1, c2, c3, c4, c5 = args
        f1 = c1 - c2 * y1 + c3 * y1**2 - c4 * y1**3 + c5 * y1**4
        f2 = (y2**2) * jnp.exp(-y2)
        return f1 * f2


# LSBF
# Typo in LSBF, sign flip on their minimum.
class Keane(AbstractMinimisationProblem):
    name: ClassVar[str] = "Keane function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.673668, (0.0, 1.39325))
    in_dim: ClassVar[int] = 2

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: PRNGKeyArray = None,
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
    ) -> list[Optional[PyTree[Array]]]:
        return [None]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        y1, y2 = y
        return ((jnp.sin(y1 - y2) ** 2) * jnp.sin(y1 + y2) ** 2) / (
            jnp.sqrt(y1**2 + y2**2)
        )


# LSBF
class Mishra2(AbstractMinimisationProblem):
    name: ClassVar[str] = "Mishra 2 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int = 99

    def __init__(self, in_dim: int = 99):
        self.in_dim = in_dim
        self.minimum = Minimum(2.0, jnp.ones(self.in_dim))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: PRNGKeyArray = None,
    ) -> PyTree[Array]:
        # arbitrary, may need to adjust
        return 0.5 * jnp.zeros(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [None]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        sumval = jnp.sum(0.5 * (y[:-1] + y[1:]))
        return (1 + self.in_dim - sumval) ** (self.in_dim - sumval)


# LSBF
# Typo  in LSBF, no square on `y2` in the sqrt.
class Mishra4(AbstractMinimisationProblem):
    name: ClassVar[str] = "Mishra 2 function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(-0.19664, (-9.94112, -9.99957))
    in_dim: ClassVar[int] = 2

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: PRNGKeyArray = None,
    ) -> PyTree[Array]:
        # arbitrary, may need to adjust
        return [array_tuple([-5.0, -5.0])]

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [jnp.array(0.01)]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        c = args
        f1 = jnp.sqrt(jnp.abs(jnp.sin(jnp.sqrt(jnp.abs(y1**2 + y2)))))
        f2 = c * (y1 + y2)
        return f1 + f2


# LSBF
# I (packquickly) don't get the same minimum as specified in LSBF.
# I am not sure if this is a mistake in LSBF, but I do not think it
# is a mistake in my implementation
class PenHolder(AbstractMinimisationProblem):
    name: ClassVar[str] = "Pen Holder function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(None, None)
    in_dim: ClassVar[int] = 2

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: PRNGKeyArray = None,
    ) -> PyTree[Array]:
        # arbitrary, may need to adjust
        return array_tuple([5.0, 5.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [None]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        del args
        y1, y2 = y
        f1 = jnp.abs(1 - jnp.sqrt(y1**2 + y2**2) / jnp.pi)
        f2 = -jnp.exp(1 / jnp.abs(jnp.cos(y1) * jnp.cos(y2) * jnp.exp(f1)))
        return f2


# LSBF
class Pathological(AbstractMinimisationProblem):
    name: ClassVar[str] = "Pathological function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: Minimum
    in_dim: int

    def __init__(self, in_dim: int = 99):
        self.in_dim = 99
        self.minimum = Minimum(0.0, jnp.zeros(self.in_dim))

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: PRNGKeyArray = None,
    ) -> PyTree[Array]:
        # arbitrary, may need to adjust
        return jnp.ones(self.in_dim)

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([0.5, 100, 1e-3])]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        c1, c2, c3 = args
        f1 = jnp.sin(jnp.sqrt(c2 * y[:-1] ** 2 + y[1:] ** 2)) ** 2 - c1
        f2 = 1 + c3 * (y[:-1] ** 2 - 2 * y[:-1] * y[1:] + y[1:] ** 2) ** 2
        return jnp.sum(0.5 + f1 / f2)


# LSBF
# Typo in LSBF, they divide `(y1/2*y2)`, which returns a NaN at their
# specified minimum.
class Rump(AbstractMinimisationProblem):
    name: ClassVar[str] = "Rump function"
    difficulty: ClassVar[Optional[Difficulty]] = None
    minimum: ClassVar[Minimum] = Minimum(0.0, (0.0, 0.0))
    in_dim: ClassVar[int] = 2

    @additive_perturbation
    def init(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: PRNGKeyArray = None,
    ) -> PyTree[Array]:
        # arbitrary, may need to adjust
        return array_tuple([-3.0, 3.0])

    @additive_perturbation
    def args(
        self,
        random_generator: Optional[RandomGenerator] = None,
        options: Optional[dict] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> list[Optional[PyTree[Array]]]:
        return [array_tuple([333.75, 11.0, 121.0, 5.5, 0.5, 2.0])]

    def fn(self, y: PyTree[Array], args: PyTree[Array]) -> PyTree[Array]:
        y1, y2 = y
        c1, c2, c3, c4, c5, c6 = args
        f1 = (c1 - y1**2) * y2**6 + c4 * y2**8 + c5 * y1 / (c6 + y2)
        f2 = (y1**2) * (c2 * (y1**2) * (y2**2) - c3 * y2**4 - c6)
        return f1 + f2
