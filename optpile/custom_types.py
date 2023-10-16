from enum import Enum
from typing import Any, Union
from typing_extensions import TypeVar

import equinox.internal as eqxi
from jaxtyping import Array, PyTree


sentinel: Any = eqxi.doc_repr(object(), "sentinel")

Out = TypeVar("Out", bound=Union[PyTree[Array], Array])
Y = TypeVar("Y", bound=Union[PyTree[Array], Array])
Args = TypeVar("Args", bound=PyTree[Array])


class Metric(Enum):
    STEPS = 1
    WALL_CLOCK = 2
    CPU_TIME = 3
