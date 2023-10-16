from collections.abc import Callable
from typing import cast

import optimistix as optx

from optpile.pretesting import optimise_parameters


class TrustRegionBFGS(optx.AbstractBFGS):
    rtol: float
    atol: float
    norm: Callable
    use_inverse: bool
    descent: optx.AbstractDescent
    search: optx.AbstractSearch

    def __init__(
        self,
        rtol: float,
        atol: float,
        high_cutoff: float,
        low_cutoff: float,
        high_constant: float,
        low_constant: float,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = optx.max_norm
        self.use_inverse = False
        self.descent = optx.DampedNewtonDescent()
        self.search = optx.ClassicalTrustRegion(
            high_cutoff, low_cutoff, high_constant, low_constant
        )


def params_to_trust_region_bfgs(params: dict):
    high_cutoff = cast(float, params.get("high_cutoff"))
    low_cutoff = cast(float, params.get("low_cutoff"))
    high_constant = cast(float, params.get("high_constant"))
    low_constant = cast(float, params.get("low_constant"))
    return TrustRegionBFGS(
        1e-5, 1e-7, high_cutoff, low_cutoff, high_constant, low_constant
    )


tr_params = [
    {
        "name": "high_cutoff",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "float",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    },
    {
        "name": "low_cutoff",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "float",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    },
    {
        "name": "high_constant",
        "type": "range",
        "bounds": [1.0, 10.0],
        "value_type": "float",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    },
    {
        "name": "low_constant",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "float",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    },
]


def test_optimise_params():
    # This is a smoke-screen test to make sure everything runs.
    best_params = optimise_parameters(
        "test optimise params",
        params_to_trust_region_bfgs,
        params=tr_params,
        n_problem_runs=2,
        n_tuning_runs=2,
        parameter_constraints=["low_cutoff <= high_cutoff"],
    )
    print(f"Found best params at {best_params}")
