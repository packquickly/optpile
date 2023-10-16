from collections.abc import Callable
from typing import cast, Optional, Union
from typing_extensions import TypeAlias

import jax.numpy as jnp
import optimistix as optx
from ax.service.ax_client import AxClient, ObjectiveProperties
from jaxtyping import Array

from .custom_types import Metric
from .opt_tester import OptTester


Solver: TypeAlias = Union[optx.AbstractMinimiser, optx.AbstractLeastSquaresSolver]

# bleh, I (packquickly) don't particularly like the method of passing parameters
# to an Ax client. I can remove that API from the end user -- and I still might --
# but it requires passing a lot of extra classes around and is sort of messy.


def optimise_parameters(
    test_name: str,
    atol: float,
    rtol: float,
    from_params_to_solver: Callable[[dict], Solver],
    params: list[dict],
    n_tuning_runs: int = 30,
    n_problem_runs: int = 5,
    metric: Metric = Metric.STEPS,
    parameter_constraints: Optional[list[str]] = None,
    verbose: bool = True,
    error_weight=1,
):
    print("[*] Beginning parameter tuning...")

    def evaluate_params(params: dict) -> dict:
        solver = from_params_to_solver(params)
        opt_tester = OptTester("pretestdefault", solver)
        results = opt_tester.run(atol, rtol, n_runs_per_problem=n_problem_runs)
        n_problems = len(results)
        sum_steps = 0
        sum_error = 0
        total_failures = 0
        n_all_runs = n_problems * n_problem_runs
        if metric == Metric.STEPS:

            if len(results) == 0:
                raise ValueError("Recieved no optimisers!")
            for problem_result in results:
                sum_steps = sum_steps + jnp.sum(problem_result.n_steps)

                if problem_result.min_possible is not None:
                    error_for_problem = jnp.abs(
                        problem_result.min_found - problem_result.min_possible
                    )
                    within_reason = error_for_problem < 1e7
                else:
                    error_for_problem = jnp.array(0.0)
                    within_reason = True

                # Having `within_reason` exists mostly for outlier problems
                # which heavily skewed results. I (packquickly) don't have a great
                # methodology for handling such problems. Catching them, and
                # and penalising the amount of problems not within reason seemed
                # to also lead to issues, especially since it introduces another,
                # relatively difficult to tune hyperparameter.
                sum_error = jnp.where(
                    within_reason, sum_error + error_for_problem, sum_error
                )
                total_failures = total_failures + jnp.where(within_reason, 1, 0)

            # make Pyright happy
            mean = cast(Array, sum_steps / n_all_runs)
            # Not sure if this should be averaged.
            # mean_error = cast(Array, sum_error / n_all_runs)
            print(f"Ave steps this run:             {mean.item()}")
            print(f"Ave total error this run:       {jnp.mean(sum_error)}\n")
            print(f"Total failures this run:        {total_failures}\n")

            output = {"loss": jnp.sum(mean + error_weight * sum_error).item()}

        elif metric == Metric.WALL_CLOCK:
            assert False
        elif metric == Metric.CPU_TIME:
            assert False
        return output

    if metric == Metric.STEPS:
        objective = {"loss": ObjectiveProperties(minimize=True)}
    elif metric == Metric.WALL_CLOCK:
        # not implemented yet!
        assert False
    elif metric == Metric.CPU_TIME:
        # not implemented yet!
        assert False
    else:
        raise ValueError(
            "`metric` must be one of `Metric.STEPS`, "
            "`Metric.WALL_CLOCK`, or `Metric.CPU_TIME. These can be called via` "
            "`optpile.Metric.STEPS`..."
        )

    ax_client = AxClient()
    ax_client.create_experiment(
        name=test_name,
        parameters=params,
        objectives=objective,
        parameter_constraints=parameter_constraints,
    )

    for _ in range(n_tuning_runs):
        parameters, trial_index = ax_client.get_next_trial()
        output = evaluate_params(parameters)
        ax_client.complete_trial(trial_index=trial_index, raw_data=output)

    return ax_client.get_best_parameters(), ax_client
