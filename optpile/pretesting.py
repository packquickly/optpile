from collections.abc import Callable
from typing import Optional, Union
from typing_extensions import TypeAlias

import optimistix as optx
from ax.service.ax_client import AxClient, ObjectiveProperties

from .opt_tester import OptTester
from .problems import AbstractTestProblem


Solver: TypeAlias = Union[optx.AbstractMinimiser, optx.AbstractLeastSquaresSolver]
Options: TypeAlias = Optional[Union[list[dict], dict]]

# bleh, I (packquickly) don't particularly like the method of passing parameters
# to an Ax client. I can remove that API from the end user -- and I still might --
# but it requires passing a lot of extra classes around and is sort of messy.


def optimise_parameters(
    test_name: str,
    problems: list[AbstractTestProblem],
    evaluate_problem: Callable,
    from_params_to_solver: Callable[[dict], Solver],
    params: list[dict],
    n_tuning_runs: int = 30,
    n_problem_runs: int = 5,
    parameter_constraints: Optional[list[str]] = None,
    problem_options: Options = None,
    solver_options: Options = None,
    verbose: bool = True,
):
    print("[*] Beginning parameter tuning...")

    def evaluate_params(params: dict) -> dict:
        solver = from_params_to_solver(params)
        opt_tester = OptTester("pretestdefault", solver)
        results = opt_tester.run(
            problems,
            n_runs_per_problem=n_problem_runs,
            problem_options=problem_options,
            solver_options=solver_options,
        )
        len(results)
        if len(results) == 0:
            raise ValueError("Recieved no optimisers!")
        loss = 0.0
        for problem_result in results:
            loss = loss + evaluate_problem(problem_result).item()
        return {"loss": loss}

    objective = {"loss": ObjectiveProperties(minimize=True)}
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
