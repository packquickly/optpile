import functools as ft
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

from .problems import AbstractTestProblem


plt.style.use("ggplot")


def performance_profile(ratios, t):
    ratios_greater_than_t = jnp.where(ratios <= t, 1, 0)
    return jnp.sum(ratios_greater_than_t) / ratios.size


# TODO(packquickly): Handle the typing of the solvers better
def plot_performance_profile(
    solvers_names: tuple[list[Any], list[str]],
    problems: list[AbstractTestProblem],
    plot_title: str,
    evaluate_problem: Callable,
    xlabel: str = r"runtime/min runtime",
    # performance_metric: Metric = Metric.STEPS,
    log2_scale: bool = True,
):
    solvers, names = solvers_names
    if len(solvers) < 2:
        raise ValueError(
            "`results` needs at least two sets of test results "
            "ran on the same set of problems to generate a performance profile."
        )

    n_solvers = len(solvers)
    n_problems = len(problems)

    perf_measures = [jnp.zeros((n_problems)) for _ in range(n_solvers)]
    losses = jnp.full(n_solvers, jnp.inf)

    for prob_id, problem in tqdm(enumerate(problems)):
        for solver_id, solver_result in enumerate(solvers):
            losses = losses.at[solver_id].set(
                evaluate_problem(solvers[solver_id], problem)
            )

        min_loss = jnp.min(losses)
        ratios = jnp.where(jnp.invert(jnp.isinf(losses)), losses / min_loss, jnp.inf)

        for solver_id in range(n_solvers):
            perf_measures[solver_id] = (
                perf_measures[solver_id].at[prob_id].set(ratios[solver_id])
            )

    perf_profiles = [
        ft.partial(performance_profile, ratios) for ratios in perf_measures
    ]
    ylabel = r"Proportion of problems solved"

    # arbitrary.
    if log2_scale:
        plot_values = jnp.linspace(0, 7, 100)
        test_values = 2**plot_values
        xlabel = r"$ \log_2 $ " + xlabel
    else:
        plot_values = jnp.linspace(0, 130, 100)
        test_values = plot_values
        xlabel = xlabel

    for test_id, profile_fn in enumerate(perf_profiles):
        profile = jax.vmap(profile_fn)(test_values)
        plt.plot(plot_values, profile, ".-", label=names[test_id])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0, 1)
    plt.title(plot_title)
    plt.legend()
    plt.show()
