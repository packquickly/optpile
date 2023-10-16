import functools as ft

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optimistix as optx

from .custom_types import Metric
from .opt_tester import TestResults


plt.style.use("ggplot")


def performance_profile(ratios, t):
    ratios_greater_than_t = jnp.where(ratios <= t, 1, 0)
    return jnp.sum(ratios_greater_than_t) / ratios.size


def plot_performance_profile(
    results: list[TestResults],
    plot_title: str,
    performance_metric: Metric = Metric.STEPS,
    log2_scale: bool = True,
):
    if len(results) < 2:
        raise ValueError(
            "`results` needs at least two sets of test results "
            "ran on the same set of problems to generate a performance profile."
        )

    n_solvers = len(results)
    n_problems = len(results[0])

    results_same_len = [len(x) == n_problems for x in results]
    all_same_len = ft.reduce(lambda x, y: x and y, results_same_len)
    if not all_same_len:
        raise ValueError(
            "`results` must be ran on the same set of test problems! "
            "Not all the results passed had the same number of test problems. This may "
            "have happened from comparing a minimiser to a least-squares solver and "
            "forgetting to set `least_squares_only` to `True`."
        )

    perf_measures = [jnp.zeros((n_problems)) for _ in results]

    for prob_id in range(n_problems):
        # TODO(packquickly): make this more generic
        if performance_metric == Metric.STEPS:
            step_vals = jnp.full(n_solvers, jnp.inf)
            succeeded = jnp.zeros(n_solvers)

            for solver_id, solver_result in enumerate(results):
                step_vals = step_vals.at[solver_id].set(
                    jnp.mean(solver_result[prob_id].n_steps)
                )
                succeeded = succeeded.at[solver_id].set(
                    jnp.all(solver_result[prob_id].result == optx.RESULTS.successful)
                )
            min_steps = jnp.min(step_vals)
            ratios = jnp.where(succeeded, step_vals / min_steps, jnp.inf)
            for solver_id in range(n_solvers):
                perf_measures[solver_id] = (
                    perf_measures[solver_id].at[prob_id].set(ratios[solver_id])
                )
        else:
            # not implemented yet
            assert False

    perf_profiles = [
        ft.partial(performance_profile, ratios) for ratios in perf_measures
    ]
    ylabel = r"Proportion of problems solved"

    # arbitrary.
    if log2_scale:
        plot_values = jnp.linspace(0, 7, 100)
        test_values = 2**plot_values
        xlabel = r" $ \log_2 $ of num steps/min num steps"
    else:
        plot_values = jnp.linspace(0, 130, 100)
        test_values = plot_values
        xlabel = r" num_steps / min num steps"

    for test_id, profile_fn in enumerate(perf_profiles):
        profile = jax.vmap(profile_fn)(test_values)
        plt.plot(plot_values, profile, ".-", label=results[test_id].test_name)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0, 1)
    plt.title(plot_title)
    plt.legend()
    plt.show()
