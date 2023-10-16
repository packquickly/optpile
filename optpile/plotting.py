import functools as ft
from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .custom_types import Metric
from .opt_tester import FullHistoryResult, OptTester, Termination, TestResults
from .random_generators import RandomGenerator


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

    perf_measures = [jnp.zeros(n_problems) for _ in results]

    for prob_id in range(n_problems):
        if performance_metric == Metric.STEPS:
            step_vals = jnp.full(n_solvers, jnp.inf)
            succeeded = jnp.zeros(n_solvers)

            for (solver_id, solver_result) in enumerate(results):
                step_vals = step_vals.at[solver_id].set(
                    jnp.mean(solver_result[prob_id].n_steps)
                )
                succeeded = succeeded.at[solver_id].set(
                    jnp.all(solver_result[prob_id].succeeded)
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

    for (test_id, profile_fn) in enumerate(perf_profiles):
        profile = jax.vmap(profile_fn)(test_values)
        plt.plot(plot_values, profile, ".-", label=results[test_id].test_name)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0, 1)
    plt.title(plot_title)
    plt.legend()
    plt.show()


# @eqx.filter_jit
def plot_convergence_profile(
    opt_testers: list[OptTester],
    atol: float,
    rtol: float,
    plot_title: str,
    n_runs=1,
    performance_metric: Metric = Metric.STEPS,
    max_steps=10_024,
    accuracy=1e-5,
    multiplicative_noise: bool = False,
    additive_noise: bool = False,
    noise_random_generator: Optional[RandomGenerator] = None,
    log2_scale: bool = True,
):
    if len(opt_testers) < 2:
        raise ValueError(
            "`results` needs at least two sets of test results "
            "ran on the same set of problems to generate a performance profile."
        )

    results = [
        tester.full_history_run(
            atol,
            rtol,
            Termination.MAX_STEPS,
            max_steps=max_steps,
            n_runs_per_problem=n_runs,
            multiplicative_output_noise=multiplicative_noise,
            additive_output_noise=additive_noise,
            noise_random_generator=noise_random_generator,
        )
        for tester in opt_testers
    ]
    n_solvers = len(opt_testers)
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

    perf_measures = [jnp.zeros((n_runs, n_problems)) for _ in results]

    for prob_id in range(n_problems):
        if performance_metric == Metric.STEPS:
            step_vals = jnp.full((n_runs, n_solvers), jnp.inf)

            mins = []
            for solver_results in results:
                mins.append(
                    jnp.min(solver_results.problem_results[prob_id].values, axis=1)
                )
            if results[0].problem_results[prob_id].min_possible is None:
                minimum = jnp.min(ft.reduce(lambda x, y: jnp.minimum(x, y), mins))
            else:
                minimum = results[0].problem_results[prob_id].min_possible

            for (solver_id, solver_result) in enumerate(results):
                vals = solver_result.problem_results[prob_id].values
                f0 = vals[:, 0]
                get_first_in_tol = lambda x, y: jnp.argmax(
                    x < minimum + accuracy * (y - minimum)
                )

                first_step_in_tol = jnp.mean(jax.vmap(get_first_in_tol)(vals, f0))
                final_val = vals[:, -1]
                final_val_above_tol = final_val > minimum + accuracy * (f0 - minimum)
                hit_max = solver_result.problem_results[prob_id].steps == max_steps

                first_step_in_tol = jnp.where(
                    final_val_above_tol & hit_max, jnp.inf, first_step_in_tol
                )

                # if solver_result.problem_results[prob_id].min_possible is not None:
                #     true_min = solver_result.problem_results[prob_id].min_possible
                #     found_mins = jnp.min(
                #         solver_result.problem_results[prob_id].values, axis=1
                #     )
                #     scale = atol + jnp.asarray(rtol) * true_min
                #     within_tol = jnp.abs(found_mins - true_min)/scale
                #     first_step_in_tol = jnp.where(
                #         within_tol, first_step_in_tol, jnp.inf
                #     )

                step_vals = step_vals.at[:, solver_id].set(first_step_in_tol)

            min_steps = jnp.min(step_vals, axis=1)
            safe_min_steps = jnp.where(
                (min_steps != 0) | (min_steps == jnp.inf), min_steps, 1
            )
            ratios = jnp.where(min_steps != 0, step_vals / safe_min_steps, 1.0)
            ratios = jnp.where(min_steps == jnp.inf, jnp.inf, ratios)
            # failed = jnp.where(succeeded, step_vals / min_steps, jnp.inf)
            for solver_id in range(n_solvers):
                perf_measures[solver_id] = (
                    perf_measures[solver_id].at[:, prob_id].set(ratios[:, solver_id])
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

    for (test_id, profile_fn) in enumerate(perf_profiles):
        profile = jax.vmap(profile_fn)(test_values)
        plt.plot(plot_values, profile, ".-", label=results[test_id].test_name)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0, 1)
    plt.title(plot_title)
    plt.legend()
    plt.show()


def plot_convergence(
    plot_title,
    full_history_runs: list[FullHistoryResult],
    n_steps_to_plot: Optional[int] = None,
):
    max_size = max([result.steps for result in full_history_runs])  # pyright: ignore
    if n_steps_to_plot is not None:
        size = jnp.minimum(max_size.item(), n_steps_to_plot)
    else:
        size = max_size
    padded_arrays = [
        result.values.flatten()[result.steps] * jnp.ones(size)
        for result in full_history_runs
    ]
    for i, result in enumerate(full_history_runs):
        if n_steps_to_plot is not None:
            vals = result.values.flatten()[:size]
        else:
            vals = result.values.flatten()

        idx = jnp.minimum(size, vals.size)
        padded_arrays[i] = padded_arrays[i].at[:idx].set(vals)
        plt.plot(padded_arrays[i], label=result.test_name)
    plt.xlabel(r"Step")
    plt.yscale("log")
    plt.ylabel(r"Function value")
    plt.title(plot_title)
    plt.legend()
    plt.show()


def plot_runtimes(*multiple_runs_of_same_problem_with_different_sizes):
    ...
