from enum import Enum
from typing import cast, Optional, Union
from typing_extensions import TypeAlias

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jr
import optimistix as optx
from equinox.internal import ω
from jaxtyping import Array, Float, Int, PRNGKeyArray, Scalar, ScalarLike
from tqdm import tqdm

from .custom_types import sentinel
from .misc import sum_squares
from .problems import (
    AbstractLeastSquaresProblem,
    AbstractMinimisationProblem,
    AbstractTestProblem,
    Difficulty,
)
from .random_generators import NormalRandomGenerator, RandomGenerator


# TODO(packquickly): allow the `OptTester` API to work with non-Optimistix
# solvers.
# TODO(packquickly): avoid using enums, they're just a bit more overhead
# for the end user.
# TODO(packquickly): support multiple termination criteria.


Solver: TypeAlias = Union[optx.AbstractMinimiser, optx.AbstractLeastSquaresSolver]
Options: TypeAlias = Optional[Union[list[dict], dict]]


def _process_options_list(
    possibly_options_list: Optional[Union[list[dict], dict]], index: int
):
    if possibly_options_list is not None:
        if isinstance(possibly_options_list, list):
            return possibly_options_list[index]
        else:
            return possibly_options_list
    else:
        return {}


class Termination(Enum):
    CAUCHY = 1
    GRADIENT = 2
    MAX_STEPS = 3


class SingleProblemResult(eqx.Module):
    problem: AbstractTestProblem
    problem_in_dim: int
    min_possible: Optional[ScalarLike]
    n_steps: Int[Array, " n_inits"]
    f0: Float[Array, " n_inits"]
    min_found: Float[Array, " n_inits"]
    result: optx.RESULTS
    wall_clock_time: Optional[Float[Array, " n_inits"]] = None
    cpu_time: Optional[Float[Array, " n_inits"]] = None
    compile_time: Optional[Float[Array, " n_inits"]] = None


class FullHistoryResult(eqx.Module):
    test_name: str
    problem: AbstractTestProblem
    problem_in_dim: int
    min_possible: Optional[ScalarLike]
    values: Float[Array, "max_steps n_runs"]
    steps: Scalar
    wall_clock_time: Optional[Float[Array, " ..."]] = None
    cpu_time: Optional[Float[Array, " ..."]] = None
    compile_time: Optional[Float[Array, " ..."]] = None


# `wall_clock_time`, `cpu_time`, and `compile_time` are
# `None` for now because I'm not aware of an efficient
# method for computing them.


class TestResults(eqx.Module):
    test_name: str
    problem_results: list[SingleProblemResult]

    def __len__(self):
        return len(self.problem_results)

    def __iter__(self):
        yield from self.problem_results

    def __getitem__(self, i):
        return self.problem_results[i]


class FullTestResults(eqx.Module):
    test_name: str
    problem_results: list[FullHistoryResult]

    def __len__(self):
        return len(self.problem_results)

    def __iter__(self):
        yield from self.problem_results

    def __getitem__(self, i):
        return self.problem_results[i]


class OptTester(eqx.Module):
    test_name: str
    solver: Solver

    # @eqx.filter_jit
    def run(
        self,
        problems: list[AbstractTestProblem],
        max_steps: Optional[int] = 10_024,
        n_runs_per_problem: int = 1,
        multiplicative_output_noise: bool = False,
        additive_output_noise: bool = False,
        noise_random_generator: Optional[RandomGenerator] = None,
        init_random_generator: Optional[RandomGenerator] = None,
        args_random_generator: Optional[RandomGenerator] = None,
        problem_options: Options = None,
        solver_options: Options = None,
        problem_difficulty: Optional[Difficulty] = sentinel,
        least_squares_only: bool = False,
        *,
        key: PRNGKeyArray = jr.PRNGKey(0),
    ):
        """Solve every problem in the Optpile using `self.solver`.
        Return's a `TestResults` object whicho holds the result of
        each problem.

        **Arguments:**
        `Termination` enum with values `CAUCHY` and `GRADIENT`.
        - `max_steps`: The maximum number of steps `self.solver` can take
        to solve each problem.
        - `n_runs_per_problem`: The number of different random inits to use
        for each problem. After the first solve of the problem, the
        problem is solved again `n_runs_per_problem - 1` times with random
        perturbations of the init to hedge against lucky initialisations.
        - `problem_difficulty`: selects problems only of a specific difficulty,
        either `Difficulty.EASY` or `Difficulty.HARD`. By default, uses all problems.
        - `least_squares_only`: whether to run only on least-squares problems. This
        is useful for comparing minimisation algorithms to nonlinear least-squares
        algorithms.
        """
        # TODO(packquickly): implement problem difficulty
        del problem_difficulty
        output_results = []

        is_lstsq_solver = isinstance(self.solver, optx.AbstractLeastSquaresSolver)
        len(problems)
        for i, problem in tqdm(enumerate(problems)):
            problem_options_i = _process_options_list(problem_options, i)
            solver_options_i = _process_options_list(solver_options, i)
            is_min_problem = isinstance(problem, AbstractMinimisationProblem)
            if is_min_problem and is_lstsq_solver:
                continue
            elif is_min_problem and least_squares_only:
                continue
            else:
                # The same key is intentionally used for each run, it makes
                # debugging easier.
                output_results.append(
                    self.single_run(
                        problem,
                        max_steps,
                        n_runs_per_problem,
                        multiplicative_output_noise,
                        additive_output_noise,
                        noise_random_generator,
                        init_random_generator,
                        args_random_generator,
                        problem_options_i,
                        solver_options_i,
                        key=key,
                    )
                )

        return TestResults(self.test_name, output_results)

    # @eqx.filter_jit
    def full_history_run(
        self,
        problems: list[AbstractTestProblem],
        max_steps: int = 10_024,
        n_runs_per_problem: int = 1,
        problem_difficulty: Optional[Difficulty] = sentinel,
        multiplicative_output_noise: bool = False,
        additive_output_noise: bool = False,
        noise_random_generator: Optional[RandomGenerator] = None,
        init_random_generator: Optional[RandomGenerator] = NormalRandomGenerator(0.1),
        args_random_generator: Optional[RandomGenerator] = None,
        problem_options: Optional[dict] = None,
        solver_options: Optional[dict] = None,
        has_aux: bool = False,
        tags: frozenset[object] = frozenset(),
        least_squares_only: bool = False,
        *,
        key: PRNGKeyArray = jr.PRNGKey(0),
    ):
        output_results = []

        is_lstsq_solver = isinstance(self.solver, optx.AbstractLeastSquaresSolver)
        len(problems)
        for i, problem in tqdm(enumerate(problems)):
            is_min_problem = isinstance(problem, AbstractMinimisationProblem)
            if is_min_problem and is_lstsq_solver:
                continue
            elif is_min_problem and least_squares_only:
                continue
            else:
                # The same key is intentionally used for each run, it makes
                # debugging easier.
                output_results.append(
                    self.single_full_history_run(
                        problem,
                        max_steps,
                        n_runs_per_problem,
                        multiplicative_output_noise,
                        additive_output_noise,
                        noise_random_generator,
                        init_random_generator,
                        args_random_generator,
                        problem_options,
                        solver_options,
                        has_aux,
                        tags,
                        key=key,
                    )
                )
        return FullTestResults(self.test_name, output_results)

    def single_run(
        self,
        problem: AbstractTestProblem,
        max_steps: Optional[int] = 10_024,
        n_runs: int = 1,
        multiplicative_output_noise: bool = False,
        additive_output_noise: bool = False,
        noise_random_generator: Optional[RandomGenerator] = None,
        init_random_generator: Optional[RandomGenerator] = NormalRandomGenerator(0.1),
        args_random_generator: Optional[RandomGenerator] = None,
        problem_options: Optional[dict] = None,
        solve_options: Optional[dict] = None,
        has_aux: bool = False,
        adjoint: optx.AbstractAdjoint = optx.ImplicitAdjoint(),
        tags: frozenset[object] = frozenset(),
        *,
        key: PRNGKeyArray,
    ):
        if isinstance(problem, AbstractLeastSquaresProblem):
            optimise = optx.least_squares
            readout = lambda x: 0.5 * sum_squares(x)
            # Why is this needed? shouldn't the conversion happen
            # automatically since optimise is specified?
            if isinstance(self.solver, optx.AbstractMinimiser):
                transform = lambda x: 0.5 * sum_squares(x)
            else:
                transform = lambda x: x
        else:
            optimise = optx.minimise
            readout = lambda x: x
            transform = lambda x: x
            if isinstance(self.solver, optx.AbstractLeastSquaresSolver):
                raise ValueError(
                    "Attempted to use a least-squares solver on a "
                    "minimisation problem. Consider using a minimiser or solve a "
                    "least-squares problem instead."
                )

        if additive_output_noise or multiplicative_output_noise:
            if noise_random_generator is None:
                raise ValueError(
                    " If using additive or multiplicative output noise "
                    "then must also pass a `noise_random_generator`. To use gaussian "
                    "noise with variance `σ` please use `op.NormalRandomGenerator(σ)`."
                )

            if additive_output_noise:

                def handle_noise(f, *, key):
                    noise = noise_random_generator(f, key=key)
                    return (f**ω + noise**ω).ω

            elif multiplicative_output_noise:

                def handle_noise(f, *, key):
                    noise = noise_random_generator(f, key=key)
                    return (f**ω * (1 + noise**ω)).ω

        else:

            def handle_noise(f, *, key):
                del key
                return f

        def auxmented(x, key_args):
            key, args = key_args
            out = problem.fn(x, args)
            if has_aux:
                f, aux = out
            else:
                f = out
                aux = None
            return transform(handle_noise(f, key=key)), (readout(f), aux)

        init_key, args_key, noise_key = jr.split(key, 3)
        init_keys = jr.split(init_key, n_runs)
        args_key = jr.split(args_key, n_runs)

        init = jax.vmap(
            lambda x: problem.init(init_random_generator, problem_options, key=x)
        )(init_keys)
        pre_args = jax.vmap(
            lambda x: problem.args(args_random_generator, problem_options, key=x)
        )(args_key)
        noise_keys = jr.split(noise_key, n_runs)
        args = (noise_keys, pre_args)

        # We pay the cost of an extra function eval and compilation to get `f0`.
        # This simplifies writing early eval parameter optimisation/profiling,
        # which is a common criteria in benchmarking optimisers.
        f0 = jax.vmap(problem.fn)(init, pre_args)

        soln = jax.vmap(
            lambda x, y: optimise(
                auxmented,
                self.solver,  # pyright: ignore
                x,
                y,
                options=solve_options,
                has_aux=True,
                max_steps=max_steps,
                adjoint=adjoint,
                throw=False,
                tags=tags,
            )
        )(init, args)

        min_found, aux = soln.aux
        steps = soln.stats["num_steps"]

        return SingleProblemResult(
            problem,
            problem.in_dim,
            problem.minimum.min,
            steps,
            f0,
            min_found,
            soln.result,
            None,
            None,
        )

    # @eqx.filter_jit
    def single_full_history_run(
        self,
        problem: AbstractTestProblem,
        max_steps: int = 10_024,
        n_runs: int = 1,
        multiplicative_output_noise: bool = False,
        additive_output_noise: bool = False,
        noise_random_generator: Optional[RandomGenerator] = None,
        init_random_generator: Optional[RandomGenerator] = NormalRandomGenerator(0.1),
        args_random_generator: Optional[RandomGenerator] = None,
        problem_options: Optional[dict] = None,
        solve_options: Optional[dict] = None,
        has_aux: bool = False,
        tags: frozenset[object] = frozenset(),
        *,
        key: PRNGKeyArray,
    ):
        """Get the losses for each step for a problem.
        Must have `max_steps` specified.
        """
        if isinstance(problem, AbstractLeastSquaresProblem):
            readout = lambda x: 0.5 * sum_squares(x)
            if isinstance(self.solver, optx.AbstractMinimiser):
                transform = lambda x: 0.5 * sum_squares(x)
            else:
                transform = lambda x: x
        else:
            readout = lambda x: x
            transform = lambda x: x
            if isinstance(self.solver, optx.AbstractLeastSquaresSolver):
                raise ValueError(
                    "Attempted to use a least-squares solver on a "
                    "minimisation problem. Consider using a minimiser or solve a "
                    "least-squares problem instead."
                )

        if additive_output_noise or multiplicative_output_noise:
            if noise_random_generator is None:
                raise ValueError(
                    " If using additive or multiplicative output noise "
                    "then must also pass a `noise_random_generator`. To use gaussian "
                    "noise with variance `σ` please use `op.NormalRandomGenerator(σ)`."
                )

            if additive_output_noise:

                def handle_noise(f, *, key):
                    noise = noise_random_generator(f, key=key)
                    return (f**ω + noise**ω).ω

            elif multiplicative_output_noise:

                def handle_noise(f, *, key):
                    noise = noise_random_generator(f, key=key)
                    return (f**ω * (1 + noise**ω)).ω

        else:

            def handle_noise(f, *, key):
                del key
                return f

        def auxmented(x, key_args):
            key, args = key_args
            out = problem.fn(x, args)
            if has_aux:
                f, aux = out
            else:
                f = out
                aux = None
            return transform(handle_noise(f, key=key)), (readout(f), aux)

        if problem_options is None:
            problem_options = {}

        init_key, args_key, noise_key = jr.split(key, 3)
        init = problem.init(init_random_generator, problem_options, key=init_key)
        pre_args = problem.args(args_random_generator, problem_options, key=args_key)
        args = (key, pre_args)
        f_struct, aux_struct = jax.eval_shape(lambda: auxmented(init, args))

        vmapped_history_solve = jax.vmap(
            lambda x, y: solve_with_history(
                auxmented,
                self.solver,
                x,
                y,
                problem.minimum.min,
                multiplicative_output_noise,
                additive_output_noise,
                max_steps,
                f_struct,
                aux_struct,
                problem_options,
                frozenset(),
            )
        )
        many_inits_key = jr.split(init_key, n_runs)
        many_args_key = jr.split(args_key, n_runs)
        many_inits = jax.vmap(
            lambda x: problem.init(init_random_generator, problem_options, key=x)
        )(many_inits_key)
        pre_args = jax.vmap(
            lambda x: problem.args(args_random_generator, problem_options, key=x)
        )(many_args_key)
        noise_keys = jr.split(noise_key, n_runs)
        many_args = (noise_keys, pre_args)
        losses, steps = vmapped_history_solve(many_inits, many_args)

        return FullHistoryResult(
            test_name=self.test_name,
            problem=problem,
            problem_in_dim=problem.in_dim,
            min_possible=problem.minimum.min,
            values=losses,
            steps=steps,
            wall_clock_time=None,
            cpu_time=None,
            compile_time=None,
        )


def solve_with_history(
    fn,
    solver,
    init,
    args,
    known_min,
    multiplicative_output_noise,
    additive_output_noise,
    max_steps,
    f_struct,
    aux_struct,
    options,
    tags,
):
    state = solver.init(fn, init, args, options, f_struct, aux_struct, tags)
    done, result = solver.terminate(fn, init, args, options, state, tags)
    losses = jnp.inf * jnp.ones(max_steps)
    init_state = ((jnp.array(0), losses, init, state, done, result),)
    (dynamic_init,), (static_state,) = eqx.partition(init_state, eqx.is_array_like)

    def cond_fn(carry):
        carry = eqx.combine(carry, static_state)
        step, losses, y, state, done, result = carry
        return jnp.invert(done)

    def body_fn(carry):
        carry = eqx.combine(carry, static_state)
        step, losses, y, state, done, result = carry
        if multiplicative_output_noise or additive_output_noise:
            key, original_args = args  # pyright: ignore
            key, _ = jr.split(key)
            new_args = (key, original_args)
        else:
            new_args = args
        y, state, (loss, aux) = solver.step(
            fn, y, new_args, cast(dict, options), state, tags
        )
        done, result = solver.terminate(
            fn, y, new_args, cast(dict, options), state, tags
        )
        losses = losses.at[step].set(loss)
        dynamic_carry = eqx.filter(
            (step + 1, losses, y, state, done, result), eqx.is_array_like
        )
        return dynamic_carry

    steps, losses, argmin, _, _, result = eqxi.while_loop(
        cond_fn, body_fn, dynamic_init, max_steps=max_steps, kind="lax"
    )
    return losses, steps
