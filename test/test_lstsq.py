import jax.numpy as jnp
import optimistix as optx
import pytest

import optpile as op

from .helpers import sum_squares, tuos_problems


@pytest.mark.parametrize("problem", tuos_problems)
def test_problems(problem):
    if isinstance(problem, op.Meyer):
        # TODO(packquickly): find the argmin of Meyer
        pass
    elif isinstance(problem, op.Trigonometric):
        # TODO(packquickly): find the argmin of Trigonometric
        pass
    elif problem.minimum.min is not None:
        if problem.minimum.argmin is not None:
            args_list = problem.args()
            # TODO(packquickly): find a more clever way to do this.
            args = args_list[0]
            found_min = sum_squares(problem.fn(problem.minimum.argmin, args))
            assert jnp.allclose(found_min, problem.minimum.min, atol=1e-5)
        else:
            # If we don't have a known argmin, just try to solve the equation
            # and see if you can get to the minimum
            solver = optx.LevenbergMarquardt(rtol=1e-6, atol=1e-6)
            init = problem.init()
            args_list = problem.args()
            for args in args_list:
                soln = optx.least_squares(
                    problem.fn, solver, init, args, max_steps=10_024
                )
                found_min = sum_squares(problem.fn(soln.value, args))
                assert jnp.allclose(found_min, problem.minimum.min, atol=1e-5)
    else:
        # If we don't have the minimum at all, just run a smoke screen test and
        # make sure the function works.
        init = problem.init()
        args_list = problem.args()
        for args in args_list:
            problem.fn(init, args)
            assert True
