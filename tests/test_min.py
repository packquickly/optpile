import jax.numpy as jnp
import optimistix as optx
import pytest

import optpile as op

from .helpers import min_problems


@pytest.mark.parametrize("problem", min_problems)
def test_problems(problem):
    if isinstance(problem, op.Meyer):
        # TODO(packquickly): find the argmin of Meyer
        pass
    elif isinstance(problem, op.Trigonometric):
        # TODO(packquickly): find the argmin of Trigonometric
        pass
    elif problem.minimum.min is not None:
        if problem.minimum.argmin is not None:
            init = problem.init()
            args = problem.args()
            found_min = problem.fn(problem.minimum.argmin, args)
            # make sure init is correct shape and not just the argmin
            problem.fn(init, args)
            assert jnp.allclose(found_min, problem.minimum.min, atol=1e-5)
        else:
            # If we don't have a known argmin, just try to solve the equation
            # and see if you can get to the minimum
            solver = optx.BFGS(rtol=1e-6, atol=1e-6)
            init = problem.init()
            args = problem.args()
            soln = optx.minimise(problem.fn, solver, init, args, max_steps=10_024)
            found_min = problem.fn(soln.value, args)
            assert jnp.allclose(found_min, problem.minimum.min, atol=1e-5)
    else:
        # If we don't have the minimum at all, just run a smoke screen test and
        # make sure the function works.
        init = problem.init()
        args = problem.args()
        found_min = problem.fn(init, args)
        assert found_min.shape == ()
