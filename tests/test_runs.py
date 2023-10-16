import jax.random as jr
import optimistix as optx

import optpile as op


def test_runs():
    key = jr.PRNGKey(0)
    tester1 = op.OptTester("test runs", optx.BFGS(1e-5, 1e-5))
    tester2 = op.OptTester("test runs", optx.LevenbergMarquardt(1e-5, 1e-5))
    tester1.single_run(op.SimpleBowl(), atol=1e-2, rtol=1e-2, n_runs=2, key=key)
    tester2.single_run(op.SimpleBowl(), atol=1e-2, rtol=1e-2, n_runs=2, key=key)
    tester1.run(1e-2, 1e-2)
    tester2.run(1e-2, 1e-2)


def test_full_history_runs():
    key = jr.PRNGKey(0)
    tester1 = op.OptTester("test runs", optx.BFGS(1e-5, 1e-5))
    tester2 = op.OptTester("test runs", optx.LevenbergMarquardt(1e-5, 1e-5))
    tester1.single_full_history_run(
        op.SimpleBowl(), atol=1e-2, rtol=1e-2, n_runs=2, key=key
    )
    tester2.single_full_history_run(
        op.SimpleBowl(), atol=1e-2, rtol=1e-2, n_runs=2, key=key
    )
    tester1.full_history_run(1e-2, 1e-2, n_runs_per_problem=2)
    tester2.full_history_run(1e-2, 1e-2, n_runs_per_problem=2)
    assert True
