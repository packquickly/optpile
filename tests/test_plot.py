import optimistix as optx

import optpile as op


def test_plot():
    solver1 = optx.BFGS(rtol=1e-6, atol=1e-8)
    solver2 = optx.NonlinearCG(rtol=1e-6, atol=1e-8)
    solver1_tester = op.OptTester("BFGS", solver1)
    solver2_tester = op.OptTester("NonlinearCG", solver2)
    op.plot_convergence_profile(
        [solver1_tester, solver2_tester], 1e-2, 1e-2, "foo", n_runs=2
    )
    solver1_results = solver1_tester.run(1e-2, 1e-2)
    solver2_results = solver2_tester.run(1e-2, 1e-2)
    op.plot_performance_profile(
        [solver1_results, solver2_results], "BFGS vs nonlinear CG test profile"
    )
    assert True
