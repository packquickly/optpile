from .opt_tester import OptTester


def evaluate_params(
    problems,
    from_params_to_solver,
    evaluate_problem,
    max_steps,
    n_problem_runs,
    problem_options,
    solver_options,
    params: dict,
) -> dict:
    solver = from_params_to_solver(params)
    opt_tester = OptTester("pretestdefault", solver)
    results = opt_tester.run(
        problems,
        max_steps=max_steps,
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
