# optpile
A collection of nonlinear programming problems for benchmarking optimisation software in JAX.

The API is a work in progress, and subject to change.

Optpile pulls from a number of optimisation test problem collections, principally "Testing Unconstrained Optimization Software" by Moré, Garbow, and Hillstrom, "An Unconstrained Optimization Test Functions Collection" by Andrei, "A Literature Survey of Benchmark Functions for Global Optimization Problems" by Jamil and Yan, A Numerical Evaluation of Several Stochastic Algorithms on Selected Continuous Global Optimization Test Problems." by Ali, et al., and "Constrained and Unconstrained Testing Environment" by Gould, Orban, and Toint. A funny side-effect of this is the repetition of many loss functions resembling the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function); it seems optimisation researchers are creative in creating small permutations of this problem. The downside is a lack of truly challenging problems: you will find many ill-behaved toy functions but few problems that replicate the high-dimensional challenges of deep learning or complex scientific model calibration.

...
