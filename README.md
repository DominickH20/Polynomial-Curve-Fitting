# Polynomial Curve Fitting
This project showcases various implementations of polynomial curve fitting algorithms. The two discussed in
detail below are a gradient descent approach to solving this optimization problem and a deterministic global
minimum solution to the problem.

## Gradient Descent Approach
The first approach taken here is one involving gradient descent. Because the sum of square residuals minimization
problem is an optimization problem, the gradient descent method can be used to find a solution. The gradient of the
error function is computed and the algorithm moves stepwise to approach the global minimum by intervals set by the learning rate. Below is an example of a polynomial fitted with this approach.

![Fit Curve](https://github.com/DominickH20/Polynomial-Curve-Fitting/Figures/Fitted_Curve.png)

Taking this approach also makes it convenience to incorporate regularization parameters into the optimization problem. Below you can see a curve that was fitted using gradient descent with regularization.

![Reg Curve](https://github.com/DominickH20/Polynomial-Curve-Fitting/Figures/Regularized_Fit.png)

## Deterministic Approach
