# Polynomial Curve Fitting
This project showcases various implementations of polynomial curve fitting algorithms. The two discussed in
detail below are a gradient descent approach to solving this optimization problem and a deterministic global
minimum solution to the problem.

## Gradient Descent Approach
The first approach taken here is one involving gradient descent. Because the sum of square residuals minimization
problem is an optimization problem, the gradient descent method can be used to find a solution. The gradient of the
error function is computed and the algorithm moves stepwise to approach the global minimum by intervals set by the learning rate. Below is an example of a polynomial fitted with this approach.

![Fit Curve](https://github.com/DominickH20/Polynomial-Curve-Fitting/blob/master/Figures/Fitted_Curve.png)

Taking this approach also makes it convenience to incorporate regularization parameters into the optimization problem. Below you can see a curve that was fitted using gradient descent with regularization.

![Reg Curve](https://github.com/DominickH20/Polynomial-Curve-Fitting/blob/master/Figures/Regularized_Fit.png)

## Deterministic Approach
The other approach taken in this project is a deterministic minimization solution. Because the sum of squared residuals minimization problem is a convex optimization problem, it has a global minimum that can be computed directly. The derivation can be found online and involves solving a system of *k* optimality conditions for
*k* polynomial coefficients. This approach allows for better computation of polynomial fits to more complex functions like *sin(x)* (seen below) since there is no need for learning rate adjustments. This implementation is also faster, but cannot be as easily regularized.

![sin Curve](https://github.com/DominickH20/Polynomial-Curve-Fitting/blob/master/Figures/sin.png)
