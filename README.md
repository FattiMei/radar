# Source localization problem
A source point is placed in $[-L,L] \times [-L,L]$. At time $T_{trigger}$ emits a signal that propagates in space by the law:
$$ T(x,y) = T_{trigger} + \sqrt{(\frac{x - x_s}{v_x})^2 + (\frac{y - y_s}{v_y})^2} $$

where:
  * $T(x,y)$ is the time the point $(x,y)$ receives the signal
  * $(x_s, y_s)$ is the position of the source

there are 20 sensor placed in the domain at fixed position. Find the position of the signal source given the arrival time measurements at sensor locations. What happens when some sensors are faulty and give outlier data?


## As an optimization problem
We are tasked to simultaneously satisfy the sensor equations
$$ T_i = T_{trigger} + \sqrt{(\frac{x - x_i}{v_x})^2 + (\frac{y - y_i}{v_y})^2} $$

since there are more equations than unknowns and there may be errors, we solve the non linear system in a **least squares sense**. To simplify the calculations we assume $T_{trigger} = 0$.
$$ min_{(x,y)} \sum_{i} \mathcal{L}(T_i - \sqrt{(\frac{x - x_i}{v_x})^2 + (\frac{y - y_i}{v_y})^2} $$

for an appropriate loss function. Note that some loss functions can attenuate the contributions of outliers.


### Regularity (or lack of)
The least squares problem is a very elegant solution, but the loss function is not $C^1$. In fact, the gradients are discontinuous on the sensor locations. This means that a line search optimizer will struggle when approaching sensor locations.

We have experimental evidence that at least 5% of the optimizations fail. Further analysis shows that the problem is ill-conditioned i.e. the termination condition on the gradient is not a good estimator of the error. The plot below shows exponential energy levels, see how squeezed they are

![plot](./resources/ill_condition.png)


### Patches
Some techniques for lowering the error rate of the optimization procedure are proposed. For instance, improving the choice of the starting point before optimization has a positive effect. A (coarse) *grid search* is used to get close to the optimum, reducing the contribution of non linearities. However, grid search is an *antipattern* and is not in the ethos of numerical optimization.

Another technique is to eliminate the sensors that produce the discontinuity in the gradient. This is very successful and fails only when the source is placed very close to the sensors.


## As an higher dimensional optimization problem
The essence of the least squares description is to satisfy simultaneously the constraint. For our purposes such constraints are level curves of functions with discontinuous gradients.


We could enforce to satisfy the sensor constraints by parametrizing them. Since they are ellipses there is a $C^{\infty}$ parametrization $\phi_i{\theta_i}$ for each sensor. As we asked a single source to satisfy all constraints, not we ask **all parametrizations to be a single source**
$$ min_{\mathbf{\theta}} Var(\phi_i{\theta_i}) $$

where the variance is a measure of how those parametrizations agree with each other. Ideally the variance is zero and all the parametrizations are a single point, otherwise the problem solution is the mean of the parametrizations.


### Regularity
The resulting problem has the dimensions of the sensors, but we expect a positive result as:
  * it has infinite regularity
  * it is bounded (the $\phi{i} are periodic)

the problems could be:
  * there may be multiple stationary points in the domain
  * the gradients and hessians are hard to compute
  * if zero variance is not reached, the resulting source location could be nonsense
