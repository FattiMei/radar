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
The least squares problem is a very elegant solution, but the loss function is not $C^1$. In fact, the gradients are discontinuous at the sensor locations. This means that a line search optimizer will struggle when approaching sensor locations.

We have experimental evidence that at least 5% of the optimizations fail. Further analysis shows that the problem is ill-conditioned i.e. the termination condition on the gradient is not a good estimator of the error. The plot below shows exponential energy levels, see how squeezed they are

![plot](./resources/ill_condition.png)


### Patches
Some techniques for lowering the error rate of the optimization procedure are proposed:
  * improve the starting point by a (coarse) grid search
  * start the optimization from the earliest sensor

![plot](./resources/fail_frequency.png)


while some unmentioned techniques were successful, I couldn't understand why. The whole process is babysitting an optimizer on a bad problem, coming up with techniques valid only for this particular scenario. I find this somewhat against the spirit of numerical optimization which to me sounds like:
> manipulating the problem as to require minimal tuning intervention to the solver/optimizer


## Failed experiment: an higher dimensional optimization problem
The essence of the least squares description is to satisfy simultaneously the constraints. For our purposes such constraints are level curves of functions with discontinuous gradients.

We could enforce to satisfy the sensor constraints by parametrizing them. Since they are ellipses there is a $C^{\infty}$ parametrization $\phi_i({\theta_i})$ for each sensor. As we asked a single source to satisfy all constraints, now we ask **all parametrizations to be a single source**

$$ min_{\mathbf{\theta}} \quad Var(\phi_i({\theta_i})) $$

where the variance is a measure of how those parametrizations agree with eachother. Ideally the variance is zero and all the parametrizations are a single point, otherwise the problem solution is the mean of the parametrizations.


### Comments
This new problem:
  * has infinite regularity
  * the loss function is bounded
  * has the dimension of the sensors

this new formalization is very elegant, but does not have the expected convergence. A last attempt is made by providing automatic gradients to the loss function.
