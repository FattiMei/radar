import numpy as np
import scipy.optimize as opt

from problem import SingleSourceProblem, L
from loss import residual, jacobian, AbstractLoss, SquaredLoss


# unified solver interface for both scipy and custom solvers
# should I call it SolverInterface?
class AbstractSolver:
    def __init__(self):
        pass

    # TODO: add return type
    def solve(self, problem: SingleSourceProblem, x0: np.ndarray):
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()


class LeastSquaresSolver(AbstractSolver):
    # TODO: we need also the keyword arguments to supply to the scipy solver
    def __init__(self, loss: AbstractLoss = SquaredLoss()):
        self.loss = loss

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray):
        loss_fcn = lambda x: residual(problem, x)
        jac_fcn = lambda x: jacobian(problem, x)

        return opt.least_squares(
            fun=loss_fcn,
            x0=x0,
            jac=jac_fcn,
            loss=self.loss.get_scipy_equivalent(),
        )

    def __str__(self) -> str:
        return f'LSQ({self.loss})'


class MinimizeSolver(AbstractSolver):
    def __init__(self, method: str):
        assert(method in [
              'Nelder-Mead'
            , 'Powell'
            , 'CG'
            , 'BFGS'
            , 'Newton-CG'
            , 'L-BFGS-B'
            , 'TNC'
            , 'COBYLA'
            , 'SLSQP'
            , 'trust-constr'
            , 'dogleg'
            , 'trust-ncg'
            , 'trust-exact'
            , 'trust-krylov'
        ])

        self.method = method
        self.loss = SquaredLoss()

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray):
        loss_fcn = lambda x: self.loss(problem, x)

        # TODO: for now we support only the SquaredLoss,
        # if we wanted to add new implementations we would have to add
        # a gradient method to the loss interface
        jac_fcn = lambda x: residual(problem, x).T @ jacobian(problem, x)

        return opt.minimize(
            fun=loss_fcn,
            x0=x0,
            method=self.method,
            jac=jac_fcn
        )

    def __str__(self) -> str:
        return f'MNMZ({self.method}, {self.loss})'


class GridSearchSolver(AbstractSolver):
    def __init__(self,
                 resolution: int,
                 lim: float = L,
                 loss: AbstractLoss = SquaredLoss(),
                 solver: AbstractSolver = None):

        mesh = np.linspace(-lim, lim, resolution)
        xx, yy = np.meshgrid(mesh, mesh)

        self.grid = np.array([
            xx.flatten(),
            yy.flatten()
        ]).T

        self.resolution = resolution
        self.loss = loss
        self.solver = solver

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray):
        loss_landscape = self.loss(problem, self.grid)

        best_starting_point = self.grid[
            np.argmin(loss_landscape)
        ]

        # TODO: I wanted to make this solver fully compatible with
        # the scipy ones. Can I call a phony minimize with zero nfev?
        # that way I'll get the proper residuals and proper formatting of results

        # Also I'm thinking that this solver can
        # incorporate GoodStartSolver also.
        #
        # Nevertheless it is only an easy software problem, not math
        class SolverResult:
            def __init__(self, x):
                self.x = x

        return SolverResult(x=best_starting_point)

    def __str__(self) -> str:
        return f"GridSearch(n={self.resolution})"


class GoodStartSolver(AbstractSolver):
    def __init__(self,
                 solver: AbstractSolver,
                 loss: AbstractLoss = SquaredLoss(),
                 lim: float = L,
                 resolution: int = 5):


        self.grid_search_solver = GridSearchSolver(
            resolution,
            lim,
            loss
        )
        self.loss = loss
        self.resolution = resolution
        self.solver = solver

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray):
        best_starting_point = self.grid_search_solver.solve(problem, x0).x

        return self.solver.solve(problem, best_starting_point)

    def __str__(self):
        return f'GoodStart({self.solver}, n={self.resolution})'


class SymmetricRestartSolver(GoodStartSolver):
    def __init__(self,
                 solver: AbstractSolver,
                 loss: AbstractLoss = SquaredLoss(),
                 lim: float = L,
                 resolution: int = 5):
        super().__init__(solver, loss, lim, resolution)

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray):
        first_solve_results = super().solve(problem, x0)
        x = first_solve_results.x
        first_solve_loss = self.loss(problem, x)

        # this is very strange: if I use the closest sensor (with respect to the norm)
        # I don't get the same results. Maybe I should use the metric induced by the
        # propagation dynamic
        #
        # also one could simply remove the sensors with a very low arrival time as
        # the source is too close to the sensor to give reliable results
        #
        # there is this phenomenon because gradients are discontinuos. Not very good
        closest_sensor = problem.sensor_locations[
            np.argmin(
                residual(problem, x)
            )
        ]

        reflected = closest_sensor - (x - closest_sensor)
        second_solve_results = self.solver.solve(problem, reflected)
        second_solve_loss = self.loss(
            problem,
            second_solve_results.x
        )

        return first_solve_results if first_solve_loss < second_solve_loss else second_solve_results

    def __str__(self):
        return f'SymmetricRestart({self.solver}, n={self.resolution})'
