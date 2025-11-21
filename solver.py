import copy
import numpy as np
import scipy.optimize as opt

from problem import SingleSourceProblem, L, compute_arrival_time
from loss import residual, jacobian, LossInterface, SquaredLoss


class OptimizeResult(opt.OptimizeResult):
    def __init__(self, other: opt.OptimizeResult = None, **kwargs):
        super().__init__()

        if other is not None:
            for k, v in other.items():
                self[k] = copy.deepcopy(v)

        for k, v in kwargs.items():
            self[k] = v

        self['loss'] = np.mean(self['fun'])

    def __lt__(self, other: 'OptimizeResult'):
        return self['loss'] < other['loss']


# unified solver interface for both scipy and custom solvers
class SolverInterface:
    def __init__(self):
        pass

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray) -> OptimizeResult:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()


class LeastSquaresSolver(SolverInterface):
    def __init__(self, loss: LossInterface = SquaredLoss()):
        self.loss = loss

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray, **kwargs) -> OptimizeResult:
        loss_fcn = lambda x: residual(problem, x)
        jac_fcn  = lambda x: jacobian(problem, x)

        return OptimizeResult(
            opt.least_squares(
                fun=loss_fcn,
                x0=x0,
                jac=jac_fcn,
                loss=self.loss.get_scipy_equivalent(),
                **kwargs
            )
        )

    def __str__(self) -> str:
        return f'LSQ({self.loss})'


class MinimizeSolver(SolverInterface):
    def __init__(self, method: str):
        assert(method in [
            'Nelder-Mead'
            'Powell',
            'CG',
            'BFGS',
            'Newton-CG',
            'L-BFGS-B',
            'TNC',
            'COBYLA',
            'SLSQP',
            'trust-constr',
            'dogleg',
            'trust-ncg',
            'trust-exact',
            'trust-krylov',
        ])

        self.method = method
        self.loss = SquaredLoss()

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray) -> OptimizeResult:
        loss_fcn = lambda x: self.loss(problem, x)
        jac_fcn = lambda x: residual(problem, x).T @ jacobian(problem, x)

        return OptimizeResult(
            opt.minimize(
                fun=loss_fcn,
                x0=x0,
                method=self.method,
                jac=jac_fcn
            )
        )

    def __str__(self) -> str:
        return f'MNMZ({self.method}, {self.loss})'


class GridSearchSolver(SolverInterface):
    def __init__(self,
                 resolution: int,
                 lim: float = L,
                 loss: LossInterface = SquaredLoss()):

        mesh = np.linspace(-lim, lim, resolution)
        xx, yy = np.meshgrid(mesh, mesh)

        self.grid = np.array([
            xx.flatten(),
            yy.flatten()
        ]).T

        self.resolution = resolution
        self.loss = loss

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray) -> OptimizeResult:
        loss_landscape = self.loss(problem, self.grid)

        best_idx = np.argmin(loss_landscape)
        best_loss = loss_landscape[best_idx]
        best_starting_point = self.grid[best_idx]

        return OptimizeResult(
            x = best_starting_point,
            success = True,
            status = 0,
            message = f'tested all points in {self.resolution} x {self.resolution} grid',
            fun = best_loss,
            nfev = self.resolution**2,
            nit = 1
        )

    def __str__(self) -> str:
        return f"GridSearch(n={self.resolution})"


class TwoPhaseSolver(SolverInterface):
    def __init__(self, first_solver: SolverInterface, second_solver: SolverInterface):
        self.first_solver = first_solver
        self.second_solver = second_solver

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray) -> OptimizeResult:
        good_starting_point = self.first_solver.solve(problem, x0).x

        return self.second_solver.solve(problem, good_starting_point)

    def __str__(self) -> str:
        return f'TwoPhase({self.first_solver}, {self.second_solver})'


class ReflectClosestSolver(SolverInterface):
    def __init__(self, solver: SolverInterface):
        self.solver = solver

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray) -> OptimizeResult:
        first_pass = self.solver.solve(problem, x0)
        x = first_pass.x

        closest_sensor = problem.sensor_locations[
            np.argmin(
                residual(problem, x)
            )
        ]
        reflected = closest_sensor - (x - closest_sensor)

        second_pass = self.solver.solve(problem, reflected)

        return min(first_pass, second_pass)

    def __str__(self) -> str:
        return f'ReflectClosest({self.solver})'


class SensorStartSolver(SolverInterface):
    def __init__(self, solver: SolverInterface):
        self.solver = solver

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray) -> OptimizeResult:

        return min(
            self.solver.solve(problem, s)
            for s in problem.sensor_locations
        )

    def __str__(self) -> str:
        return f'SensorStart({self.solver})'


# class SymmetricRestartSolver():
#     def __init__(self,
#                  solver: SolverInterface,
#                  loss: LossInterface = SquaredLoss(),
#                  lim: float = L,
#                  resolution: int = 5):
#         super().__init__(solver, loss, lim, resolution)
# 
#     def solve(self, problem: SingleSourceProblem, x0: np.ndarray):
#         first_solve_results = super().solve(problem, x0)
#         x = first_solve_results.x
#         first_solve_loss = self.loss(problem, x)
# 
#         # this is very strange: if I use the closest sensor (with respect to the norm)
#         # I don't get the same results. Maybe I should use the metric induced by the
#         # propagation dynamic
#         #
#         # also one could simply remove the sensors with a very low arrival time as
#         # the source is too close to the sensor to give reliable results
#         #
#         # there is this phenomenon because gradients are discontinuos. Not very good
#         #
#         # we should not look at the residual by at the residual gradient. I expect
#         # that the target sensor has a gradient in the opposite direction and very strong
#         # with respect to the other sensors.
#         # This way you can't really cross the otherside without increasing the loss function
#         closest_sensor = problem.sensor_locations[
#             np.argmin(
#                 residual(problem, x)
#             )
#         ]
# 
#         reflected = closest_sensor - (x - closest_sensor)
#         second_solve_results = self.solver.solve(problem, reflected)
#         second_solve_loss = self.loss(
#             problem,
#             second_solve_results.x
#         )
# 
#         return first_solve_results if first_solve_loss < second_solve_loss else second_solve_results
# 
#     def __str__(self):
#         return f'SymmetricRestart({self.solver}, n={self.resolution})'
# 
