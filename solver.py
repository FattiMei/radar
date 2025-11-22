import copy
import numpy as np
import scipy.optimize as opt
import itertools

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

    def overrides_initial_guess(self) -> bool:
        raise NotImplementedError()

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray) -> OptimizeResult:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()


class LeastSquaresSolver(SolverInterface):
    def __init__(self, loss: LossInterface = SquaredLoss(), **kwargs):
        self.loss = loss
        self.kwargs = kwargs

    def overrides_initial_guess(self) -> bool:
        return False

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray) -> OptimizeResult:
        loss_fcn = lambda x: residual(problem, x)
        jac_fcn  = lambda x: jacobian(problem, x)

        return OptimizeResult(
            opt.least_squares(
                fun=loss_fcn,
                x0=x0,
                jac=jac_fcn,
                loss=self.loss.get_scipy_equivalent(),
                **self.kwargs
            )
        )

    def __str__(self) -> str:
        return f'LSQ({self.loss})'


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

    def overrides_initial_guess(self) -> bool:
        return True

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

        assert(not self.second_solver.overrides_initial_guess())

    def overrides_initial_guess(self) -> bool:
        return True

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray) -> OptimizeResult:
        good_starting_point = self.first_solver.solve(problem, x0).x

        return self.second_solver.solve(problem, good_starting_point)

    def __str__(self) -> str:
        return f'TwoPhase({self.first_solver}, {self.second_solver})'


class SensorStartSolver(SolverInterface):
    def __init__(self, solver: SolverInterface, loss: LossInterface = SquaredLoss()):
        self.solver = solver
        self.loss = loss

    def overrides_initial_guess(self) -> bool:
        return True

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray) -> OptimizeResult:
        sensor_loss = self.loss(problem, problem.sensor_locations)
        closest_sensor = problem.sensor_locations[
            np.argmin(sensor_loss)
        ]

        return self.solver.solve(problem, closest_sensor)

    def __str__(self) -> str:
        return f'SensorStart({self.solver})'
