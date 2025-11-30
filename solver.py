import copy
import numpy as np
import scipy.optimize as opt
import itertools

from problem import SingleSourceProblem, L
from loss import ResidualInterface, LossInterface


# this is the class that gets passed to the solvers
#   * residuals and jacobians need to be together
#   * you can't compute a loss without knowing the residual
class OptimizationProblem:
    def __init__(self,
                 residual_obj: ResidualInterface,
                 loss_obj: LossInterface):

        self.residual_obj = residual_obj
        self.loss_obj = loss_obj

    def loss(x: np.ndarray, problem: SingleSourceProblem):
        return self.loss_obj(self.residual_obj(x, problem))


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
    def __init__(self, residual_obj: ResidualInterface, loss_obj: LossInterface):
        self.residual_obj = residual_obj
        self.loss_obj = loss_obj

    def overrides_initial_guess(self) -> bool:
        raise NotImplementedError()

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray) -> OptimizeResult:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()


class LeastSquaresSolver(SolverInterface):
    def __init__(self, residual_obj: ResidualInterface, loss_obj: LossInterface, **kwargs):
        super().__init__(residual_obj, loss_obj)
        self.kwargs = kwargs

    def overrides_initial_guess(self) -> bool:
        return False

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray) -> OptimizeResult:
        residuals = lambda x: self.residual_obj(x, problem)
        jacobians = lambda x: self.residual_obj.jacobian(x, problem)

        return OptimizeResult(
            opt.least_squares(
                fun=residuals,
                x0=x0,
                jac=jacobians,
                loss=self.loss_obj.get_scipy_equivalent(),
                **self.kwargs
            )
        )

    def __str__(self) -> str:
        return f'LSQ({self.loss_obj}, {self.residual_obj})'


class GridSearchSolver(SolverInterface):
    def __init__(self,
                 residual_obj: ResidualInterface,
                 loss_obj: LossInterface,
                 resolution: int,
                 lim: float = L):

        super().__init__(residual_obj, loss_obj)

        mesh = np.linspace(-lim, lim, resolution)
        xx, yy = np.meshgrid(mesh, mesh)

        self.grid = np.array([
            xx.flatten(),
            yy.flatten()
        ]).T

        self.resolution = resolution

    def overrides_initial_guess(self) -> bool:
        return True

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray) -> OptimizeResult:
        loss_landscape = self.loss_obj(self.residual_obj(self.grid, problem))

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
        return self.first_solver.overrides_initial_guess()

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray) -> OptimizeResult:
        good_starting_point = self.first_solver.solve(problem, x0).x

        return self.second_solver.solve(problem, good_starting_point)

    def __str__(self) -> str:
        return f'TwoPhase({self.first_solver}, {self.second_solver})'


class SensorStartSolver(SolverInterface):
    def __init__(self, solver: SolverInterface):
        self.solver = solver

    def overrides_initial_guess(self) -> bool:
        return True

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray) -> OptimizeResult:
        residual_obj = self.solver.residual_obj
        loss_obj = self.solver.loss_obj

        sensor_loss = loss_obj(residual_obj(problem.sensor_locations, problem))
        closest_sensor = problem.sensor_locations[
            np.argmin(sensor_loss)
        ]

        return self.solver.solve(problem, closest_sensor)

    def __str__(self) -> str:
        return f'SensorStart({self.solver})'


class CardinalStartSolver(SolverInterface):
    def __init__(self, solver: SolverInterface):
        self.solver = solver

    def overrides_initial_guess(self) -> bool:
        return True

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray) -> OptimizeResult:
        west_x0 = np.array([-L, 0.0])
        east_x0 = -west_x0
        north_x0 = np.array([0.0, L])
        south_x0 = -north_x0

        return min(
            self.solver.solve(problem, x0)
            for x0 in [west_x0, east_x0, north_x0, south_x0]
        )

    def __str__(self) -> str:
        return f'CardinalStart({self.solver})'
