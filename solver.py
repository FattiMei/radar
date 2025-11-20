from problem import SingleSourceProblem, L
from loss import residual, jacobian, AbstractLoss, SquaredLoss
import numpy as np
import scipy.optimize as opt


# unified solver interface for both scipy and custom solvers
class AbstractSolver:
    def __init__(self):
        pass

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()


class LeastSquaresSolver(AbstractSolver):
    def __init__(self, loss: str = 'linear', gtol: float = 1e-8):
        assert(loss in [
            'linear',
            'soft_l1',
            'huber',
            'cauchy',
            'arctan',
        ])

        self.loss = loss
        self.gtol = gtol

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray):
        loss_fcn = lambda x: residual(problem, x)
        jac_fcn = lambda x: jacobian(problem, x)

        return opt.least_squares(
            fun=loss_fcn,
            x0=x0,
            jac=jac_fcn,
            gtol=self.gtol
        )

    def __str__(self):
        return f'LSQ({self.loss}, gtol={self.gtol})'


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

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray):
        loss_fcn = lambda x: 0.5 * np.sum(residual(problem, x)**2)
        jac_fcn = lambda x: residual(problem, x).T @ jacobian(problem, x)

        return opt.minimize(
            fun=loss_fcn,
            x0=x0,
            method=self.method,
            jac=jac_fcn
        )

    def __str__(self):
        return f'{self.method}'


class GridSearchSolver(AbstractSolver):
    def __init__(self,
                 resolution: int,
                 lim: float = L,
                 loss: AbstractLoss = SquaredLoss()):

        mesh = np.linspace(-lim, lim, resolution)
        xx, yy = np.meshgrid(mesh, mesh)

        self.grid = np.array([
            xx.flatten(),
            yy.flatten()
        ]).T

        self.resolution = resolution
        self.loss = loss

    def solve(self, problem: SingleSourceProblem, x0: np.ndarray):
        loss_landscape = self.loss(problem, self.grid)

        best_starting_point = self.grid[
            np.argmin(loss_landscape)
        ]

        class SolverResult:
            def __init__(self, x):
                self.x = x

        return SolverResult(x=best_starting_point)

    def __str__(self):
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
