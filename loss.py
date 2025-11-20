import numpy as np
from problem import compute_arrival_time, SingleSourceProblem


def residual(problem: SingleSourceProblem, x: np.ndarray) -> np.ndarray:
    return problem.arrival_times - compute_arrival_time(
        problem.sensor_locations,
        problem.velocity,
        problem.trigger_time,
        x
    )


# I was not able to use autograd libraries, so here's an hand-written implementation
# there was an error because I didn't divide by velocity
def jacobian(problem: SingleSourceProblem, x: np.ndarray) -> np.ndarray:
    diffs = (x - problem.sensor_locations) / problem.velocity
    return -diffs / (np.linalg.norm(diffs, axis=-1, keepdims=True) * problem.velocity)


# I'm thinking of calling it "LossInterface"
class AbstractLoss:
    def get_scipy_equivalent(self) -> str:
        raise NotImplementedError()

    def __call__(self, problem: SingleSourceProblem, x: np.ndarray) -> float:
        raise NotImplementedError()

    def gradient(self, problem: SingleSourceProblem, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()


class SquaredLoss(AbstractLoss):
    def get_scipy_equivalent(self) -> str:
        return 'linear'

    def __call__(self, problem: SingleSourceProblem, x: np.ndarray) -> float:
        residuals = residual(problem, x)

        return 0.5 * np.sum(residuals**2, axis=-1)

    def __str__(self) -> str:
        return "SquaredLoss()"


class HuberLoss(AbstractLoss):
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def get_scipy_equivalent(self) -> str:
        return 'huber'

    def __call__(self, problem: SingleSourceProblem, x: np.ndarray) -> float:
        raise NotImplementedError()

    def __str__(self) -> str:
        return f"HuberLoss({self.delta})"


class CauchyLoss(AbstractLoss):
    def __init__(self, c: float = 1.0):
        self.c = c

    def get_scipy_equivalent(self) -> str:
        return 'cauchy'

    def __call__(self, problem: SingleSourceProblem, x: np.ndarray) -> float:
        residuals = residual(problem, x)

        return np.sum(self.c**2 / 2 * np.log1p((residuals / self.c)**2), axis=-1)

    def __str__(self) -> str:
        return "CauchyLoss()"


class ArctanLoss(AbstractLoss):
    def __init__(self, c: float = 1.0):
        self.c = c

    def get_scipy_equivalent(self) -> str:
        return 'arctan'

    def __call__(self, problem: SingleSourceProblem, x: np.ndarray) -> float:
        residuals = residual(problem, x)

        return np.sum(self.c**2 * np.arctan((residuals / self.c)**2), axis=-1)

    def __str__(self) -> str:
        return "ArctanLoss()"
