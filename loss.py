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
    return -diffs / (np.linalg.norm(diffs, axis=1, keepdims=True) * problem.velocity)


class AbstractLoss:
    def __call__(self, residuals: np.ndarray) -> float:
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()


class SquaredLoss(AbstractLoss):
    def __call__(self, residuals: np.ndarray) -> float:
        return 0.5 * np.sum(residuals**2, axis=-1)

    def __str__(self):
        return "SquaredLoss()"


class HuberLoss(AbstractLoss):
    def __init__(self, delta: float):
        self.delta = delta

    def __call__(self, residuals: np.ndarray) -> float:
        raise NotImplementedError()

    def __str__(self):
        return f"HuberLoss({self.delta})"


class CauchyLoss(AbstractLoss):
    def __init__(self, c: float = 1.0):
        self.c = c

    def __call__(self, residuals: np.ndarray) -> float:
        return np.sum(self.c**2 / 2 * np.log1p((residuals / self.c)**2), axis=-1)

    def __str__(self):
        return "CauchyLoss()"


class ArctanLoss(AbstractLoss):
    def __init__(self, c: float = 1.0):
        self.c = c

    def __call__(self, residuals: np.ndarray) -> float:
        return np.sum(self.c**2 * np.arctan((residuals / self.c)**2), axis=-1)

    def __str__(self):
        return "ArctanLoss()"
