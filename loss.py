import numpy as np
from problem import compute_activation_times, SingleSourceProblem


class ResidualInterface:
    def __call__(self, x: np.ndarray, problem: SingleSourceProblem) -> np.ndarray:
        raise NotImplementedError()

    def jacobian(self, x: np.ndarray, problem: SingleSourceProblem) -> np.ndarray:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()


class Residual(ResidualInterface):
    def __call__(problem: SingleSourceProblem, x: np.ndarray) -> np.ndarray:
        return problem.activation_times - compute_activation_times(problem.trigger_time,
                                                                   problem.velocity,
                                                                   x,
                                                                   problem.sensor_locations)

    def jacobian(problem: SingleSourceProblem, x: np.ndarray) -> np.ndarray:
        # we need to protect ourselves from NaNs in this calculation
        diffs = (x - problem.sensor_locations) / problem.velocity

        result = -diffs / (np.linalg.norm(diffs, axis=-1, keepdims=True) * problem.velocity)

        nan_mask = np.isnan(result)
        inf_mask = np.isinf(result)

        reject_mask = nan_mask | inf_mask
        result[reject_mask] = 0

        return result

    def __str__(self) -> str:
        return 'OriginalResidual'


class SmoothResidual(ResidualInterface):
    def __call__(problem: SingleSourceProblem, x: np.ndarray) -> np.ndarray:
        time_deltas = problem.activation_times - problem.trigger_time
        displacement_to_sensors = np.expand_dims(x, axis=-2) - sensor_locations

        return time_deltas**2 - np.sum((displacement_to_sensors / problem.velocity)**2, axis=-1)

    def jacobian(problem: SingleSourceProblem, x: np.ndarray) -> np.ndarray:
        displacement_to_sensors = np.expand_dims(x, axis=-2) - sensor_locations

        return -2.0 * displacement_to_sensors / (velocity**2)


class LossInterface:
    def __call__(self, r: np.ndarray) -> np.array:
        raise NotImplementedError()

    def gradient(self, r: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_scipy_equivalent(self) -> str:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()


class SquaredLoss(LossInterface):
    def __call__(self, r: np.ndarray) -> np.array:
        return 0.5 * np.sum(r**2, axis=-1)

    def gradient(self, r: np.ndarray) -> np.ndarray:
        return r

    def get_scipy_equivalent(self) -> str:
        return 'linear'

    def __str__(self) -> str:
        return "SquaredLoss()"
