import numpy as np


# box dimension
L = 2.0


# positions of the sensors. Obtained from legacy problem specification
SENSOR_LOCATIONS = np.array([
    [3.42658477, 2.46352549],
    [3.23637347, 2.40172209],
    [2.66573956, 2.2163119 ],
    [2.47552826, 2.1545085 ],
    [2.        , 3.5       ],
    [2.        , 3.3       ],
    [2.        , 2.7       ],
    [2.        , 2.5       ],
    [0.57341523, 2.46352549],
    [0.76362653, 2.40172209],
    [1.33426044, 2.2163119 ],
    [1.52447174, 2.1545085 ],
    [1.11832212, 0.78647451],
    [1.23587917, 0.94827791],
    [1.58855032, 1.4336881 ],
    [1.70610737, 1.5954915 ],
    [2.88167788, 0.78647451],
    [2.76412083, 0.94827791],
    [2.41144968, 1.4336881 ],
    [2.29389263, 1.595491  ],
]) - L


# velocity of the signal in the cardiac tissue
CONDUCTION_VELOCITY = 80.0
A_RATIO = 5.0
VELOCITY = CONDUCTION_VELOCITY / np.array([1, A_RATIO])


# time at which the source emits the signal.
# for simplicity is null, otherwise a substitution in the sensor equations must be made
T_TRIGGER = 0.0


# propagation dynamic
def compute_arrival_time(sensor_locations,
                         velocity,
                         trigger_time,
                         source_location):
    return trigger_time + np.linalg.norm(
        (np.expand_dims(source_location, axis=-2) - sensor_locations) / velocity,
        axis=-1
    )


class SingleSourceProblem:
    def __init__(self,
                 trigger_time:     float,
                 velocity:         np.ndarray,
                 source_location:  np.ndarray,
                 sensor_locations: np.ndarray,
                 arrival_times:    np.ndarray,
                 perturbation:     np.ndarray):

        self.trigger_time     = trigger_time
        self.velocity         = velocity
        self.source_location  = source_location
        self.sensor_locations = sensor_locations
        self.sensor_count     = sensor_locations.shape[0]
        self.arrival_times    = arrival_times
        self.perturbation     = perturbation


    def generate_random_instance(outliers: int = 0,
                                 sigma: float = 0.0,
                                 velocity: np.ndarray = VELOCITY) -> 'SingleSourceProblem':
        n = SENSOR_LOCATIONS.shape[0]
        assert(0 <= outliers <= n)
        assert(0.0 <= sigma)

        trigger_time = T_TRIGGER
        source_location = np.array(np.random.uniform(-L, L, size=2))

        arrival_times = compute_arrival_time(
            SENSOR_LOCATIONS,
            VELOCITY,
            trigger_time,
            source_location
        )

        perturbation = np.zeros(n)
        perturbation[
            np.random.choice(n, size=outliers, replace=False)
        ] = np.random.normal(loc=0.0, scale=sigma, size=outliers)

        return SingleSourceProblem(
            trigger_time,
            VELOCITY,
            source_location,
            SENSOR_LOCATIONS,
            arrival_times + perturbation,
            perturbation
        )


    def get_random_subproblem(self, n: int) -> 'SingleSourceProblem':
        assert(2 < n <= self.sensor_count)

        idx = np.random.choice(
            self.sensor_count,
            size=n,
            replace=False,
        )

        return SingleSourceProblem(
            self.trigger_time,
            self.velocity,
            self.source_location,
            self.sensor_locations[idx],
            self.arrival_times[idx],
            self.perturbation[idx],
        )
