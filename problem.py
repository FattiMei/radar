import numpy as np


# all the sensors and random sources are placed in [-L,L] x [-L,L]
L: float = 2.0


# positions of the sensors, obtained from legacy problem specification
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
assert(np.all(np.abs(SENSOR_LOCATIONS <= L)))


# velocity of the signal in the cardiac tissue
# the signal travels faster in the x direction
CONDUCTION_VELOCITY: float = 80.0
A_RATIO: float = 5.0
VELOCITY = CONDUCTION_VELOCITY / np.array([1, A_RATIO])


# time at which the source emits the signal.
# for simplicity is null, otherwise a substitution in the sensor equations must be made
TRIGGER_TIME: float = 0.0
assert(TRIGGER_TIME == 0.0)


# this function computes the time registered at each sensor location
# if one or multiple sources were to fire at trigger_time
def compute_activation_times(trigger_time:     float,
                             velocity:         np.array,
                             source_location:  np.ndarray,
                             sensor_locations: np.ndarray):

    return trigger_time + np.linalg.norm(
        (np.expand_dims(source_location, axis=-2) - sensor_locations) / velocity,
        axis=-1
    )


class SingleSourceProblem:
    def __init__(self,
                 trigger_time:     float,
                 velocity:         np.array,
                 sensor_locations: np.ndarray,
                 source_location:  np.array):

        self.trigger_time     = trigger_time
        self.velocity         = velocity
        self.sensor_locations = sensor_locations
        self.source_location  = source_location
        self.activation_times = compute_activation_times(trigger_time,
                                                         velocity,
                                                         source_location,
                                                         sensor_locations)


    def generate_random_instance(size:         int = SENSOR_LOCATIONS.shape[0],
                                 trigger_time: float = TRIGGER_TIME,
                                 velocity:     np.array = VELOCITY,
                                 rng = np.random.default_rng()):

        source_location = rng.uniform(-L, L, size=2)
        sensor_locations = SENSOR_LOCATIONS[rng.choice(SENSOR_LOCATIONS.shape[0],
                                                       size=size,
                                                       replace=False)]

        return SingleSourceProblem(trigger_time,
                                   velocity,
                                   sensor_locations,
                                   source_location)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from visualization import plot_activation_field

    problem = SingleSourceProblem.generate_random_instance()
    plot_activation_field(problem)
    plt.show()
