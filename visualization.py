import numpy as np
import matplotlib.pyplot as plt

from problem import L, SingleSourceProblem, compute_activation_times

RESOLUTION = 128


def plot_activation_field(problem: SingleSourceProblem,
                          resolution: int = RESOLUTION,
                          ax = None):

    if ax is None:
        _, ax = plt.subplots()


    mesh = np.linspace(-L, L, RESOLUTION)
    xx, yy = np.meshgrid(mesh, mesh)
    sample_points = np.array([xx, yy]).T
    activation_field = compute_activation_times(problem.trigger_time,
                                                problem.velocity,
                                                problem.source_location,
                                                sample_points).T

    ax.pcolormesh(mesh, mesh, activation_field)
    ax.scatter(problem.sensor_locations[:,0],
                problem.sensor_locations[:,1],
                c='w',
                label='sensors')

    ax.scatter(problem.source_location[0],
                problem.source_location[1],
                marker='x',
                c='r',
                label='source')

    ax.set_title("Activation field")
    ax.set_aspect('equal')
    ax.legend()
