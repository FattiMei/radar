import numpy as np
import matplotlib.pyplot as plt

from loss import ResidualInterface, LossInterface
from problem import L, SingleSourceProblem, compute_activation_times


RESOLUTION = 128
MESH = np.linspace(-L, L, RESOLUTION)
XX, YY = np.meshgrid(MESH, MESH)
SAMPLE_POINTS = np.array([XX, YY]).T


def plot_activation_field(problem: SingleSourceProblem,
                          ax = None):

    if ax is None:
        _, ax = plt.subplots()


    activation_field = compute_activation_times(problem.trigger_time,
                                                problem.velocity,
                                                problem.source_location,
                                                SAMPLE_POINTS).T

    ax.pcolormesh(MESH, MESH, activation_field)
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


def plot_loss_landscape(problem: SingleSourceProblem,
                        residual_obj: ResidualInterface,
                        loss_obj: LossInterface,
                        ax = None):

    if ax is None:
        _, ax = plt.subplots()

    residuals = residual_obj(SAMPLE_POINTS, problem)
    loss_landscape = loss_obj(residuals).T

    ax.pcolormesh(MESH, MESH, loss_landscape)
    ax.scatter(problem.sensor_locations[:,0],
               problem.sensor_locations[:,1],
               c='w',
               label='sensors')

    ax.scatter(problem.source_location[0],
               problem.source_location[1],
               marker='x',
               c='r',
               label='source')

    ax.set_title(f"{loss_obj}({residual_obj}) landscape")
    ax.set_aspect('equal')
