import numpy as np
from tqdm import tqdm

from advection import advect
from datastructures.staggered_grid import StaggeredGrid
from projection import project, setup_solver


def setup_vortex(spacial_dim, vortex_speed, vortex_center, clockwise=True):
    velocities = StaggeredGrid(spacial_dim)
    direction = 1 if clockwise else -1

    for row in range(spacial_dim):
        for col in range(spacial_dim):
            # calculate the velocity at the center of the right wall
            x, y = velocities.coords(row, col, velocities.RIGHT)
            r = np.array([x, y]) - vortex_center
            r_norm = np.linalg.norm(r)
            u = vortex_speed / r_norm ** 2 * -r[1] * direction
            velocities.set_right(row, col, u)

            # calculate the velocity at the center of the bottom wall
            x, y = velocities.coords(row, col, velocities.BOTTOM)
            r = np.array([x, y]) - vortex_center
            r_norm = np.linalg.norm(r)
            v = vortex_speed / r_norm ** 2 * r[0] * direction
            velocities.set_bottom(row, col, v)

    return velocities


def step(velocities, solve, dt, rho=1):
    """
    Perform one step of the fluid simulation.
    :param velocities: the grid containing the velocities
    :param dt: the time step
    :param rho: the density of the fluid
    :return: the new grid containing the velocities
    """
    velocities = advect(velocities, dt)
    velocities, pressure = project(solve, velocities, dt, rho)
    return velocities, pressure


def run_simulation(spacial_dim, vortex_speeds, vortex_centers, clockwise, steps, dt, rho=1):
    print("Setting up vortexes...")
    velocities = StaggeredGrid(spacial_dim)
    for vortex_speed, vortex_center, is_clockwise in zip(vortex_speeds, vortex_centers, clockwise):
        vortex = setup_vortex(spacial_dim, vortex_speed, vortex_center, is_clockwise)
        velocities.u += vortex.u
        velocities.v += vortex.v

    print("Preparing solver...")
    solve = setup_solver(spacial_dim)

    print("Performing first pressure projection...")
    velocities, pressures = project(solve, velocities, dt, rho)

    resulting_velocities = [velocities]
    resulting_pressures = [pressures]

    for _ in tqdm(range(steps), desc="Running simulation", unit="steps"):
        velocities, pressures = step(velocities, solve, dt, rho)
        resulting_velocities.append(velocities)
        resulting_pressures.append(pressures)

    return list(map(lambda x: x.to_regular_grid(), resulting_velocities)), resulting_pressures



