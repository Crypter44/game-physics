import numpy as np

from staggered_grid import StaggeredGrid


def setup_vortex(spacial_dim, vortex_speed, vortex_center, clockwise=True):
    velocities = StaggeredGrid(spacial_dim)
    direction = 1 if clockwise else -1

    for row in range(spacial_dim):
        for col in range(spacial_dim):
            # calculate the velocity at the center of the left wall
            x, y = velocities.coords(row, col, velocities.LEFT)
            r = np.array([x, y]) - vortex_center
            r_norm = np.linalg.norm(r)
            u = vortex_speed / r_norm ** 2 * -r[1] * direction
            velocities.set_left(row, col, u)

            # calculate the velocity at the center of the top wall
            x, y = velocities.coords(row, col, velocities.TOP)
            r = np.array([x, y]) - vortex_center
            r_norm = np.linalg.norm(r)
            v = vortex_speed / r_norm ** 2 * r[0] * direction
            velocities.set_top(row, col, v)

            # calculate the velocity at the center of the right wall, if we are at the rightmost column
            if col == spacial_dim - 1:
                x, y = velocities.coords(row, col, velocities.RIGHT)
                r = np.array([x, y]) - vortex_center
                r_norm = np.linalg.norm(r)
                u = vortex_speed / r_norm ** 2 * -r[1] * direction
                velocities.set_right(row, col, u)

            # calculate the velocity at the center of the bottom wall, if we are at the bottom row
            if row == spacial_dim - 1:
                x, y = velocities.coords(row, col, velocities.BOTTOM)
                r = np.array([x, y]) - vortex_center
                r_norm = np.linalg.norm(r)
                v = vortex_speed / r_norm ** 2 * r[0] * direction
                velocities.set_bottom(row, col, v)

    return velocities




