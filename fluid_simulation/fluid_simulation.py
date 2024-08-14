import numpy as np


def setup_vortex(spacial_dim, vortex_speed, vortex_center, clockwise=True):
    velocities = np.zeros((spacial_dim, spacial_dim, 2))
    for row in range(spacial_dim):
        for col in range(spacial_dim):
            # TODO: Init the staggered grid instead
            x = col
            y = row
            r = np.array([x, y]) - vortex_center
            r_norm = np.linalg.norm(r)
            velocities[row, col] = np.array([-r[1], r[0]]) * vortex_speed / r_norm ** 2
            if not clockwise:
                velocities[row, col] = -velocities[row, col]

    return velocities




