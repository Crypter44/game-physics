import numpy as np


def run_simulation(spacial_dim):
    positions = np.zeros((spacial_dim, spacial_dim, 3))
    velocities = np.zeros((spacial_dim, spacial_dim, 3))

    for x in range(spacial_dim):
        for y in range(spacial_dim):
            positions[x][y] = np.array([x, y, 0])

    print(positions)



run_simulation(3)
