import rk2_simulation as rk2
import implicit_euler as ie
import numpy as np
from tqdm import tqdm


def setup_positions(spacial_dim, spacing):
    positions = np.zeros((spacial_dim, spacial_dim, 3))
    for i in range(spacial_dim):
        for j in range(spacial_dim):
            positions[i][j] = np.array([j * spacing, i * spacing * 1, spacial_dim // 2])

    # The resulting positions can be accessed as position[y][x] where y is the row and x is the column
    return positions


def run_simulation(
        spacial_dim,
        mass,
        spacing,
        spring_constants,
        damping_constants,
        gravity,
        dt,
        num_steps,
        simulation_type='rk2',
        num_of_fixed_corners=2
):
    """
    Run a simulation of a cloth using the given parameters
    :param spacial_dim: The number of vertices along each axis of the cloth
    :param mass: the mass of each vertex in kg
    :param spacing: the distance between each vertex in meters
    :param spring_constants: a Vector of spring constants for each type of spring (structural, shear, flexion)
    :param damping_constants: a Vector of damping constants for each type of spring (structural, shear, flexion)
    :param gravity: the acceleration due to gravity in m/s^2
    :param dt: the step size
    :param num_steps: the number of steps to simulate
    :param simulation_type: the type of simulation to run (rk2, implicit_euler)
    :param num_of_fixed_corners: the number of corners to fix in place
    :return: a list of the positions of the vertices at each time step in the format [(X1, Y1, Z1), (X2, Y2, Z2), ...]
    """
    positions = [setup_positions(spacial_dim, spacing).reshape((spacial_dim * spacial_dim, 3))]
    velocities = [np.zeros((spacial_dim, spacial_dim, 3)).reshape((spacial_dim * spacial_dim), 3)]

    if simulation_type == 'rk2':
        for i in tqdm(range(num_steps), desc="Running simulation", unit="steps"):
            pos, vel = rk2.step(
                spacial_dim,
                positions[i],
                velocities[i],
                mass,
                spacing,
                spring_constants,
                damping_constants,
                gravity,
                dt,
                num_of_fixed_corners
            )
            velocities.append(vel)
            positions.append(pos)
    elif simulation_type == 'implicit_euler':
        M = ie.setup_M(mass, spacial_dim)
        Ds = ie.setup_Ds(damping_constants, spacial_dim)
        for i in tqdm(range(num_steps), desc="Running simulation", unit="steps"):
            pos, vel = ie.step(
                spacial_dim,
                positions[i],
                velocities[i],
                M,
                spacing,
                spring_constants,
                damping_constants,
                Ds,
                gravity,
                dt,
                num_of_fixed_corners
            )
            velocities.append(vel)
            positions.append(pos)
    else:
        raise ValueError("Invalid simulation type. Please choose 'rk2' or 'implicit_euler'")

    results = []
    for i in tqdm(range(len(positions)), desc="Reshaping results", unit="steps"):
        positions[i] = positions[i].reshape(spacial_dim, spacial_dim, 3)
        results.append((positions[i][:, :, 0], positions[i][:, :, 1], positions[i][:, :, 2]))

    return results
