import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from tqdm import tqdm


def generate_sparse_fdm_matrix(spacial_dim, boundary_conditions) -> sp.lil_matrix:
    # matrix has spacial_dim**2 rows and columns since we have spacial_dim**2 grid points which we need update
    # the resulting matrix has the coefficients for the finite difference method formula for each grid point in one row,
    # treating the row as the 2D grid flattened to a 1D array in row-major order
    sparse_matrix = sp.lil_matrix((spacial_dim**2, spacial_dim**2))

    # set each row of the matrix, so that the fdm formula uses the correct neighbors:
    # u_dot = a / h**2 * (u[i, j-1] + u[i, j+1] + u[i-1, j] + u[i+1, j] - 4 * u[i, j])
    if boundary_conditions == 'wrap_around':
        for i in range(spacial_dim**2):
            sparse_matrix[i, i] = -4

            # check if we have a neighbor to the left
            if i % spacial_dim > 0:
                sparse_matrix[i, i - 1] = 1
            else:
                sparse_matrix[i, i + spacial_dim - 1] = 1

            # check if we have a neighbor to the right
            if i % spacial_dim < spacial_dim - 1:
                sparse_matrix[i, i + 1] = 1
            else:
                sparse_matrix[i, i - spacial_dim + 1] = 1

            # check if we have a neighbor above
            if i >= spacial_dim:
                sparse_matrix[i, i - spacial_dim] = 1
            else:
                sparse_matrix[i, (spacial_dim * (spacial_dim - 1)) + i % spacial_dim] = 1

            # check if we have a neighbor below
            if i < spacial_dim ** 2 - spacial_dim:
                sparse_matrix[i, i + spacial_dim] = 1
            else:
                sparse_matrix[i, i % spacial_dim] = 1

    elif boundary_conditions == 'isolated':
        for i in range(spacial_dim ** 2):
            sparse_matrix[i, i] = -4

            # check if we have a neighbor to the left
            if i % spacial_dim > 0:
                sparse_matrix[i, i - 1] = 1
            else :
                sparse_matrix[i, i] += 1

            # check if we have a neighbor to the right
            if i % spacial_dim < spacial_dim - 1:
                sparse_matrix[i, i + 1] = 1
            else:
                sparse_matrix[i, i] += 1

            # check if we have a neighbor above
            if i >= spacial_dim:
                sparse_matrix[i, i - spacial_dim] = 1
            else:
                sparse_matrix[i, i] += 1

            # check if we have a neighbor below
            if i < spacial_dim**2 - spacial_dim:
                sparse_matrix[i, i + spacial_dim] = 1
            else:
                sparse_matrix[i, i] += 1

    elif boundary_conditions == 'unisolated':
        for i in range(spacial_dim ** 2):
            sparse_matrix[i, i] = -4

            # check if we have a neighbor to the left
            if i % spacial_dim > 0:
                sparse_matrix[i, i - 1] = 1

            # check if we have a neighbor to the right
            if i % spacial_dim < spacial_dim - 1:
                sparse_matrix[i, i + 1] = 1

            # check if we have a neighbor above
            if i >= spacial_dim:
                sparse_matrix[i, i - spacial_dim] = 1

            # check if we have a neighbor below
            if i < spacial_dim**2 - spacial_dim:
                sparse_matrix[i, i + spacial_dim] = 1

    else:
        raise ValueError("Invalid boundary condition. Please choose 'wrap_around', 'isolated', or 'unisolated'")

    return sparse_matrix


def calculate_forward_euler_step(grid_vector, dx, dt, a, fdm_matrix) -> np.ndarray:
    # calculate the next step of the simulation using the forward euler method
    return grid_vector + a * dt / dx**2 * fdm_matrix.dot(grid_vector)


def calculate_implicit_euler_step(grid_vector, solve) -> np.ndarray:
    return solve(grid_vector)


def run_simulation(
        spacial_dim,
        time_steps,
        a,
        dx,
        dt,
        heat_positions,
        boundary_conditions='unisolated',
        simulator='forward'
) -> list[np.ndarray]:
    results = []

    # generate the sparse matrix for the finite difference method
    fdm_matrix = generate_sparse_fdm_matrix(spacial_dim, boundary_conditions)

    if simulator == 'implicit':
        solve = spl.factorized(sp.eye(spacial_dim**2).tocsc() - a * dt / dx**2 * fdm_matrix.tocsc())

    # initialize the grid with the initial conditions
    results.append(np.zeros(spacial_dim**2))
    for position in heat_positions:
        results[0][position[0] * spacial_dim + position[1]] = position[2]

    # run the simulation for the specified number of time steps
    for i in tqdm(range(time_steps), desc="Running simulation", unit="steps"):
        if simulator == 'forward':
            results.append(calculate_forward_euler_step(results[i], dx, dt, a, fdm_matrix))
        elif simulator == 'implicit':
            results.append(calculate_implicit_euler_step(results[i], solve))
        else:
            raise ValueError("Invalid simulator. Please choose 'forward' or 'implicit'")

    return list(map(lambda x: x.reshape(spacial_dim, spacial_dim), results))
