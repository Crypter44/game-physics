import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from datastructures.staggered_grid import StaggeredGrid


def project(solve, velocities, dt, rho=1):
    """
    Project the velocities to be mass-conserving.
    :param velocities: the grid containing the velocities
    :param dt: the time step
    :return: the new grid containing the velocities
    """
    divergence = calculate_divergence(velocities)
    pressure = solve_poisson_equation(solve, divergence, dt, rho)
    pressure = pressure.reshape(velocities.grid_dim, velocities.grid_dim) # reshape the pressure back to grid form
    velocities = correct_velocities(velocities, pressure, dt)
    new_divergence = calculate_divergence(velocities)
    return velocities, pressure


def calculate_divergence(velocities):
    """
    Calculate the divergence of the velocities.
    :param velocities: the grid containing the velocities
    :return: the divergence of the velocities, reshaped into a 1D array for compatibility with the next steps
    """
    divergence = np.zeros((velocities.grid_dim, velocities.grid_dim))
    for row in range(velocities.grid_dim):
        for col in range(velocities.grid_dim):
            up, right, down, left = velocities[row, col]
            divergence[row, col] = right - left + down - up  # dx = 1, so we don't need to divide by dx
    return divergence.reshape(-1)


def solve_poisson_equation(solve, divergence, dt, rho=1):
    """
    Solve the Poisson equation for the pressure.
    :param solve: the function to solve the linear system of equations
    :param divergence: the divergence of the velocities
    :param dt: the time step
    :param rho: the density of the fluid
    :return: the pressure field
    """
    return solve(-rho / dt * divergence)


def setup_solver(grid_dim):
    """
    Set up the solver for the Poisson equation.
    :param grid_dim: the dimension of the grid
    :return: a function that solves the linear system of equations for the Poisson equation
    """
    main_diagonal = np.ones(grid_dim ** 2) * 4
    for i in range(grid_dim ** 2):
        if i < grid_dim:
            main_diagonal[i] -= 1
        if i >= grid_dim ** 2 - grid_dim:
            main_diagonal[i] -= 1
        if (i + 1) % grid_dim == 0:
            main_diagonal[i] -= 1
        if i % grid_dim == 0:
            main_diagonal[i] -= 1

    right_neighbour_diagonal = np.ones(grid_dim ** 2) * -1
    for i in range(grid_dim ** 2):
        if i % grid_dim == 0:
            right_neighbour_diagonal[i] = 0

    left_neighbour_diagonal = np.ones(grid_dim ** 2) * -1
    for i in range(grid_dim ** 2):
        if i % grid_dim == grid_dim - 1:
            left_neighbour_diagonal[i] = 0

    vertical_neighbour_diagonal = np.ones(grid_dim ** 2) * -1

    data = np.array([
        main_diagonal,
        right_neighbour_diagonal,
        left_neighbour_diagonal,
        vertical_neighbour_diagonal,
        vertical_neighbour_diagonal
    ])
    offsets = np.array([0, 1, -1, grid_dim, -grid_dim])

    A = sp.dia_matrix(
        (data, offsets),
        shape=(grid_dim ** 2, grid_dim ** 2)
    )  # dx = 1, so we don't need to divide by dx**2

    for i in range(grid_dim ** 2):
        assert np.sum(np.array(A.toarray())[i]) == 0, \
            f"Row {i} is not zero. Check the boundary conditions of the poisson matrix."

    return spl.factorized(A.tocsc())


def correct_velocities(velocities, pressure, dt, rho=1):
    """
    Correct the velocities to be mass-conserving.
    :param velocities: the grid containing the velocities
    :param pressure: the pressure field
    :param dt: the time step
    :param rho: the density of the fluid
    :return: the new grid containing the velocities
    """
    new_velocities = StaggeredGrid(velocities.grid_dim)
    for row in range(velocities.grid_dim):
        for col in range(velocities.grid_dim):
            # correct the right velocity, if we are not at the right boundary
            if col < velocities.grid_dim - 1:
                new_velocities[row, col, new_velocities.RIGHT] = (
                        velocities[row, col, velocities.RIGHT]
                        - dt / rho * (
                                pressure[row, col + 1] - pressure[row, col]  # TODO: check if this needs to be swapped
                        )
                )
            # correct the bottom velocity, if we are not at the bottom boundary
            if row < velocities.grid_dim - 1:
                new_velocities[row, col, new_velocities.BOTTOM] = (
                        velocities[row, col, velocities.BOTTOM]
                        - dt / rho * (
                                pressure[row + 1, col] - pressure[row, col]  # TODO: check if this needs to be swapped
                        )
                )

    return new_velocities
