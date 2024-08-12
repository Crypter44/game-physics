import numpy as np


def step(
        spacial_dim,
        positions,
        velocities,
        mass,
        spacing,
        spring_constants,
        damping_constants,
        gravity,
        dt,
        num_of_fixed_corners
):
    # inner step
    pos, vel = F(
        spacial_dim,
        positions,
        velocities,
        mass,
        spacing,
        spring_constants,
        damping_constants,
        gravity,
        num_of_fixed_corners
    )
    pos = positions + 0.5 * dt * pos
    vel = velocities + 0.5 * dt * vel

    # outer step
    pos, vel = F(
        spacial_dim,
        pos,
        vel,
        mass,
        spacing,
        spring_constants,
        damping_constants,
        gravity,
        num_of_fixed_corners
    )
    return positions + dt * pos, velocities + dt * vel


def F(
        spacial_dim,
        positions,
        velocities,
        mass,
        spacing,
        spring_constants,
        damping_constants,
        gravity,
        num_of_fixed_corners
):
    return (
        velocities,
        1 / mass * f(
            spacial_dim,
            positions,
            velocities,
            spacing,
            spring_constants,
            damping_constants,
            gravity,
            num_of_fixed_corners
        )
    )


def f(
        spacial_dim,
        positions,
        velocities,
        spacing,
        spring_constants,
        damping_constants,
        gravity,
        num_of_fixed_corners
):
    forces = np.zeros((spacial_dim * spacial_dim, 3))
    # calculate n corners
    corners = calculate_corners(spacial_dim, num_of_fixed_corners)
    for i in range(len(positions)):
        calculate_forces(forces, i, spacial_dim, positions, velocities, spacing, spring_constants, damping_constants,
                         gravity)
        if i in corners:
            forces[i] = 0

    return forces


def calculate_corners(spacial_dim, num_of_fixed_corners):
    return [
               0,
               spacial_dim - 1,
               spacial_dim * (spacial_dim - 1),
               spacial_dim ** 2 - 1
           ][0:num_of_fixed_corners]


def calculate_forces(
        forces,
        i,
        spacial_dim,
        positions,
        velocities,
        spacing,
        spring_constants,
        damping_constants,
        gravity
):
    # structural spring right
    if i % spacial_dim < spacial_dim - 1:
        force = calculate_spring(i, i + 1, positions, velocities, spacing, spring_constants[0], damping_constants[0])
        forces[i] += force
        forces[i + 1] += -force

    # structural spring down
    if i < spacial_dim ** 2 - spacial_dim:
        force = calculate_spring(i, i + spacial_dim, positions, velocities, spacing, spring_constants[0],
                                 damping_constants[0])
        forces[i] += force
        forces[i + spacial_dim] += -force

    # shear spring right down
    if i % spacial_dim < spacial_dim - 1 and i < spacial_dim ** 2 - spacial_dim:
        force = calculate_spring(i, i + spacial_dim + 1, positions, velocities, np.sqrt(2) * spacing,
                                 spring_constants[1], damping_constants[1])
        forces[i] += force
        forces[i + spacial_dim + 1] += -force

    # shear spring left down
    if i % spacial_dim > 0 and i < spacial_dim ** 2 - spacial_dim:
        force = calculate_spring(i, i + spacial_dim - 1, positions, velocities, np.sqrt(2) * spacing,
                                 spring_constants[1], damping_constants[1])
        forces[i] += force
        forces[i + spacial_dim - 1] += -force

    # flexion spring right
    if i % spacial_dim < spacial_dim - 2:
        force = calculate_spring(i, i + 2, positions, velocities, 2 * spacing, spring_constants[2],
                                 damping_constants[2])
        forces[i] += force
        forces[i + 2] += -force

    # flexion spring down
    if i < spacial_dim ** 2 - 2 * spacial_dim:
        force = calculate_spring(i, i + 2 * spacial_dim, positions, velocities, 2 * spacing, spring_constants[2],
                                 damping_constants[2])
        forces[i] += force
        forces[i + 2 * spacial_dim] += -force

    forces[i] += np.array([0, 0, -1]) * gravity


def calculate_spring(i, j, positions, velocities, l0, ks, kd):
    x12 = positions[j] - positions[i]
    x12_norm = np.linalg.norm(x12)
    x12_hat = x12 / x12_norm
    v12 = velocities[j] - velocities[i]

    spring_force = ks * (x12_norm - l0) + kd * np.dot(v12, x12_hat)
    return spring_force * x12_hat
