import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import rk2_simulation as rk2


def step(
        spacial_dim,
        positions,
        velocities,
        M,
        spacing,
        spring_constants,
        damping_constants,
        Ds,
        gravity,
        dt,
        num_of_fixed_corners
):
    Ks = calc_Ks(positions, spacing, spring_constants, spacial_dim)
    fs = calc_fs(spacial_dim, num_of_fixed_corners, spring_constants, damping_constants, spacing, positions, velocities)

    delta_v = np.zeros((3 * spacial_dim ** 2))

    # do step for structural springs
    delta_v += solve_step(M, Ds[0], Ks[0], fs[0], dt, velocities, spacial_dim)

    # do step for shear springs
    delta_v += solve_step(M, Ds[1], Ks[1], fs[1], dt, velocities, spacial_dim)

    # do step for flexion springs
    delta_v += solve_step(M, Ds[2], Ks[2], fs[2], dt, velocities, spacial_dim)

    # do step for gravity
    delta_v -= dt * np.tile(gravity, spacial_dim ** 2)

    # fix corners
    corners = rk2.calculate_corners(spacial_dim, num_of_fixed_corners)
    for i in corners:
        delta_v[i * 3:(i + 1) * 3] = [0, 0, 0]

    next_vel = velocities + delta_v.reshape((spacial_dim ** 2, 3))
    next_pos = positions + dt * next_vel

    return next_pos, next_vel


def setup_M(mass, spacial_dim):
    return sp.diags([mass] * 3 * spacial_dim * spacial_dim).tocsc()


def setup_Ds(damping_constants, spacial_dim):
    Ds = []
    for i in range(3):
        Ds.append(sp.diags([damping_constants[i]] * 3 * spacial_dim * spacial_dim).tocsc())
    return Ds


def calc_Ks(positions, spacing, spring_constants, spacial_dim):
    return [
        setup_K_structural(positions, spacing, spring_constants[0], spacial_dim).tocsc(),
        setup_K_shear(positions, spacing * np.sqrt(2), spring_constants[1], spacial_dim).tocsc(),
        setup_K_flexion(positions, spacing * 2, spring_constants[2], spacial_dim).tocsc()
    ]


def setup_K_structural(positions, l0, k, spacial_dim):
    data = []
    indices = []
    indptr = [0]

    for i in range(len(positions)):
        for j in range(len(positions)):
            if i == j:
                continue
            else:
                # check if the two positions are adjacent
                if (
                        # right
                        (i + 1 == j and i % spacial_dim < spacial_dim - 1)
                        # left
                        or (i - 1 == j and i % spacial_dim > 0)
                        # down
                        or (i + spacial_dim == j and i < spacial_dim ** 2 - spacial_dim)
                        # up
                        or (i - spacial_dim == j and i >= spacial_dim)
                ):
                    block = calc_entry(positions, i, j, k, l0)
                    data.append(block)
                    indices.append(j)
                # else: block is 0
        indptr.append(len(indices))

    calc_diagonal_entries(data, indices, indptr, spacial_dim)

    return sp.bsr_matrix((np.array(data), indices, indptr), shape=(3 * spacial_dim ** 2, 3 * spacial_dim ** 2))


def setup_K_shear(positions, l0, k, spacial_dim):
    data = []
    indices = []
    indptr = [0]

    for i in range(len(positions)):
        for j in range(len(positions)):
            if i == j:
                continue
            else:
                # check if the two positions are diagonally adjacent
                if (
                        # right down
                        (i + spacial_dim + 1 == j
                         and i % spacial_dim < spacial_dim - 1
                         and i < spacial_dim ** 2 - spacial_dim)
                        # left down
                        or (i + spacial_dim - 1 == j
                            and i % spacial_dim > 0
                            and i < spacial_dim ** 2 - spacial_dim)
                        # right up
                        or (i - spacial_dim + 1 == j
                            and i % spacial_dim < spacial_dim - 1
                            and i >= spacial_dim)
                        # left up
                        or (i - spacial_dim - 1 == j
                            and i % spacial_dim > 0
                            and i >= spacial_dim)
                ):
                    block = calc_entry(positions, i, j, k, l0)
                    data.append(block)
                    indices.append(j)
                # else: block is 0
        indptr.append(len(indices))

    calc_diagonal_entries(data, indices, indptr, spacial_dim)

    return sp.bsr_matrix((np.array(data), indices, indptr), shape=(3 * spacial_dim ** 2, 3 * spacial_dim ** 2))


def setup_K_flexion(positions, l0, k, spacial_dim):
    data = []
    indices = []
    indptr = [0]

    for i in range(len(positions)):
        for j in range(len(positions)):
            if i == j:
                continue
            else:
                # check if the two positions are two away
                if (
                        (i + 2 == j and i % spacial_dim < spacial_dim - 2)
                        or (i - 2 == j and i % spacial_dim > 1)
                        or (i + 2 * spacial_dim == j and i < spacial_dim ** 2 - 2 * spacial_dim)
                        or (i - 2 * spacial_dim == j and i >= 2 * spacial_dim)
                ):
                    block = calc_entry(positions, i, j, k, l0)
                    data.append(block)
                    indices.append(j)
                # else: block is 0
        indptr.append(len(indices))

    calc_diagonal_entries(data, indices, indptr, spacial_dim)

    return sp.bsr_matrix((np.array(data), indices, indptr), shape=(3 * spacial_dim ** 2, 3 * spacial_dim ** 2))


def calc_entry(positions, i, j, k, l0):
    xij = positions[j] - positions[i]
    norm_xij = np.linalg.norm(xij)
    return k * ((norm_xij - l0) / norm_xij * np.eye(3) + l0 * (np.outer(xij, xij)) / norm_xij ** 3)


def calc_diagonal_entries(data, indices, indptr, spacial_dim):
    for i in range(spacial_dim ** 2):
        row = data[indptr[i]:indptr[i + 1]]
        if len(row) == 0:
            block = np.zeros((3, 3))
        else:
            block = sum(row)
        data.insert(i * (spacial_dim + 1), -block)
        indices.insert(i * (spacial_dim + 1), i)
        for j in range(i + 1, spacial_dim ** 2 + 1):
            indptr[j] += 1


def solve_step(M, D, K, f, delta_t, velocities, spacial_dim):
    return spl.spsolve(
        (M + delta_t * D + delta_t ** 2 * K),
        delta_t * (f + delta_t * K @ velocities.reshape((3 * spacial_dim ** 2)))
    )


def calc_fs(spacial_dim, num_of_fixed_corners, spring_constants, damping_constants, spacing, positions, velocities):
    return [
        f_structural(
            spacial_dim,
            num_of_fixed_corners,
            spacing,
            spring_constants[0],
            damping_constants[0],
            positions,
            velocities
        ),
        f_shear(
            spacial_dim,
            num_of_fixed_corners,
            spacing * np.sqrt(2),
            spring_constants[1],
            damping_constants[1],
            positions,
            velocities
        ),
        f_flexion(
            spacial_dim,
            num_of_fixed_corners,
            spacing * 2,
            spring_constants[2],
            damping_constants[2],
            positions,
            velocities
        )
    ]


def f_structural(spacial_dim, num_of_fixed_corners, l0, ks, kd, positions, velocities):
    forces = np.zeros((spacial_dim * spacial_dim, 3))
    # calculate n corners
    corners = rk2.calculate_corners(spacial_dim, num_of_fixed_corners)
    for i in range(len(positions)):
        # structural spring right
        if i % spacial_dim < spacial_dim - 1:
            force = rk2.calculate_spring(i, i + 1, positions, velocities, l0, ks, kd)
            forces[i] += force
            forces[i + 1] += -force

        # structural spring down
        if i < spacial_dim ** 2 - spacial_dim:
            force = rk2.calculate_spring(i, i + spacial_dim, positions, velocities, l0, ks, kd)
            forces[i] += force
            forces[i + spacial_dim] += -force

        if i in corners:
            forces[i] = [0, 0, 0]

    return forces.reshape((spacial_dim * spacial_dim * 3))


def f_shear(spacial_dim, num_of_fixed_corners, l0, ks, kd, positions, velocities):
    forces = np.zeros((spacial_dim * spacial_dim, 3))
    # calculate n corners
    corners = rk2.calculate_corners(spacial_dim, num_of_fixed_corners)
    for i in range(len(positions)):
        # shear spring right down
        if i % spacial_dim < spacial_dim - 1 and i < spacial_dim ** 2 - spacial_dim:
            force = rk2.calculate_spring(i, i + spacial_dim + 1, positions, velocities, l0, ks, kd)
            forces[i] += force
            forces[i + spacial_dim + 1] += -force

        # shear spring left down
        if i % spacial_dim > 0 and i < spacial_dim ** 2 - spacial_dim:
            force = rk2.calculate_spring(i, i + spacial_dim - 1, positions, velocities, l0, ks, kd)
            forces[i] += force
            forces[i + spacial_dim - 1] += -force

        if i in corners:
            forces[i] = [0, 0, 0]

    return forces.reshape((spacial_dim * spacial_dim * 3))


def f_flexion(spacial_dim, num_of_fixed_corners, l0, ks, kd, positions, velocities):
    forces = np.zeros((spacial_dim * spacial_dim, 3))
    # calculate n corners
    c = 0
    corners = rk2.calculate_corners(spacial_dim, num_of_fixed_corners)
    for i in range(len(positions)):
        # flexion spring right
        if i % spacial_dim < spacial_dim - 2:
            force = rk2.calculate_spring(i, i + 2, positions, velocities, l0, ks, kd)
            forces[i] += force
            forces[i + 2] += -force

        # flexion spring down
        if i < spacial_dim ** 2 - 2 * spacial_dim:
            force = rk2.calculate_spring(i, i + 2 * spacial_dim, positions, velocities, l0, ks, kd)
            forces[i] += force
            forces[i + 2 * spacial_dim] += -force

        if i in corners:
            forces[i] = [0, 0, 0]

    return forces.reshape((spacial_dim * spacial_dim * 3))
