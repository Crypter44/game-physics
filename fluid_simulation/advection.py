import numpy as np

from datastructures.errors import StaggeredGridIndexError
from datastructures.staggered_grid import StaggeredGrid


def advect(velocities, dt):
    new_grid = StaggeredGrid(velocities.grid_dim)
    for row in range(velocities.grid_dim):
        for col in range(velocities.grid_dim):
            # trace the particle located at the left wall
            start_x, start_y = trace_particle()
            u = interpolate_u()
            new_grid.set_left(row, col, u)

            # trace the particle located at the top wall
            start_x, start_y = trace_particle()
            v = interpolate_v()
            new_grid.set_top(row, col, v)

            # trace the particle located at the right wall, if we are at the rightmost column
            if col == velocities.grid_dim - 1:
                start_x, start_y = trace_particle()
                u = interpolate_u()
                new_grid.set_right(row, col, u)

            # trace the particle located at the bottom wall, if we are at the bottom row
            if row == velocities.grid_dim - 1:
                start_x, start_y = trace_particle()
                v = interpolate_v()
                new_grid.set_bottom(row, col, v)


def trace_particle(row, col, side, velocities, dt):
    """
    Traces a particle back in time by dt seconds.
    The particle is located at the position (row, col) of the grid and is located at the side of the grid specified by
    the side parameter.
    :param row: the row of the grid where the particle is located
    :param col: the column of the grid where the particle is located
    :param side: the side of the grid where the particle is located
    :param velocities: the grid containing the velocities
    :param dt: the time in seconds to trace the particle back
    :return: the origin of the particle that was traced back in time
    """
    x, y = velocities.coords(row, col, side)
    if side == velocities.LEFT or side == velocities.RIGHT:
        u = velocities[row, col, side]
        v = interpolate_v(x, y, velocities)
    elif side == velocities.TOP or side == velocities.BOTTOM:
        u = interpolate_u(x, y, velocities)
        v = velocities[row, col, side]
    else:
        raise ValueError("Invalid side")

    return x - dt * u, y - dt * v


def interpolate_u(x, y, velocities):
    row, col = int(np.round(y)), int(np.round(x))
    left1 = get_or_extrapolate(row, col, velocities.LEFT, velocities)
    pos_l1 = velocities.coords(row, col, velocities.LEFT)

    right1 = get_or_extrapolate(row, col, velocities.RIGHT, velocities)
    pos_r1 = velocities.coords(row, col, velocities.RIGHT)

    if y > row:
        left2 = get_or_extrapolate(row + 1, col, velocities.LEFT, velocities)
        pos_l2 = velocities.coords(row + 1, col, velocities.LEFT)
        right2 = get_or_extrapolate(row + 1, col, velocities.RIGHT, velocities)
    else:
        left2 = get_or_extrapolate(row - 1, col, velocities.LEFT, velocities)
        pos_l2 = velocities.coords(row - 1, col, velocities.LEFT)
        right2 = get_or_extrapolate(row - 1, col, velocities.RIGHT, velocities)

    # weights (alpha, 1-alpha) for linear interpolation along x-axis
    alpha = (x - pos_l1[0]) / (pos_r1[0] - pos_l1[0])

    # weights (beta, 1-beta) for linear interpolation along y-axis
    beta = (y - pos_l1[1]) / (pos_l2[1] - pos_l1[1])

    # linear interpolation along x-axis
    x_vel1 = left1 * (1 - alpha) + right1 * alpha
    x_vel2 = left2 * (1 - alpha) + right2 * alpha

    # linear interpolation along y-axis
    u = x_vel1 * (1 - beta) + x_vel2 * beta
    return u


def interpolate_v(x, y, velocities):
    row, col = int(np.round(y)), int(np.round(x))
    top1 = get_or_extrapolate(row, col, velocities.TOP, velocities)
    pos_t1 = velocities.coords(row, col, velocities.TOP)

    bottom1 = get_or_extrapolate(row, col, velocities.BOTTOM, velocities)
    pos_b1 = velocities.coords(row, col, velocities.BOTTOM)

    if x > col:
        top2 = get_or_extrapolate(row, col + 1, velocities.TOP, velocities)
        bottom2 = get_or_extrapolate(row, col + 1, velocities.BOTTOM, velocities)
        pos_t2 = velocities.coords(row, col + 1, velocities.TOP)
    else:
        top2 = get_or_extrapolate(row, col - 1, velocities.TOP, velocities)
        bottom2 = get_or_extrapolate(row, col - 1, velocities.BOTTOM, velocities)
        pos_t2 = velocities.coords(row, col - 1, velocities.TOP)

    # weights (alpha, 1-alpha) for linear interpolation along y-axis
    alpha = (y - pos_t1[1]) / (pos_b1[1] - pos_t1[1])

    # weights (beta, 1-beta) for linear interpolation along x-axis
    beta = (x - pos_t1[0]) / (pos_t2[0] - pos_t1[0])

    # linear interpolation along y-axis
    y_vel1 = top1 * (1 - alpha) + bottom1 * alpha
    y_vel2 = top2 * (1 - alpha) + bottom2 * alpha

    # linear interpolation along x-axis
    v = y_vel1 * (1 - beta) + y_vel2 * beta
    return v


def get_or_extrapolate(row, col, side, velocities):
    try:
        return velocities[row, col, side]
    except StaggeredGridIndexError:
        return extrapolate(row, col, side, velocities)


def extrapolate(row, col, side, velocities):
    closest_row = np.clip(row, 0, velocities.grid_dim - 1)
    closest_col = np.clip(col, 0, velocities.grid_dim - 1)

    if row == closest_row:
        # Extrapolate in the x-direction
        return extrapolate_horizontally(row, closest_col, side, velocities, abs(closest_col - col))

    elif col == closest_col:
        # Extrapolate in the y-direction
        return extrapolate_vertically(closest_row, col, side, velocities, abs(closest_row - row))
    else:
        # Extrapolate in both directions and interpolate
        horizontal_steps = abs(col - closest_col)
        vertical_steps = abs(row - closest_row)

        h = extrapolate_horizontally(closest_row, closest_col, side, velocities, horizontal_steps + vertical_steps)
        v = extrapolate_vertically(closest_row, closest_col, side, velocities, horizontal_steps + vertical_steps)

        return (horizontal_steps * h + vertical_steps * v) / (horizontal_steps + vertical_steps)

        pass


def extrapolate_horizontally(row, closest_col, side, velocities, steps):
    # check if we are at the left or right boundary
    if closest_col == 0:
        # we are at the left boundary
        # if the side we are looking for is right, we take a step less and use the left side
        if side == velocities.RIGHT:
            steps -= 1
            side = velocities.LEFT

        du = (
                velocities[row, closest_col, side]
                - velocities[row, closest_col + 1, side]
        )
        return velocities[row, closest_col, side] + du * steps
    elif closest_col == velocities.grid_dim - 1:
        # we are at the right boundary
        # if the side we are looking for is left, we take a step less and use the right side
        if side == velocities.LEFT:
            steps -= 1
            side = velocities.RIGHT
        du = (
                velocities[row, closest_col - 1, side]
                - velocities[row, closest_col, side]
        )
        return velocities[row, closest_col, side] + du * -steps
    else:
        raise ValueError("This is not a boundary cell")


def extrapolate_vertically(closest_row, col, side, velocities, steps):
    # check if we are at the top or bottom boundary
    if closest_row == 0:
        # we are at the top boundary
        # if the side we are looking for is bottom, we take a step less and use the top side
        if side == velocities.BOTTOM:
            steps -= 1
            side = velocities.TOP

        dv = (
                velocities[closest_row, col, side]
                - velocities[closest_row + 1, col, side]
        )
        return velocities[closest_row, col, side] + dv * steps
    elif closest_row == velocities.grid_dim - 1:
        # we are at the bottom boundary
        # if the side we are looking for is top, we take a step less and use the bottom side
        if side == velocities.TOP:
            steps -= 1
            side = velocities.BOTTOM

        dv = (
                velocities[closest_row - 1, col, side]
                - velocities[closest_row, col, side]
        )
        return velocities[closest_row, col, side] + dv * -steps
    else:
        raise ValueError("This is not a boundary cell")
