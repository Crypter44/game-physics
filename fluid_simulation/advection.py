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
    end_x, end_y = velocities.coords(row, col, side)
    if side == velocities.LEFT or side == velocities.RIGHT:
        u = velocities[row, col, side]
        v = velocities.top(row, col)


def interpolate_u(start_x, start_y, velocities):
    # TODO: Add bounds checking for each velocity access, and handle extrapolation
    row, col = int(np.round(start_y)), int(np.round(start_x))
    left1 = velocities[row, col, velocities.LEFT]
    pos_l1 = velocities.coords(row, col, velocities.LEFT)

    right1 = velocities[row, col, velocities.RIGHT]
    pos_r1 = velocities.coords(row, col, velocities.RIGHT)

    if start_y > row:
        left2 = velocities[row + 1, col, velocities.LEFT]
        pos_l2 = velocities.coords(row + 1, col, velocities.LEFT)
        right2 = velocities[row + 1, col, velocities.RIGHT]
    else:
        left2 = velocities[row - 1, col, velocities.LEFT]
        pos_l2 = velocities.coords(row - 1, col, velocities.LEFT)
        right2 = velocities[row - 1, col, velocities.RIGHT]

    # weights (alpha, 1-alpha) for linear interpolation along x-axis
    alpha = (start_x - pos_l1[0]) / (pos_r1[0] - pos_l1[0])

    # weights (beta, 1-beta) for linear interpolation along y-axis
    beta = (start_y - pos_l1[1]) / (pos_l2[1] - pos_l1[1])

    # linear interpolation along x-axis
    x_vel1 = left1 * (1 - alpha) + right1 * alpha
    x_vel2 = left2 * (1 - alpha) + right2 * alpha

    # linear interpolation along y-axis
    u = x_vel1 * (1 - beta) + x_vel2 * beta
    return u


def interpolate_v(start_x, start_y, velocities):
    row, col = int(np.round(start_y)), int(np.round(start_x))
    top1 = velocities[row, col, velocities.TOP]
    pos_t1 = velocities.coords(row, col, velocities.TOP)

    bottom1 = velocities[row, col, velocities.BOTTOM]
    pos_b1 = velocities.coords(row, col, velocities.BOTTOM)

    if start_x > col:
        top2 = velocities[row, col + 1, velocities.TOP]
        bottom2 = velocities[row, col + 1, velocities.BOTTOM]
        pos_t2 = velocities.coords(row, col + 1, velocities.TOP)
    else:
        top2 = velocities[row, col - 1, velocities.TOP]
        bottom2 = velocities[row, col - 1, velocities.BOTTOM]
        pos_t2 = velocities.coords(row, col - 1, velocities.TOP)

    # weights (alpha, 1-alpha) for linear interpolation along y-axis
    alpha = (start_y - pos_t1[1]) / (pos_b1[1] - pos_t1[1])

    # weights (beta, 1-beta) for linear interpolation along x-axis
    beta = (start_x - pos_t1[0]) / (pos_t2[0] - pos_t1[0])

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
        return extrapolate_vertically(closest_row, col, side, velocities, abs(closest_row - row))  # TODO check if negative
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
    # TODO: ensure, that if we are at the left boundary, left is used,
    #  and if we are at the right boundary, right is used,
    #  top and bottom do not matter,
    #  for that change steps accordingly

    if closest_col == 0:
        du = (
                velocities[row, closest_col, side]
                - velocities[row, closest_col + 1, side]
        )
        return velocities[row, closest_col, side] + du * steps
    elif closest_col == velocities.grid_dim - 1:
        du = (
                velocities[row, closest_col - 1, side]
                - velocities[row, closest_col, side]
        )
        return velocities[row, closest_col, side] + du * -steps
    else:
        raise ValueError("This is not a boundary cell")


def extrapolate_vertically(closest_row, col, side, velocities, steps):
    # check if we are at the top or bottom boundary
    # TODO: ensure, that if we are at the top boundary, top is used,
    #  and if we are at the bottom boundary, bottom is used,
    #  left and right do not matter,
    #  for that change steps accordingly
    if closest_row == 0:
        dv = (
                velocities[closest_row, col, side]
                - velocities[closest_row + 1, col, side]
        )
        return velocities[closest_row, col, side] + dv * steps
    elif closest_row == velocities.grid_dim - 1:
        dv = (
                velocities[closest_row - 1, col, side]
                - velocities[closest_row, col, side]
        )
        return velocities[closest_row, col, side] + dv * -steps
    else:
        raise ValueError("This is not a boundary cell")
