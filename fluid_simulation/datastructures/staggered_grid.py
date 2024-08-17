import numpy as np

from datastructures.errors import StaggeredGridIndexError


class StaggeredGrid:
    TOP = 0
    RIGHT = 1
    BOTTOM = 2
    LEFT = 3

    def __init__(self, grid_dim: int):
        """
        Initializes a staggered grid with zeros
        :param grid_dim: the grid dimension
        """
        self.grid_dim = grid_dim
        self.u = np.zeros((grid_dim, grid_dim + 1)) # u velocity component (x-axis)
        self.v = np.zeros((grid_dim + 1, grid_dim)) # v velocity component (y-axis)

    def top(self, row, col):
        self.test_bounds(row, col, self.TOP)
        return self.v[row, col]

    def set_top(self, row, col, value):
        self.test_bounds(row, col, self.TOP)
        self.v[row, col] = value

    def right(self, row, col):
        self.test_bounds(row, col, self.RIGHT)
        return self.u[row, col + 1]

    def set_right(self, row, col, value):
        self.test_bounds(row, col, self.RIGHT)
        self.u[row, col + 1] = value

    def bottom(self, row, col):
        self.test_bounds(row, col, self.BOTTOM)
        return self.v[row + 1, col]

    def set_bottom(self, row, col, value):
        self.test_bounds(row, col, self.BOTTOM)
        self.v[row + 1, col] = value

    def left(self, row, col):
        self.test_bounds(row, col, self.LEFT)
        return self.u[row, col]

    def set_left(self, row, col, value):
        self.test_bounds(row, col, self.LEFT)
        self.u[row, col] = value

    def __getitem__(self, key):
        if len(key) == 2:
            row, col = key
            side = -1  # Default value for side
        else:
            row, col, side = key

        if side == -1:
            return self.top(row, col), self.right(row, col), self.bottom(row, col), self.left(row, col)
        elif side == self.TOP:
            return self.top(row, col)
        elif side == self.RIGHT:
            return self.right(row, col)
        elif side == self.BOTTOM:
            return self.bottom(row, col)
        elif side == self.LEFT:
            return self.left(row, col)
        else:
            raise ValueError("Invalid side")

    def __setitem__(self, key, value):
        row, col, side = key
        if side == self.TOP:
            self.v[row, col] = value
        elif side == self.RIGHT:
            self.u[row, col + 1] = value
        elif side == self.BOTTOM:
            self.v[row + 1, col] = value
        elif side == self.LEFT:
            self.u[row, col] = value
        else:
            raise ValueError("Invalid side")

    def coords(self, row, col, side):
        """
        Returns the coordinates of the center of the side of the cell at (row, col)
        :param row: row index
        :param col: column index
        :param side: which side of the cell
        :return: the coordinates (x, y) of the center of the side of the cell at (row, col)
        """

        if side == self.TOP:
            return col, row - 0.5

        if side == self.RIGHT:
            return col + 0.5, row

        if side == self.BOTTOM:
            return col, row + 0.5

        if side == self.LEFT:
            return col - 0.5, row

        raise ValueError("Invalid side")

    def to_regular_grid(self):
        """
        Converts the staggered grid to a regular grid, by averaging the velocities of the sides
        :return: a regular grid of shape (n, n, 2) where n is the grid dimension
        """
        grid_dim = self.u.shape[0]
        grid = np.zeros((grid_dim, grid_dim, 2))
        for row in range(grid_dim):
            for col in range(grid_dim):
                grid[row, col] = (
                    (self.left(row, col) + self.right(row, col)) / 2,
                    (self.top(row, col) + self.bottom(row, col)) / 2
                )

        return grid

    def test_bounds(self, row, col, side):
        if row < -1 or row > self.grid_dim:
            raise StaggeredGridIndexError("Row index out of bounds")

        if col < -1 or col > self.grid_dim:
            raise StaggeredGridIndexError("Column index out of bounds")

        if row == -1 and side != self.BOTTOM:
            raise StaggeredGridIndexError("Your row is out of regular bounds. You can only access the bottom side!")

        if row == self.grid_dim and side != self.TOP:
            raise StaggeredGridIndexError("Your row is out of regular bounds. You can only access the top side!")

        if col == -1 and side != self.RIGHT:
            raise StaggeredGridIndexError("Your column is out of regular bounds. You can only access the right side!")

        if col == self.grid_dim and side != self.LEFT:
            raise StaggeredGridIndexError("Your column is out of regular bounds. You can only access the left side!")

def from_regular_grid(grid) -> StaggeredGrid:
    """
    Initializes a staggered grid from a regular grid, by averaging the velocities of the cells
    :param grid: a regular grid of shape (n, n, 2) where n is the grid dimension
    """
    assert grid.shape[0] == grid.shape[1]
    grid_dim = grid.shape[0]

    s = StaggeredGrid(grid_dim)

    s.u = np.zeros((grid_dim, grid_dim + 1))  # u velocity component (x-axis)
    s.v = np.zeros((grid_dim + 1, grid_dim))  # v velocity component (y-axis)
    for row in range(grid_dim):
        for col in range(grid_dim):
            if col != grid_dim - 1:
                s.u[row, col + 1] = (grid[row, col][0] + grid[row, col + 1][0]) / 2
            if row != grid_dim - 1:
                s.v[row + 1, col] = (grid[row, col][1] + grid[row + 1, col][1]) / 2

    return s