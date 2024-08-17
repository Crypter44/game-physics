import unittest
import numpy as np

import staggered_grid as sg
from staggered_grid import StaggeredGrid


class MyTestCase(unittest.TestCase):
    grid = np.array(
        [[[1, 1], [2, 2]],
         [[3, 3], [4, 4]]]
    )

    def test_constructor(self):
        staggered_grid = sg.from_regular_grid(self.grid)

        assert staggered_grid[0, 0] == (0, 1.5, 2, 0), \
            f"Construction failed for (0,0): expected (0, 1.5, 2, 0), but got {staggered_grid[0, 0]}"
        assert staggered_grid[0, 1] == (0, 0, 3, 1.5), \
            f"Construction failed for (0,1): expected (0, 0, 3, 1.5), but got {staggered_grid[0, 1]}"
        assert staggered_grid[1, 0] == (2, 3.5, 0, 0), \
            f"Construction failed for (1,0): expected (2, 3.5, 0, 0), but got {staggered_grid[1, 0]}"
        assert staggered_grid[1, 1] == (3, 0, 0, 3.5), \
            f"Construction failed for (1,1): expected (3, 0, 0, 3.5), but got {staggered_grid[1, 1]}"

    def test_to_regular_grid(self):
        staggered_grid = sg.from_regular_grid(self.grid)
        regular_grid = staggered_grid.to_regular_grid()

        assert np.array_equal(regular_grid[0, 0], [0.75, 1]), \
            f"Conversion failed for (0,0): expected [0.75, 1], but got {regular_grid[0, 0]}"
        assert np.array_equal(regular_grid[0, 1], [0.75, 1.5]), \
            f"Conversion failed for (0,1): expected [0.75, 1.5], but got {regular_grid[0, 1]}"
        assert np.array_equal(regular_grid[1, 0], [1.75, 1]), \
            f"Conversion failed for (1,0): expected [1.75, 1], but got {regular_grid[1, 0]}"
        assert np.array_equal(regular_grid[1, 1], [1.75, 1.5]), \
            f"Conversion failed for (1,1): expected [1.75, 1.5], but got {regular_grid[1, 1]}"

    def test_bounds(self):
        staggered_grid = StaggeredGrid(1)
        with self.assertRaises(ValueError):
            staggered_grid.test_bounds(-2, 0, staggered_grid.BOTTOM)
        with self.assertRaises(ValueError):
            staggered_grid.test_bounds(0, -2, staggered_grid.RIGHT)
        with self.assertRaises(ValueError):
            staggered_grid.test_bounds(2, 0, staggered_grid.TOP)
        with self.assertRaises(ValueError):
            staggered_grid.test_bounds(0, 2, staggered_grid.LEFT)
        with self.assertRaises(ValueError):
            staggered_grid.test_bounds(-1, 0, staggered_grid.TOP)
        with self.assertRaises(ValueError):
            staggered_grid.test_bounds(0, -1, staggered_grid.LEFT)
        with self.assertRaises(ValueError):
            staggered_grid.test_bounds(1, 0, staggered_grid.BOTTOM)
        with self.assertRaises(ValueError):
            staggered_grid.test_bounds(0, 1, staggered_grid.RIGHT)
        with self.assertRaises(ValueError):
            staggered_grid.test_bounds(1, 1, staggered_grid.TOP)

        staggered_grid.test_bounds(0, 0, staggered_grid.BOTTOM)
        staggered_grid.test_bounds(0, 0, staggered_grid.RIGHT)
        staggered_grid.test_bounds(0, 0, staggered_grid.TOP)
        staggered_grid.test_bounds(0, 0, staggered_grid.LEFT)

        staggered_grid.test_bounds(1, 0, staggered_grid.TOP)
        staggered_grid.test_bounds(0, 1, staggered_grid.LEFT)
        staggered_grid.test_bounds(0, -1, staggered_grid.RIGHT)
        staggered_grid.test_bounds(-1, 0, staggered_grid.BOTTOM)








if __name__ == '__main__':
    unittest.main()
