import unittest
import numpy as np

from advection import interpolate_u, interpolate_v, extrapolate_horizontally, extrapolate_vertically, extrapolate
from datastructures.staggered_grid import StaggeredGrid
from datastructures.errors import StaggeredGridIndexError


class TestAdvection(unittest.TestCase):
    def test_interpolate_u(self):
        s = StaggeredGrid(2)
        s.u = np.array([[1, 2, 3], [4, 5, 6]])
        s.v = np.array([[1, 2], [3, 4], [5, 6]])

        self.assertEqual(interpolate_u(0.25, 0.25, s), 2.5)
        self.assertEqual(interpolate_u(0.75, 0.25, s), 3)
        self.assertEqual(interpolate_u(0.25, 0.75, s), 4)
        self.assertEqual(interpolate_u(0.75, 0.75, s), 4.5)

        with self.assertRaises(StaggeredGridIndexError): # TODO remove when extrapolation is implemented
            interpolate_u(0.25, 1.25, s)
        with self.assertRaises(StaggeredGridIndexError):
            interpolate_u(1.75, 0.25, s)

    def test_interpolate_v(self):
        s = StaggeredGrid(2)
        s.u = np.array([[1, 2, 3], [4, 5, 6]])
        s.v = np.array([[1, 2], [3, 4], [5, 6]])

        self.assertEqual(interpolate_v(0.25, 0.25, s), 2.75)
        self.assertEqual(interpolate_v(0.75, 0.25, s), 3.25)
        self.assertEqual(interpolate_v(0.25, 0.75, s), 3.75)
        self.assertEqual(interpolate_v(0.75, 0.75, s), 4.25)

        with self.assertRaises(StaggeredGridIndexError): # TODO remove when extrapolation is implemented
            interpolate_v(0.25, 1.75, s)
        with self.assertRaises(StaggeredGridIndexError):
            interpolate_v(1.25, 0.25, s)

    def test_extrapolate_horizontally(self):
        s = StaggeredGrid(2)
        s.u = np.array([[1, 2, 3], [4, 5, 6]])
        s.v = np.array([[1, 2], [3, 4], [5, 6]])

        self.assertEqual(extrapolate_horizontally(0, 1, s.RIGHT, s, 1), 4)
        self.assertEqual(extrapolate_horizontally(0, 1, s.RIGHT, s, 2), 5)

        self.assertEqual(extrapolate_horizontally(0, 0, s.TOP, s, 1), 0)
        self.assertEqual(extrapolate_horizontally(0, 1, s.TOP, s, 1), 3)

        self.assertEqual(extrapolate_horizontally(0, 0, s.BOTTOM, s, 1), 2)
        self.assertEqual(extrapolate_horizontally(0, 1, s.BOTTOM, s, 1), 5)

        self.assertEqual(extrapolate_horizontally(0, 0, s.LEFT, s, 1), 0)

    def test_extrapolate_vertically(self):
        s = StaggeredGrid(2)
        s.u = np.array([[1, 2, 3], [4, 5, 6]])
        s.v = np.array([[1, 2], [3, 4], [5, 6]])

        self.assertEqual(extrapolate_vertically(0, 0, s.TOP, s, 1), -1)
        self.assertEqual(extrapolate_vertically(1, 0, s.BOTTOM, s, 1), 7)

        self.assertEqual(extrapolate_vertically(0, 0, s.RIGHT, s, 1), -1)
        self.assertEqual(extrapolate_vertically(1, 0, s.RIGHT, s, 1), 8)

        self.assertEqual(extrapolate_vertically(0, 0, s.LEFT, s, 1), -2)
        self.assertEqual(extrapolate_vertically(1, 0, s.LEFT, s, 1), 7)

    def test_extrapolate(self):
        s = StaggeredGrid(2)
        s.u = np.array([[1, 2, 3], [4, 5, 6]])
        s.v = np.array([[1, 2], [3, 4], [5, 6]])

        self.assertEqual(extrapolate(0, -1, s.LEFT, s), 0)
        self.assertEqual(extrapolate(0, 2, s.RIGHT, s), 4)

        self.assertEqual(extrapolate(-1, 0, s.TOP, s), -1)
        self.assertEqual(extrapolate(2, 0, s.BOTTOM, s), 7)

        self.assertEqual(extrapolate(-1, -1, s.LEFT, s), -3)


if __name__ == '__main__':
    unittest.main()
