import unittest
import numpy as np

from advection import interpolate_u, interpolate_v, extrapolate_horizontally, extrapolate_vertically, extrapolate
from datastructures.staggered_grid import StaggeredGrid
from datastructures.errors import StaggeredGridIndexError


class AdvectionTest(unittest.TestCase):
    def test_interpolate_u(self):
        s = StaggeredGrid(2)
        s.u = np.array([[1, 2, 3], [4, 5, 6]])
        s.v = np.array([[1, 2], [3, 4], [5, 6]])

        self.assertEqual(interpolate_u(0.25, 0.25, s), 2.5)
        self.assertEqual(interpolate_u(0.75, 0.25, s), 3)
        self.assertEqual(interpolate_u(0.25, 0.75, s), 4)
        self.assertEqual(interpolate_u(0.75, 0.75, s), 4.5)

        self.assertEqual(interpolate_u(0.25, 1.25, s), 5.5)
        self.assertEqual(interpolate_u(1.75, 0.25, s), 4)

    def test_interpolate_v(self):
        s = StaggeredGrid(2)
        s.u = np.array([[1, 2, 3], [4, 5, 6]])
        s.v = np.array([[1, 2], [3, 4], [5, 6]])

        self.assertEqual(interpolate_v(0.25, 0.25, s), 2.75)
        self.assertEqual(interpolate_v(0.75, 0.25, s), 3.25)
        self.assertEqual(interpolate_v(0.25, 0.75, s), 3.75)
        self.assertEqual(interpolate_v(0.75, 0.75, s), 4.25)

        self.assertEqual(interpolate_v(0.25, 1.75, s), 5.75)
        self.assertEqual(interpolate_v(1.25, 0.25, s), 3.75)

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
        self.assertEqual(extrapolate(0, 0, s.LEFT, s), 1)

    def test_extrapolate_u_to_bigger_grid(self):
        s = StaggeredGrid(2)
        s.u = np.array([[1, 2, 3], [4, 5, 6]])

        u_ext_left = np.zeros((6, 7))
        u_ext_right = np.zeros((6, 7))

        for i in range(6):
            for j in range(7):
                u_ext_left[i, j] = extrapolate(i - 2, j - 2, s.LEFT, s)

        for i in range(6):
            for j in range(7):
                u_ext_right[i, j] = extrapolate(i - 2, j - 2, s.RIGHT, s)

        self.assertTrue(np.array_equal(
            np.array([[-7, -6, -5, -4, -3, -2, -1],
                      [-4, -3, -2, -1, 0, 1, 2],
                      [-1, 0, 1, 2, 3, 4, 5],
                      [2, 3, 4, 5, 6, 7, 8],
                      [5, 6, 7, 8, 9, 10, 11],
                      [8, 9, 10., 11, 12, 13, 14]]),
            u_ext_left
        ))

        self.assertTrue(np.array_equal(
            np.array([[-6., -5., -4., -3., -2., -1., 0.],
                      [-3., -2., -1., 0., 1., 2., 3.],
                      [0., 1., 2., 3., 4., 5., 6.],
                      [3., 4., 5., 6., 7., 8., 9.],
                      [6., 7., 8., 9., 10., 11., 12.],
                      [9., 10., 11., 12., 13., 14., 15.]]),
            u_ext_right
        ))


    def test_extrapolate_v_to_bigger_grid(self):
        s = StaggeredGrid(2)
        s.v = np.array([[1, 2], [3, 4], [5, 6]])

        v_ext_top = np.zeros((7, 6))
        v_ext_bottom = np.zeros((7, 6))

        for i in range(7):
            for j in range(6):
                v_ext_top[i, j] = extrapolate(i - 2, j - 2, s.TOP, s)

        for i in range(7):
            for j in range(6):
                v_ext_bottom[i, j] = extrapolate(i - 2, j - 2, s.BOTTOM, s)

        self.assertTrue(np.array_equal(
            np.array([[-5., -4., -3., -2., -1., 0.],
                      [-3., -2., -1., 0., 1., 2.],
                      [-1., 0., 1., 2., 3., 4.],
                      [1., 2., 3., 4., 5., 6.],
                      [3., 4., 5., 6., 7., 8.],
                      [5., 6., 7., 8., 9., 10.],
                      [7., 8., 9., 10., 11., 12.]]),
            v_ext_top
        ))

        self.assertTrue(np.array_equal(
            np.array([[-3., -2., -1., 0., 1., 2.],
                      [-1., 0., 1., 2., 3., 4.],
                      [1., 2., 3., 4., 5., 6.],
                      [3., 4., 5., 6., 7., 8.],
                      [5., 6., 7., 8., 9., 10.],
                      [7., 8., 9., 10., 11., 12.],
                      [9., 10., 11., 12., 13., 14.]]),
            v_ext_bottom
        ))

if __name__ == '__main__':
    unittest.main()
