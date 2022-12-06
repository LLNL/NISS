import sys
import numpy as np

import ml.smoother

sys.path.append('../../NISS')
import torch # noqa: E402
import unittest  # noqa: E402
import lfa  # noqa: E402
import utils  # noqa: E402


class TestTensorUtils(unittest.TestCase):
    def test_complex_multiply(self):
        # complex_multiply: real
        x = 2 * torch.ones(1, 4)
        y = 3 * torch.ones(1, 4)
        z = utils.complex_multiply(x, y)
        self.assertTrue(z.size(0) == 1)
        self.assertTrue(torch.all(z == 6))
        # complex_multiply: complex
        x = torch.ones(2, 2)
        y = torch.ones(2, 2)
        z = utils.complex_multiply(x, y)
        self.assertTrue(z.size(0) == 2)
        self.assertTrue(torch.all(z[0, :] == 0))
        self.assertTrue(torch.all(z[1, :] == 2))

    def test_centrosymmetric_strict_upper_coord(self):
        x, y = utils.centrosymmetric_strict_upper_coord(1)
        self.assertTrue(np.array_equal(x, []))
        self.assertTrue(np.array_equal(y, []))
        x, y = utils.centrosymmetric_strict_upper_coord(2)
        self.assertTrue(np.array_equal(x, [1, 1]))
        self.assertTrue(np.array_equal(y, [0, 1]))
        x, y = utils.centrosymmetric_strict_upper_coord(3)
        self.assertTrue(np.array_equal(x, [1, 2, 2, 2]))
        self.assertTrue(np.array_equal(y, [2, 0, 1, 2]))

    def test_centrosymmetric_strict_lower_coord(self):
        x, y = utils.centrosymmetric_strict_lower_coord(1)
        self.assertTrue(np.array_equal(x, []))
        self.assertTrue(np.array_equal(y, []))
        x, y = utils.centrosymmetric_strict_lower_coord(2)
        self.assertTrue(np.array_equal(x, [0, 0]))
        self.assertTrue(np.array_equal(y, [0, 1]))
        x, y = utils.centrosymmetric_strict_lower_coord(3)
        self.assertTrue(np.array_equal(x, [0, 0, 0, 1]))
        self.assertTrue(np.array_equal(y, [0, 1, 2, 0]))

    def test_centrosymmetric_upper_coord(self):
        x, y = utils.centrosymmetric_upper_coord(1)
        self.assertTrue(np.array_equal(x, [0]))
        self.assertTrue(np.array_equal(y, [0]))
        x, y = utils.centrosymmetric_upper_coord(2)
        self.assertTrue(np.array_equal(x, [1, 1]))
        self.assertTrue(np.array_equal(y, [0, 1]))
        x, y = utils.centrosymmetric_upper_coord(3)
        self.assertTrue(np.array_equal(x, [1, 1, 2, 2, 2]))
        self.assertTrue(np.array_equal(y, [1, 2, 0, 1, 2]))

    def test_centrosymmetric_lower_coord(self):
        x, y = utils.centrosymmetric_lower_coord(1)
        self.assertTrue(np.array_equal(x, [0]))
        self.assertTrue(np.array_equal(y, [0]))
        x, y = utils.centrosymmetric_lower_coord(2)
        self.assertTrue(np.array_equal(x, [0, 0]))
        self.assertTrue(np.array_equal(y, [0, 1]))
        x, y = utils.centrosymmetric_lower_coord(3)
        self.assertTrue(np.array_equal(x, [0, 0, 0, 1, 1]))
        self.assertTrue(np.array_equal(y, [0, 1, 2, 0, 1]))

    def test_centrosymmetrize_upper(self):
        x = torch.tensor([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])
        utils.centrosymmetrize_upper(x)
        y = torch.tensor([[1, 2, 3],
                          [4, 5, 4],
                          [3, 2, 1]])
        self.assertTrue(np.array_equal(x.numpy(), y.numpy()))


class TestLFASmooth(unittest.TestCase):
    def test_main(self):
        err, ret = lfa.smooth.main(do_print=False, do_plot=False)
        self.assertEqual(err, 0)
        self.assertAlmostEqual(ret[0], 0.3332664966583252)
        self.assertAlmostEqual(ret[1], 0.05882382392883301)


class TestLFAStencil(unittest.TestCase):
    def test_main(self):
        err, ret = lfa.stencil.main(do_print=False, do_plot=False)
        self.assertEqual(err, 0)
        self.assertAlmostEqual(ret[0], 3.9997992515563965)
        self.assertAlmostEqual(ret[1], 0.0003012418746948242)


class TestLFATheta(unittest.TestCase):
    def test_theta_grid(self):
        theta = lfa.Theta2D(num_theta=4, start=0, end=torch.pi, quadrant=torch.tensor([0, 1, 2, 3]))
        self.assertEqual(theta.theta.size(0), 4)
        self.assertEqual(theta.theta_grid.size(0), 4)
        self.assertEqual(theta.theta_grid.size(1), 4)
        self.assertEqual(theta.theta_grid.size(2), 2)
        self.assertTrue(np.array_equal(theta.theta_quad[:, :, :, 0], theta.theta_grid))
        self.assertTrue(np.array_equal(theta.theta_quad[:, :, :, 1], theta.theta_grid + torch.tensor([torch.pi, 0])))
        self.assertTrue(np.array_equal(theta.theta_quad[:, :, :, 2], theta.theta_grid + torch.tensor([0, torch.pi])))
        self.assertTrue(np.array_equal(theta.theta_quad[:, :, :, 3], theta.theta_grid + torch.pi))


class TestMLSmoother(unittest.TestCase):
    def test_ml_smoother(self):
        err, ret = ml.smoother.main(smoother_size=3, num_steps=200, do_print=False, do_plot=False)
        self.assertEqual(err, 0)
        self.assertAlmostEqual(ret[0], 5.88238239e-02)
        self.assertAlmostEqual(ret[1], 6.30908394e+00)
        self.assertAlmostEqual(ret[2], 6.30908394e+00)


if __name__ == '__main__':
    unittest.main()
