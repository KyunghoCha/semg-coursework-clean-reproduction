import unittest
import numpy as np
from ar_features import ar_features


class ARTests(unittest.TestCase):
    def test_independent_least_squares(self):
        w = np.random.default_rng(918).normal(size=(3, 200, 5))
        actual = ar_features(w)
        for i in range(3):
            for c in range(5):
                x = w[i, :, c] - w[i, :, c].mean()
                d = np.array([[x[t-k] for k in range(1, 7)] for t in range(6, 200)])
                b = np.linalg.lstsq(d, x[6:], rcond=1e-12)[0]
                np.testing.assert_allclose(actual[i, c*6:c*6+6], b, atol=1e-12)
                self.assertAlmostEqual(actual[i, 30+c], np.sqrt(np.mean((x[6:]-d@b)**2)), places=12)

    def test_zero_constant_and_channel_order(self):
        for w in [np.zeros((2,200,5)), np.full((2,200,5), .3)]:
            np.testing.assert_array_equal(ar_features(w), np.zeros((2,35)))
        w = np.random.default_rng(42).normal(size=(2,200,5))
        a = ar_features(w); b = ar_features(w[:,:,::-1])
        np.testing.assert_allclose(a[:,:30].reshape(2,5,6)[:,::-1], b[:,:30].reshape(2,5,6), atol=1e-12)
        np.testing.assert_allclose(a[:,30:][:,::-1], b[:,30:], atol=1e-12)

    def test_scale_offset_and_window_locality(self):
        w = np.random.default_rng(14).normal(size=(4,200,5))
        a = ar_features(w); b = ar_features(w*7 + 11)
        np.testing.assert_allclose(a[:,:30], b[:,:30], atol=1e-12)
        np.testing.assert_allclose(a[:,30:]*7, b[:,30:], atol=1e-12)
        np.testing.assert_allclose(a[:1], ar_features(w[:1]), atol=1e-12)
        w[1:] = 1e5
        np.testing.assert_array_equal(a[:1], ar_features(w)[:1])
        with self.assertRaises(ValueError):ar_features(np.zeros((1,201,5)))
        w[0,0,0] = np.nan
        with self.assertRaises(ValueError):ar_features(w)


if __name__ == "__main__":unittest.main()
