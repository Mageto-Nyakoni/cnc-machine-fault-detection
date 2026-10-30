"""Check vibration features against signals with known statistics."""

import unittest

import numpy as np

from utils.data_loader_utils import extract_window_features


class WindowFeatureTests(unittest.TestCase):
    def test_sine_wave_statistics(self):
        time = np.arange(4000) / 2000
        signal = np.sin(2 * np.pi * 10 * time)
        result = extract_window_features(np.column_stack([signal] * 3))

        self.assertEqual(len(result), 15)
        for axis in ("X", "Y", "Z"):
            self.assertAlmostEqual(result[f"{axis}_mean"], 0)
            self.assertAlmostEqual(result[f"{axis}_std"], 1 / np.sqrt(2))
            self.assertAlmostEqual(result[f"{axis}_rms"], 1 / np.sqrt(2))
            self.assertAlmostEqual(result[f"{axis}_peak_to_peak"], 2)
            self.assertAlmostEqual(result[f"{axis}_kurtosis"], -1.5)

    def test_constant_axes_produce_finite_features(self):
        values = np.tile([3, -4, 0], (4000, 1))
        result = extract_window_features(values)

        self.assertTrue(np.isfinite(list(result.values())).all())
        for axis, mean in zip(("X", "Y", "Z"), (3, -4, 0)):
            self.assertEqual(result[f"{axis}_mean"], mean)
            self.assertEqual(result[f"{axis}_std"], 0)
            self.assertEqual(result[f"{axis}_rms"], abs(mean))
            self.assertEqual(result[f"{axis}_peak_to_peak"], 0)
            self.assertEqual(result[f"{axis}_kurtosis"], 0)

    def test_integer_samples_do_not_overflow_when_squared(self):
        values = np.full((4000, 3), 30000, dtype=np.int16)
        result = extract_window_features(values)
        self.assertEqual(result["X_rms"], 30000)

    def test_invalid_shapes_are_rejected(self):
        for shape in ((0, 3), (4000, 2), (4000,), (2, 3, 1)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                extract_window_features(np.zeros(shape))

    def test_nonfinite_samples_are_rejected(self):
        for invalid in (np.nan, np.inf, -np.inf):
            values = np.zeros((4000, 3))
            values[0, 0] = invalid
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                extract_window_features(values)


if __name__ == "__main__":
    unittest.main()
