"""Regression checks for the saved Keras weights loaded by PyTorch."""

import unittest
from pathlib import Path

import numpy as np

from torch_classifier import grad_cam_heatmap, load_classifier, predict_probabilities


class TorchClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = Path(__file__).resolve().parents[1] / "brain_tumor.h5"
        cls.model = load_classifier(model_path)
        rng = np.random.default_rng(17)
        cls.image = (
            rng.integers(0, 256, size=(1, 150, 150, 3), dtype=np.uint8)
            .astype(np.float64) / 255.0
        )

    def test_saved_keras_prediction_is_preserved(self):
        # Reference from brain_tumor.h5 loaded with Keras and the same input.
        expected = np.array([
            2.6220527615805622e-06,
            0.23470132052898407,
            0.7652742266654968,
            2.1756566638941877e-05,
        ])
        actual = predict_probabilities(self.model, self.image)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_grad_cam_has_finite_image_sized_values(self):
        probabilities = predict_probabilities(self.model, self.image)
        heatmap = grad_cam_heatmap(
            self.model, self.image, int(probabilities.argmax()), (80, 120)
        )
        self.assertEqual(heatmap.shape, (80, 120))
        self.assertTrue(np.isfinite(heatmap).all())
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
