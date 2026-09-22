import unittest

from evaluation import calculate_metrics


class EvaluationMetricsTests(unittest.TestCase):
    def test_metrics_include_all_classes_and_zero_division(self):
        metrics, matrix = calculate_metrics([0, 0, 1, 1], [0, 1, 1, 1])

        self.assertEqual(matrix.tolist(), [
            [1, 1, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["per_class"]["Glioma"]["precision"], 1.0)
        self.assertAlmostEqual(metrics["per_class"]["Glioma"]["recall"], 0.5)
        self.assertAlmostEqual(metrics["per_class"]["No tumor"]["f1"], 0.0)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 0.8) / 4)
        self.assertAlmostEqual(metrics["weighted_f1"], (2 * (2 / 3) + 2 * 0.8) / 4)


if __name__ == "__main__":
    unittest.main()
