import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from image_processing import nearest_neighbor, correlate, equalize_histogram, binary_morphology, load_sample

try:
    import cv2 as cv
except ImportError:
    cv = None

class AlgorithmTests(unittest.TestCase):
    def test_nearest_neighbor_repeats_selected_pixels(self):
        image = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(nearest_neighbor(image, 4, 4), np.repeat(np.repeat(image, 2, 0), 2, 1))

    def test_correlation_keeps_negative_edge_response(self):
        result = correlate(np.array([[0, 10, 0]]), np.array([[-1, 0, 1]]))
        np.testing.assert_array_equal(result, [[10, 0, -10]])

    def test_equalization_uses_first_occupied_histogram_bin(self):
        image = np.array([[50, 50], [100, 200]], dtype=np.uint8)
        np.testing.assert_array_equal(equalize_histogram(image), [[0, 0], [128, 255]])

    def test_equalization_preserves_constant_image(self):
        image = np.full((3, 3), 123, dtype=np.uint8)
        np.testing.assert_array_equal(equalize_histogram(image), image)

    def test_cross_dilation_of_single_foreground_pixel(self):
        image = np.zeros((5, 5), dtype=bool)
        image[2, 2] = True
        cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        result = binary_morphology(image, cross, "dilate")
        self.assertEqual(int(result.sum()), 5)
        np.testing.assert_array_equal(result[1:4, 1:4], cross.astype(bool))

    def test_erosion_has_zero_border(self):
        result = binary_morphology(np.ones((3, 3)), np.ones((3, 3)), "erode")
        expected = np.zeros((3, 3), dtype=bool)
        expected[1, 1] = True
        np.testing.assert_array_equal(result, expected)

    def test_samples_load_with_correct_dimensions(self):
        self.assertEqual(load_sample().shape, (128, 128))
        self.assertEqual(load_sample("color").shape, (128, 128, 3))

    @unittest.skipIf(cv is None, "OpenCV is not installed")
    def test_opencv_filtering_and_morphology_comparisons(self):
        image = load_sample()
        kernel = np.array([[-1., 0., 1.]])
        np.testing.assert_allclose(correlate(image, kernel), cv.filter2D(image, cv.CV_64F, kernel, borderType=cv.BORDER_CONSTANT))
        binary = load_sample("binary") > 128
        square = np.ones((3, 3), dtype=np.uint8)
        for operation, builtin in [("erode", cv.erode), ("dilate", cv.dilate)]:
            expected = builtin(binary.astype(np.uint8), square, borderType=cv.BORDER_CONSTANT, borderValue=0).astype(bool)
            np.testing.assert_array_equal(binary_morphology(binary, square, operation), expected)

if __name__ == "__main__":
    unittest.main()
