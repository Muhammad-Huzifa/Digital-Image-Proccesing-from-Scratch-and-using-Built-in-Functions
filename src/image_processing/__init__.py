"""Small NumPy implementations shared by the image-processing lessons."""
from pathlib import Path
import numpy as np
from PIL import Image

DATA = Path(__file__).resolve().parents[2] / "data/sample_images"

def load_sample(name="grayscale"):
    """Load a bundled PNG without depending on the notebook working directory."""
    with Image.open(DATA / f"{name}.png") as image:
        return np.array(image)

def show_images(images, titles, gray=True):
    """Display images inline; no desktop window or keyboard input is required."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4), squeeze=False)
    for ax, image, title in zip(axes[0], images, titles):
        ax.imshow(image, cmap="gray" if gray else None)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def nearest_neighbor(image, height, width):
    """Resize with floor-based pixel selection, matching OpenCV INTER_NEAREST."""
    if height < 1 or width < 1:
        raise ValueError("Output dimensions must be positive.")
    rows = np.minimum(np.arange(height) * image.shape[0] // height, image.shape[0] - 1)
    cols = np.minimum(np.arange(width) * image.shape[1] // width, image.shape[1] - 1)
    return image[rows[:, None], cols]

def correlate(image, kernel):
    """Apply an odd-sized kernel with zero padding and preserve signed results.

    This is correlation: the kernel is not reversed. For convolution, pass
    np.flip(kernel). OpenCV filter2D also computes correlation.
    """
    kernel = np.asarray(kernel, dtype=float)
    if image.ndim != 2 or kernel.ndim != 2 or any(s % 2 == 0 for s in kernel.shape):
        raise ValueError("Use a grayscale image and an odd-sized 2D kernel.")
    h, w = kernel.shape
    padded = np.pad(image.astype(float), ((h // 2, h // 2), (w // 2, w // 2)))
    output = np.zeros(image.shape, dtype=float)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            patch = padded[row:row + h, col:col + w]
            output[row, col] = np.sum(patch * kernel)
    return output

def equalize_histogram(image):
    """Equalize a uint8 grayscale image using its nonzero cumulative histogram."""
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Use a uint8 grayscale image.")
    histogram = np.bincount(image.ravel(), minlength=256)
    cdf = histogram.cumsum()
    cdf_min = cdf[histogram > 0][0]
    if cdf_min == image.size:
        return image.copy()
    table = np.clip(np.rint((cdf - cdf_min) * 255 / (image.size - cdf_min)), 0, 255).astype(np.uint8)
    return table[image]

def binary_morphology(image, kernel, operation):
    """Erode or dilate a binary image using an odd, symmetric structuring element.

    The lessons use square and cross elements. Pixels outside the image are zero.
    """
    kernel = np.asarray(kernel, dtype=bool)
    if image.ndim != 2 or kernel.ndim != 2 or any(s % 2 == 0 for s in kernel.shape) or not kernel.any():
        raise ValueError("Use a binary image and a nonempty, odd-sized 2D kernel.")
    if not np.array_equal(kernel, np.flip(kernel)):
        raise ValueError("Use a symmetric structuring element.")
    if operation not in {"erode", "dilate"}:
        raise ValueError("Operation must be erode or dilate.")
    h, w = kernel.shape
    padded = np.pad(image.astype(bool), ((h // 2, h // 2), (w // 2, w // 2)))
    output = np.zeros(image.shape, dtype=bool)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            values = padded[row:row + h, col:col + w][kernel]
            output[row, col] = values.all() if operation == "erode" else values.any()
    return output
