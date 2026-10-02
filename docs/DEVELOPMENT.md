# Development

Run notebooks from a fresh kernel. Shared functions find sample files relative to the package location, so notebook working directories do not affect loading.

Numerical conventions:

- Intensity examples use uint8 images and convert to floating point before arithmetic.
- Filtering preserves signed responses and uses zero padding. `correlate` does not reverse kernels.
- Binary morphology uses odd, symmetric elements and zero-valued pixels beyond the image border.
- Nearest neighbor uses floor-based source indices; OpenCV uses width, height ordering for resize dimensions.
- RGB images are converted from OpenCV's BGR order before display.
- Histogram probabilities divide counts by the total number of pixels.
- Histogram equalization handles constant images without division by zero. Library rounding can differ by one intensity level.

Run checks with `python -m unittest discover -s tests -v`. OpenCV comparisons are skipped with an explicit reason when the package is unavailable. A successful NumPy check does not imply an OpenCV installation has been validated.

References: [OpenCV filtering](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html), [JupyterLab installation](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html).
