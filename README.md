# Digital Image Processing

NumPy implementations and OpenCV comparisons for learning how image-processing operations work. The six notebooks move from pixels and sampling to intensity transformations, filtering, morphology, and color.

The examples use small, generated images included in the repository. No textbook image archive, GPU, or machine-specific path is required.

## Learning path

| Order | Notebook | Topics |
| --- | --- | --- |
| 1 | [Image fundamentals](notebooks/01_image_fundamentals.ipynb) | Arrays, channels, cropping, replication, sampling, quantization |
| 2 | [Sampling and interpolation](notebooks/02_sampling_and_interpolation.ipynb) | Nearest neighbor, bilinear, bicubic |
| 3 | [Intensity and histograms](notebooks/03_intensity_and_histograms.ipynb) | Negative, logarithm, gamma, stretching, slicing, bit planes, equalization |
| 4 | [Spatial filtering](notebooks/04_spatial_filtering.ipynb) | Correlation, smoothing, median, derivatives, Laplacian |
| 5 | [Morphology](notebooks/05_morphology.ipynb) | Erosion, dilation, structuring elements, area and centroid |
| 6 | [Color processing](notebooks/06_color_processing.ipynb) | RGB/BGR, channels, mixing, pseudocolor, scaling |

## Quick start

Use Python 3.11 or 3.12. Clone the repository and open a terminal in its root:

```bash
git clone https://github.com/Muhammad-Huzifa/Digital-Image-Proccesing-from-Scratch-and-using-Built-in-Functions.git
cd Digital-Image-Proccesing-from-Scratch-and-using-Built-in-Functions
python -m venv .venv
```

Activate the environment using the command for your terminal:

| Terminal | Command |
| --- | --- |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |
| Windows Git Bash | `source .venv/Scripts/activate` |
| Linux/macOS | `source .venv/bin/activate` |

```bash
python -m pip install -r requirements.txt
jupyter lab
```

Open the first notebook and run its cells in order. Every notebook has its own setup cell and can also be run independently. OpenCV comparisons run when OpenCV is installed; the NumPy examples remain usable without it. Figures display inline, including in hosted notebook environments.

## Project structure

| Path | Purpose |
| --- | --- |
| `notebooks/` | Six lessons in learning order |
| `src/image_processing/` | Shared image loading and small NumPy algorithms |
| `data/sample_images/` | Reproducible grayscale, color, and binary PNGs |
| `docs/` | Development notes and original notebook mapping |
| `tests/` | Array-level correctness checks |

## Check the implementations

```bash
python -m unittest discover -s tests -v
```

The tests cover signed filtering, nearest-neighbor indexing, equalization, and binary morphology. They also compare with OpenCV when it is installed.

See [development notes](docs/DEVELOPMENT.md) for numerical conventions and [the source map](docs/SOURCE_MAP.md) for the original chapter notebooks.

## Author

Muhammad Huzifa — [GitHub](https://github.com/Muhammad-Huzifa)
