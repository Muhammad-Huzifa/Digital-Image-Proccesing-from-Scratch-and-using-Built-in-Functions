# Sample images

These 128×128 PNGs are generated from intensity ramps and geometric shapes for predictable, fast demonstrations:

- `grayscale.png`: intensity ramp with a bright square and dark circle.
- `color.png`: three shifted copies of the grayscale ramp in RGB channels.
- `binary.png`: foreground square and circle with a small hole.

They are teaching fixtures rather than photographs or benchmark data. To use a real image, replace a notebook's `load_sample()` call with your image-loading code and check its dimensions, dtype, and channel order.
