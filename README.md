# Hypsometric Map to Heightmap Converter

Converts color-coded elevation maps (hypsometric coloring) to grayscale heightmaps suitable for 3D printing, game engines, or terrain visualization.

## Requirements

```bash
pip install numpy pillow scipy matplotlib
```

## Usage

```bash
# Basic usage
python heightmap_extract.py map.png

# Specify output directory
python heightmap_extract.py map.png --output-dir ./output

# Adjust smoothing (higher = smoother terrain)
python heightmap_extract.py map.png --smoothing 5.0

# Crop margins (top, bottom, left, right in pixels)
python heightmap_extract.py map.png --margins 60,50,0,400
```

## Output Files

- `*_heightmap_16bit.png` - 16-bit grayscale heightmap (use for Blender/3D tools)
- `*_heightmap_8bit.png` - 8-bit grayscale heightmap (for preview)
- `*_visualization.png` - Debug visualization showing extraction process

## Using in Blender

1. Create a plane mesh, subdivide it (e.g., 500x500)
2. Add a **Displace** modifier
3. Create new texture → Image → Load `*_heightmap_16bit.png`
4. Set texture color space to **Non-Color**
5. Adjust **Strength** in modifier for desired elevation scale
6. Apply modifier and export as STL

## Customizing the Color LUT

The script uses a color lookup table (LUT) mapping RGB colors to elevations. For different map styles, you'll need to sample colors from your specific map:

```python
# Sample colors at known elevation points
from PIL import Image
import numpy as np

img = np.array(Image.open('your_map.png'))
print(f"Color at (y=400, x=300): {img[400, 300, :]}")  # Sample a point
```

Then update `ELEVATION_COLORS` in the script with your sampled colors.

## Limitations

- Text labels and borders are masked and interpolated (may cause artifacts)
- Works best with high-resolution source maps
- Color matching is approximate; results depend on map color scheme

## License

MIT
