# Hypsometric Map to Heightmap Converter

Converts color-coded elevation maps (hypsometric coloring) to grayscale heightmaps suitable for 3D printing, game engines, or terrain visualization.

## Requirements

```bash
pip install numpy pillow scipy matplotlib
```

## Usage

```bash
# Basic usage (outputs to ./output/, uses cmap.json if present)
python heightmap_extract.py map.png

# Adjust smoothing (higher = smoother terrain)
python heightmap_extract.py map.png --smoothing 5.0

# Crop margins (top, bottom, left, right in pixels)
python heightmap_extract.py map.png --margins 60,50,0,400

# Use custom colormap
python heightmap_extract.py map.png --colormap my_colors.json
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir`, `-o` | `output` | Output directory |
| `--smoothing`, `-s` | `3.0` | Gaussian smoothing sigma (0 to disable) |
| `--margins`, `-m` | `0,0,0,0` | Crop margins: top,bottom,left,right |
| `--colormap`, `-c` | `cmap.json` | JSON colormap file (uses built-in if not found) |

## Output Files

Pipeline outputs are numbered by processing step:

| File | Description |
|------|-------------|
| `*_01_cropped.png` | Input image after margin cropping |
| `*_02_mask.png` | Binary mask (white = non-terrain pixels) |
| `*_03_masked_overlay.png` | Original with masked areas in magenta |
| `*_04_heightmap_raw.png` | Color→elevation mapping (holes where masked) |
| `*_05_heightmap_filled.png` | After interpolating masked regions |
| `*_06_heightmap_16bit.png` | Final smoothed heightmap (16-bit) |
| `*_06_heightmap_8bit.png` | Final smoothed heightmap (8-bit preview) |
| `*_07_visualization.png` | Debug visualization with 3D preview |

## Custom Colormap

Create a JSON file (default: `cmap.json`) to define custom elevation colors:

```json
{
  "elevation_colors": [
    {"rgb": [255, 251, 250], "elevation": 4500, "description": "snow peaks"},
    {"rgb": [200, 211, 153], "elevation": 1000, "description": "hills"},
    {"rgb": [151, 190, 125], "elevation": 100, "description": "lowlands"},
    {"rgb": [206, 242, 254], "elevation": 0, "description": "sea level"},
    {"rgb": [135, 177, 191], "elevation": -2000, "description": "ocean"}
  ],
  "mask_colors": [
    {"rgb": [255, 255, 255], "description": "white background"},
    {"rgb": [0, 0, 0], "description": "black text"},
    {"rgb": [200, 50, 50], "description": "red border"}
  ]
}
```

To sample colors from your map:

```python
from PIL import Image
import numpy as np

img = np.array(Image.open('your_map.png'))
print(f"Color at (y=400, x=300): {img[400, 300, :3]}")  # Sample a point
```

## Using in Blender

1. Create a plane mesh, subdivide it (e.g., 500x500)
2. Add a **Displace** modifier
3. Create new texture → Image → Load `*_06_heightmap_16bit.png`
4. Set texture color space to **Non-Color**
5. Adjust **Strength** in modifier for desired elevation scale
6. Apply modifier and export as STL

## Limitations

- Text labels and borders are masked and interpolated (may cause artifacts)
- Works best with high-resolution source maps
- Color matching is approximate; results depend on map color scheme

## License

MIT
