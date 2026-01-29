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

# Tune mask detection (less aggressive)
python heightmap_extract.py map.png --elev-artifact-threshold 1000 --dilate-iterations 1
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir`, `-o` | `output` | Output directory |
| `--smoothing`, `-s` | `3.0` | Gaussian smoothing sigma (0 to disable) |
| `--margins`, `-m` | `0,0,0,0` | Crop margins: top,bottom,left,right |
| `--colormap`, `-c` | `cmap.json` | JSON colormap file (uses built-in if not found) |
| `--elev-artifact-threshold` | `300` | Elevation std threshold for artifact detection |
| `--dilate-iterations` | `3` | Mask dilation iterations (0 to disable) |
| `--save-river-mask` | flag | Save detected rivers as separate image |

## Hybrid Mask Detection Pipeline

The script uses a multi-stage approach to detect and mask non-terrain elements:

1. **Color → Elevation**: Map pixel colors to elevation values using colormap
2. **RGB-space detection**: Detect artifacts by color heuristics
   - Red borders: `R > 150 AND R > G*1.8 AND R > B*1.8`
   - Near-white: `R > 245 AND G > 245 AND B > 245`
   - Near-black: `R < 50 AND G < 50 AND B < 50`
   - Gray text: `max(RGB) - min(RGB) < 25 AND R < 180`
3. **River detection**: Blue-dominant pixels in land areas (saved separately)
4. **Elevation-space detection**: High local standard deviation indicates text/border artifacts
5. **Morphological cleanup**: Dilation to catch anti-aliasing fringes
6. **Interpolation**: Fill masked regions from surrounding terrain
7. **Smoothing**: Gaussian filter for final output

### Tuning Parameters

- **`--elev-artifact-threshold`**: Higher = less masking. Start with 300, increase if too much terrain is masked.
- **`--dilate-iterations`**: Controls AA fringe cleanup. Use 0-3 depending on map quality.

Typical results should have 10-30% masked pixels. If masking exceeds 40%, increase the threshold.

## Output Files

Pipeline outputs are numbered by processing step:

| File | Description |
|------|-------------|
| `*_01_cropped.png` | Input image after margin cropping |
| `*_02_mask_rgb.png` | RGB-space detected mask (red, white, black, gray) |
| `*_03_mask_elevation.png` | Elevation artifact mask (local std threshold) |
| `*_04_mask_combined.png` | Combined + dilated mask |
| `*_05_rivers_detected.png` | River detection overlay (cyan) |
| `*_06_heightmap_raw.png` | Color→elevation mapping (holes where masked) |
| `*_07_heightmap_filled.png` | After interpolating masked regions |
| `*_08_heightmap_16bit.png` | Final smoothed heightmap (16-bit) |
| `*_08_heightmap_8bit.png` | Final smoothed heightmap (8-bit preview) |
| `*_09_visualization.png` | Debug visualization with 3D preview |

## Custom Colormap

Create a JSON file (default: `cmap.json`) to define custom elevation colors. The elevation range is auto-detected from the colormap.

```json
{
  "elevation_colors": [
    {"rgb": [177, 126, 81], "elevation": 5500, "description": "high peaks"},
    {"rgb": [200, 211, 153], "elevation": 1000, "description": "hills"},
    {"rgb": [142, 186, 113], "elevation": 200, "description": "lowlands"},
    {"rgb": [238, 245, 255], "elevation": 0, "description": "sea level"},
    {"rgb": [115, 172, 221], "elevation": -5000, "description": "deep ocean"}
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
3. Create new texture → Image → Load `*_08_heightmap_16bit.png`
4. Set texture color space to **Non-Color**
5. Adjust **Strength** in modifier for desired elevation scale
6. Apply modifier and export as STL

## Limitations

- Text labels and borders are masked and interpolated (may cause artifacts in dense label areas)
- Works best with high-resolution source maps
- Color matching is approximate; results depend on map color scheme
- Rivers are detected but not specially handled (future feature)

## License

MIT
