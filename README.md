# Hypsometric Map Extraction and Mesh Generation

- written by @claude 🤖

Converts color-coded elevation maps (hypsometric coloring) to grayscale heightmaps and 3D meshes suitable for 3D printing, game engines, or terrain visualization.

## Requirements

```bash
pip install numpy pillow scipy matplotlib
```

## Usage

```bash
# Basic usage (outputs to ./output/, uses cmap.json)
python heightmap_extract.py map.png

# Adjust smoothing (higher = smoother terrain)
python heightmap_extract.py map.png --smoothing 5.0

# Use custom colormap
python heightmap_extract.py map.png --colormap my_colors.json

# Tune mask detection (less aggressive)
python heightmap_extract.py map.png --color-artifact-threshold 40 --dilate-iterations 1
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir`, `-o` | `output` | Output directory |
| `--smoothing`, `-s` | `5.0` | Gaussian smoothing sigma (0 to disable) |
| `--colormap`, `-c` | `cmap.json` | JSON colormap file (required) |
| `--color-artifact-threshold` | `30` | RGB distance threshold for artifact detection |
| `--dilate-iterations` | `3` | Mask dilation iterations (0 to disable) |
| `--mesh-scale` | `100,100,20` | Mesh scale as X,Y,Z |
| `--mesh-decimate` | `4` | Mesh decimation factor (1=full, 4=quarter res) |

## Mask Detection Pipeline

The script uses a multi-stage approach to detect and mask non-terrain elements:

1. **Color → Elevation**: Map pixel colors to elevation values using colormap
2. **Explicit mask colors**: Match colors defined in `mask_colors` with tolerance
   - Supports conditional masking via `next_to` for translucent overlays
3. **Color distance detection**: Pixels far from any palette color are artifacts
4. **Morphological cleanup**: Dilation to catch anti-aliasing fringes
5. **Interpolation**: Fill masked regions from surrounding terrain
6. **Smoothing**: Gaussian filter for final output
7. **Mesh generation**: Automatic OBJ and STL export

### Conditional Masking

For maps with translucent borders (e.g., red border over terrain), use `next_to` to only mask colors when adjacent to anchor colors:

```json
"mask_colors": [
  {"color": "#FF2628", "tolerance": 40, "id": "red_border"},
  {"color": "#C69A5F", "tolerance": 30, "id": "red_trans", "next_to": ["red_border"]}
]
```

This prevents brownish pixels on mountain peaks from being masked as "translucent red border".

## Output Files

Pipeline outputs are numbered by processing step:

| File | Description |
|------|-------------|
| `*_01_input.png` | Input image |
| `*_02_mask_explicit.png` | Mask colors from cmap.json |
| `*_03_mask_color.png` | Color distance artifact mask |
| `*_04_mask_combined.png` | Combined + dilated mask |
| `*_05_water.png` | Water layer (river + lake) |
| `*_06_heightmap_raw.png` | Color→elevation mapping (holes where masked) |
| `*_07_heightmap_filled.png` | After interpolating masked regions |
| `*_08_heightmap_16bit.png` | Final smoothed heightmap (16-bit) |
| `*_08_heightmap_8bit.png` | Final smoothed heightmap (8-bit preview) |
| `*_09_visualization.png` | Debug visualization with 3D preview |
| `*_10_terrain.obj` | 3D mesh (Wavefront OBJ) |
| `*_10_terrain.stl` | 3D mesh (binary STL) |

## Colormap Format

Create a JSON file (default: `cmap.json`) with hex colors for IDE preview support:

```json
{
  "elevation_colors": [
    {"color": "#B17E51", "elevation": 5500},
    {"color": "#8EBA71", "elevation": 200},
    {"color": "#EEF5FF", "elevation": 0},
    {"color": "#73ACDD", "elevation": -5000}
  ],
  "mask_colors": [
    {"color": "#FFFFFF", "tolerance": 15, "id": "white"},
    {"color": "#000000", "tolerance": 50, "id": "black"},
    {"color": "#FF2628", "tolerance": 40, "id": "red_border"},
    {"color": "#1993B9", "tolerance": 30, "id": "river"},
    {"color": "#EEF5FC", "tolerance": 15, "id": "lake"},
    {"color": "#C69A5F", "tolerance": 30, "id": "red_trans", "next_to": ["red_border", "river"]}
  ],
  "water_ids": ["river", "lake"]
}
```

### Fields

- **`elevation_colors`**: Color-to-elevation mapping (hypsometric palette)
- **`mask_colors`**: Colors to mask as non-terrain
  - `color`: Hex color (#RRGGBB) or `rgb` array [R,G,B]
  - `tolerance`: Euclidean distance in RGB space
  - `id`: Unique identifier for referencing
  - `next_to`: (optional) Only mask if adjacent to these anchor ids
- **`water_ids`**: Which mask color ids are water (for separate layer output)

### Sampling Colors

```python
from PIL import Image
import numpy as np

img = np.array(Image.open('your_map.png'))
r, g, b = img[400, 300, :3]
print(f"Color at (y=400, x=300): #{r:02X}{g:02X}{b:02X}")
```

## Using in Blender

### Method 1: Import Pre-built Mesh (Recommended)

1. Run the script to generate mesh files
2. In Blender: File → Import → Wavefront (.obj)
3. Select `*_10_terrain.obj`

### Method 2: Displace Modifier

1. Create a plane mesh, subdivide it (e.g., 500x500)
2. Add a **Displace** modifier
3. Create new texture → Image → Load `*_08_heightmap_16bit.png`
4. Set texture color space to **Non-Color**
5. Adjust **Strength** in modifier for desired elevation scale

## Limitations

- Text labels and borders are masked and interpolated (may cause artifacts in dense label areas)
- Works best with high-resolution source maps
- Color matching is approximate; results depend on map color scheme

## License

MIT
