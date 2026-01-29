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

## Heightmap to 3D Mesh

Convert heightmaps directly to 3D mesh files using `heightmap_to_mesh.py`:

```bash
# Convert to OBJ (recommended for Blender)
python heightmap_to_mesh.py output/map_08_heightmap_16bit.png -o terrain.obj

# With scale: width=100, depth=100, height exaggeration=20
python heightmap_to_mesh.py heightmap.png -o terrain.obj --scale 100,100,20

# Reduce resolution for large heightmaps (1/16 triangles)
python heightmap_to_mesh.py heightmap.png -o terrain.obj --decimate 4

# Export as STL for 3D printing
python heightmap_to_mesh.py heightmap.png -o terrain.stl --scale 100,100,10
```

### Mesh Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output`, `-o` | required | Output file (.obj, .stl, or .ply) |
| `--scale`, `-s` | `1,1,1` | Scale as X,Y,Z (width, depth, height) |
| `--decimate`, `-d` | `1` | Decimation factor (2=half res, 4=quarter) |
| `--no-normals` | flag | Skip normal calculation (faster) |
| `--ascii-stl` | flag | ASCII STL format (larger but readable) |

### Output Formats

- **OBJ**: Recommended for Blender, includes normals
- **STL**: Best for 3D printing (binary format by default)
- **PLY**: Alternative format with normal support

## Using in Blender

### Method 1: Import Pre-built Mesh (Recommended)

1. Generate mesh: `python heightmap_to_mesh.py heightmap.png -o terrain.obj --scale 100,100,20`
2. In Blender: File → Import → Wavefront (.obj)
3. Select the generated `.obj` file

### Method 2: Displace Modifier

1. Create a plane mesh, subdivide it (e.g., 500x500)
2. Add a **Displace** modifier
3. Create new texture → Image → Load `*_08_heightmap_16bit.png`
4. Set texture color space to **Non-Color**
5. Adjust **Strength** in modifier for desired elevation scale
6. Apply modifier and export as STL

## Limitations

- Text labels and borders are masked and interpolated (may cause artifacts)
- Works best with high-resolution source maps
- Color matching is approximate; results depend on map color scheme

## License

MIT
