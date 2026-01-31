#!/usr/bin/env python3
"""
Hypsometric Map to Heightmap Converter

Converts color-coded elevation maps (hypsometric coloring) to grayscale heightmaps
suitable for 3D printing, game engines, or terrain visualization.

Pipeline:
1. Define color→elevation LUT based on hypsometric color scheme
2. For each pixel, find nearest color match in LUT
3. Mask non-terrain elements (text, borders, legends)
4. Interpolate masked regions
5. Apply smoothing to reduce artifacts
6. Output 16-bit grayscale PNG suitable for 3D displacement

Usage:
    python heightmap_extract.py input_map.png [--output-dir ./output]

The color LUT may need adjustment for different map styles. Sample colors from
your specific map and update the LUT accordingly.
"""

import argparse
import json
import os
import struct
import numpy as np
from PIL import Image
from scipy.spatial import KDTree
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


# Processing parameters
SMOOTHING_SIGMA = 3.0     # Gaussian smoothing sigma
ELEVATION_MIN = -5000     # Default minimum elevation for normalization
ELEVATION_MAX = 6000      # Default maximum elevation for normalization


def load_colormap(path):
    """
    Load elevation colors from JSON file.

    Expected JSON format:
    {
      "elevation_colors": [
        {"rgb": [255, 251, 250], "elevation": 4500},
        {"rgb": [151, 190, 125], "elevation": 100},
        ...
      ],
      "mask_colors": [
        {"rgb": [255, 255, 255], "description": "white background"},
        {"rgb": [0, 0, 0], "description": "black text"},
        ...
      ]
    }

    Returns:
        elevation_colors: List of (rgb, elevation) tuples
        mask_colors: List of rgb values
        water_colors: Dict with 'river' and 'lake' entries (rgb, tolerance)
        elev_min: Minimum elevation in colormap
        elev_max: Maximum elevation in colormap
    """
    with open(path) as f:
        data = json.load(f)

    elevation_colors = [(item['rgb'], item['elevation']) for item in data['elevation_colors']]

    # Parse mask colors with tolerance
    mask_colors = []
    for item in data.get('mask_colors', []):
        mask_colors.append({
            'rgb': np.array(item['rgb']),
            'tolerance': item.get('tolerance', 30),
            'description': item.get('description', '')
        })

    # Parse water colors (river, lake)
    water_colors = {}
    if 'water_colors' in data:
        for key, val in data['water_colors'].items():
            water_colors[key] = {
                'rgb': np.array(val['rgb']),
                'tolerance': val.get('tolerance', 30)
            }

    elevations = [item['elevation'] for item in data['elevation_colors']]
    elev_min = min(elevations)
    elev_max = max(elevations)

    return elevation_colors, mask_colors, water_colors, elev_min, elev_max


def load_image(path):
    """Load image and convert to numpy array."""
    img = Image.open(path)
    return np.array(img)



def interpolate_masked_regions(heightmap):
    """Fill masked (NaN) regions using nearest-neighbor interpolation."""
    valid_mask = ~np.isnan(heightmap)
    indices = ndimage.distance_transform_edt(
        ~valid_mask, return_distances=False, return_indices=True
    )
    return heightmap[tuple(indices)]


# ============================================================================
# HYBRID MASK DETECTION FUNCTIONS
# ============================================================================

def detect_mask_colors(image, mask_colors):
    """
    Detect non-terrain pixels by matching colors from cmap.json.

    Uses Euclidean distance in RGB space with per-color tolerance.

    Args:
        image: RGB or RGBA image as numpy array
        mask_colors: List of dicts with 'rgb' (array) and 'tolerance' (float)

    Returns:
        Boolean mask (True = mask this pixel)
    """
    # Handle RGBA
    if image.shape[2] == 4:
        image = image[:, :, :3]

    rgb = image.astype(np.float32)
    mask = np.zeros(image.shape[:2], dtype=bool)

    if not mask_colors:
        return mask

    for entry in mask_colors:
        target_rgb = entry['rgb'].astype(np.float32)
        tolerance = entry['tolerance']

        # Euclidean distance in RGB space
        dist = np.sqrt(np.sum((rgb - target_rgb) ** 2, axis=2))
        color_match = dist < tolerance
        mask |= color_match

        count = np.sum(color_match)
        if count > 0:
            desc = entry.get('description', 'unnamed')
            print(f"    {desc}: {count:,} pixels")

    return mask


def detect_rivers(image, elevation, water_colors):
    """
    Detect river and lake pixels by color matching.

    Uses specific colors from colormap with tolerance for anti-aliasing.
    Checks if pixel is on land by looking at neighboring pixels' elevation.

    Args:
        image: RGB or RGBA image as numpy array
        elevation: 2D array of elevation values
        water_colors: Dict with 'river' and/or 'lake' entries containing rgb and tolerance

    Returns:
        Boolean mask for water pixels (rivers and lakes on land)
    """
    # Handle RGBA
    if image.shape[2] == 4:
        image = image[:, :, :3]

    rgb = image.astype(np.float32)
    mask = np.zeros(image.shape[:2], dtype=bool)

    if not water_colors:
        return mask

    # Check if surrounded by land using max filter on elevation
    # A pixel is "on land" if nearby pixels have positive elevation
    land_nearby = ndimage.maximum_filter(elevation, size=7) > 50

    for water_type, info in water_colors.items():
        target_rgb = info['rgb'].astype(np.float32)
        tolerance = info['tolerance']

        # Euclidean distance in RGB space
        dist = np.sqrt(np.sum((rgb - target_rgb) ** 2, axis=2))
        color_match = dist < tolerance

        # Rivers/lakes must be surrounded by land
        water_mask = color_match & land_nearby
        mask |= water_mask

        count = np.sum(water_mask)
        if count > 0:
            print(f"  {water_type}: {count:,} pixels")

    return mask


def detect_color_artifacts(color_distances, threshold=30.0):
    """
    Detect artifacts by color distance to palette.

    Pixels with colors far from any hypsometric palette entry are likely
    text, borders, or other non-terrain elements.

    Args:
        color_distances: 2D array of Euclidean RGB distance to nearest palette color
        threshold: Distance threshold (pixels above this are artifacts)

    Returns:
        Boolean mask (True = artifact pixel)
    """
    return color_distances > threshold


def combine_and_cleanup_mask(rgb_mask, elev_mask, dilate_iterations=3):
    """
    Combine masks and apply morphological cleanup.

    Args:
        rgb_mask: Boolean mask from RGB detection
        elev_mask: Boolean mask from elevation artifact detection
        dilate_iterations: Number of dilation iterations to catch AA fringes

    Returns:
        Combined and cleaned boolean mask
    """
    from scipy.ndimage import binary_dilation

    combined = rgb_mask | elev_mask
    if dilate_iterations > 0:
        dilated = binary_dilation(combined, iterations=dilate_iterations)
        return dilated
    return combined


def colors_to_elevation(map_region, color_tree, lut_elevations):
    """
    Convert map colors to elevation values.

    Args:
        map_region: RGB or RGBA image as numpy array
        color_tree: KDTree of elevation colors
        lut_elevations: Array of elevation values corresponding to colors

    Returns:
        heightmap: 2D array of elevation values
        color_distances: 2D array of Euclidean distance to nearest palette color
    """
    h, w, c = map_region.shape
    # Handle RGBA images by taking only RGB channels
    if c == 4:
        map_region = map_region[:, :, :3]
    pixels = map_region.reshape(-1, 3)

    # Find nearest elevation color for each pixel
    distances, indices = color_tree.query(pixels)

    # Map to elevations
    pixel_elevations = lut_elevations[indices].astype(float)
    heightmap = pixel_elevations.reshape(h, w)
    color_distances = distances.reshape(h, w)

    return heightmap, color_distances


def save_heightmap(heightmap, output_path, bits=16, elev_min=ELEVATION_MIN, elev_max=ELEVATION_MAX):
    """
    Save heightmap as PNG with specified bit depth.
    
    Args:
        heightmap: 2D array of elevation values
        output_path: Output file path
        bits: Bit depth (8 or 16)
        elev_min: Minimum elevation for normalization
        elev_max: Maximum elevation for normalization
    """
    # Normalize to 0-1 range
    normalized = (heightmap - elev_min) / (elev_max - elev_min)
    normalized = np.clip(normalized, 0, 1)
    
    if bits == 16:
        data = (normalized * 65535).astype(np.uint16)
        img = Image.fromarray(data)
    else:
        data = (normalized * 255).astype(np.uint8)
        img = Image.fromarray(data, mode='L')
    
    img.save(output_path)


# ============================================================================
# MESH GENERATION FUNCTIONS
# ============================================================================

def create_mesh(heightmap, scale=(1.0, 1.0, 1.0), decimate=1):
    """
    Create 3D mesh vertices and faces from heightmap.

    Args:
        heightmap: 2D numpy array of height values (0-1 normalized)
        scale: (x, y, z) scale factors
        decimate: Decimation factor (1 = full res, 2 = half, etc.)

    Returns:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of triangle indices
    """
    # Flip Y-axis: image Y=0 is top, but 3D Y=0 is bottom
    heightmap = np.flipud(heightmap)

    # Decimate if requested
    if decimate > 1:
        heightmap = heightmap[::decimate, ::decimate]

    h, w = heightmap.shape
    scale_x, scale_y, scale_z = scale

    # Create vertex grid
    x = np.linspace(-0.5, 0.5, w) * scale_x
    y = np.linspace(-0.5, 0.5, h) * scale_y
    xx, yy = np.meshgrid(x, y)

    # Z is height value scaled
    zz = heightmap * scale_z

    # Flatten to vertex list
    vertices = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    # Create triangle faces
    faces = []
    for row in range(h - 1):
        for col in range(w - 1):
            v0 = row * w + col
            v1 = row * w + col + 1
            v2 = (row + 1) * w + col
            v3 = (row + 1) * w + col + 1
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    return vertices, np.array(faces, dtype=np.int32)


def save_obj(path, vertices, faces):
    """Save mesh in Wavefront OBJ format."""
    with open(path, 'w') as f:
        f.write("# Heightmap mesh\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def save_stl_binary(path, vertices, faces):
    """Save mesh in binary STL format."""
    with open(path, 'wb') as f:
        f.write(b'\0' * 80)  # Header
        f.write(struct.pack('<I', len(faces)))
        for face in faces:
            v0, v1, v2 = vertices[face]
            edge1, edge2 = v1 - v0, v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal /= norm
            f.write(struct.pack('<3f', *normal))
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<H', 0))


def create_visualization(map_region, heightmap, mask, output_path,
                         elev_min=ELEVATION_MIN, elev_max=ELEVATION_MAX):
    """Create visualization of the extraction process."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].imshow(map_region)
    axes[0, 0].set_title("Original map region")
    axes[0, 0].axis('off')

    im1 = axes[0, 1].imshow(heightmap, cmap='cividis', vmin=elev_min, vmax=elev_max)
    axes[0, 1].set_title("Extracted heightmap")
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], label='Elevation (m)', shrink=0.8)

    axes[1, 0].imshow(mask, cmap='gray')
    axes[1, 0].set_title(f"Masked regions: {100*mask.mean():.1f}%")
    axes[1, 0].axis('off')

    # 3D preview
    ax3d = fig.add_subplot(2, 2, 4, projection='3d')
    h, w = heightmap.shape
    step = max(1, min(h, w) // 100)
    Y_grid, X_grid = np.mgrid[0:h:step, 0:w:step]
    Z = heightmap[::step, ::step]
    ax3d.plot_surface(X_grid, Y_grid, Z, cmap='terrain', linewidth=0, antialiased=True)
    ax3d.set_title("3D Preview")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def process_map(input_path, output_dir='.', smoothing=SMOOTHING_SIGMA,
                elevation_colors=None, mask_colors=None,
                elev_min=None, elev_max=None,
                water_colors=None,
                color_artifact_threshold=30.0, dilate_iterations=3,
                mesh_scale=(100.0, 100.0, 20.0), mesh_decimate=4):
    """
    Main processing pipeline with hybrid mask detection.

    Pipeline:
    1. Load image
    2. Color → Elevation mapping
    3. Mask color detection (from cmap.json mask_colors)
    4. River detection (separate layer, not masked)
    5. Color-space artifact detection (distance to palette)
    6. Combine masks + morphological cleanup (dilation)
    7. Interpolate masked regions
    8. Smooth, save, and generate 3D meshes

    Args:
        input_path: Path to input map image
        output_dir: Directory for output files
        smoothing: Gaussian smoothing sigma (0 to disable)
        elevation_colors: List of (rgb, elevation) tuples (required)
        mask_colors: List of colors to mask (from cmap.json)
        elev_min: Minimum elevation for normalization
        elev_max: Maximum elevation for normalization
        color_artifact_threshold: RGB distance threshold for artifact detection
        dilate_iterations: Mask dilation iterations for cleanup
        mesh_scale: (x, y, z) scale for mesh generation
        mesh_decimate: Decimation factor for mesh (1=full, 4=quarter res)

    Returns:
        Dict with paths to output files
    """
    if elev_min is None:
        elev_min = ELEVATION_MIN
    if elev_max is None:
        elev_max = ELEVATION_MAX

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # Step 1: Load image
    print(f"Loading image: {input_path}")
    map_region = load_image(input_path)
    print(f"  Shape: {map_region.shape}, dtype: {map_region.dtype}")

    # Step 2: Color → Elevation (without masking)
    print("Building color lookup tables...")
    lut_colors = np.array([c[0] for c in elevation_colors])
    lut_elevations = np.array([c[1] for c in elevation_colors])
    color_tree = KDTree(lut_colors)
    print(f"  {len(elevation_colors)} elevation colors")

    print("Converting colors to elevation...")
    heightmap_raw, color_distances = colors_to_elevation(map_region, color_tree, lut_elevations)
    print(f"  Raw elevation range: {heightmap_raw.min():.0f}m to {heightmap_raw.max():.0f}m")
    print(f"  Color distance range: {color_distances.min():.1f} to {color_distances.max():.1f}")

    # Step 3: Mask color detection (from cmap.json)
    print("Detecting mask colors...")
    mask_explicit = detect_mask_colors(map_region, mask_colors or [])
    print(f"  Mask colors: {mask_explicit.sum()} pixels ({100*mask_explicit.mean():.1f}%)")

    # Step 4: River detection (separate layer)
    print("Detecting rivers/lakes...")
    rivers = detect_rivers(map_region, heightmap_raw, water_colors or {})
    print(f"  Total water detected: {rivers.sum()} pixels ({100*rivers.mean():.1f}%)")

    # Step 5: Color-space artifact detection (pixels far from palette)
    print(f"Detecting color artifacts (threshold={color_artifact_threshold})...")
    mask_color = detect_color_artifacts(color_distances, threshold=color_artifact_threshold)
    print(f"  Color artifact mask: {mask_color.sum()} pixels ({100*mask_color.mean():.1f}%)")

    # Step 6: Combine masks + morphological cleanup
    print(f"Combining masks and cleanup (dilate={dilate_iterations})...")
    mask_combined = combine_and_cleanup_mask(mask_explicit, mask_color, dilate_iterations)
    print(f"  Combined mask: {mask_combined.sum()} pixels ({100*mask_combined.mean():.1f}%)")

    # Apply mask to heightmap
    heightmap = heightmap_raw.copy()
    heightmap[mask_combined] = np.nan

    # Step 7: Interpolate masked regions
    print("Interpolating masked regions...")
    heightmap_filled = interpolate_masked_regions(heightmap)

    # Step 8: Smooth
    if smoothing > 0:
        print(f"Applying Gaussian smoothing (sigma={smoothing})...")
        heightmap_final = gaussian_filter(heightmap_filled, sigma=smoothing)
    else:
        heightmap_final = heightmap_filled

    print(f"  Final range: {heightmap_final.min():.0f}m to {heightmap_final.max():.0f}m")

    # Save outputs
    outputs = {}

    # 01: input image
    path_input = os.path.join(output_dir, f"{base_name}_01_input.png")
    Image.fromarray(map_region).save(path_input)
    outputs['input'] = path_input
    print(f"  Saved: {path_input}")

    # 02: RGB-space mask
    path_mask_explicit = os.path.join(output_dir, f"{base_name}_02_mask_explicit.png")
    Image.fromarray((mask_explicit * 255).astype(np.uint8), mode='L').save(path_mask_explicit)
    outputs['mask_explicit'] = path_mask_explicit
    print(f"  Saved: {path_mask_explicit}")

    # 03: Color artifact mask
    path_mask_color = os.path.join(output_dir, f"{base_name}_03_mask_color.png")
    Image.fromarray((mask_color * 255).astype(np.uint8), mode='L').save(path_mask_color)
    outputs['mask_color'] = path_mask_color
    print(f"  Saved: {path_mask_color}")

    # 04: Combined + dilated mask
    path_mask_combined = os.path.join(output_dir, f"{base_name}_04_mask_combined.png")
    Image.fromarray((mask_combined * 255).astype(np.uint8), mode='L').save(path_mask_combined)
    outputs['mask_combined'] = path_mask_combined
    print(f"  Saved: {path_mask_combined}")

    # 05: Rivers detected (grayscale mask)
    path_rivers = os.path.join(output_dir, f"{base_name}_05_rivers_detected.png")
    Image.fromarray((rivers * 255).astype(np.uint8), mode='L').save(path_rivers)
    outputs['rivers'] = path_rivers
    print(f"  Saved: {path_rivers}")

    # 06: heightmap before interpolation (with holes shown as black)
    path_raw = os.path.join(output_dir, f"{base_name}_06_heightmap_raw.png")
    save_heightmap(np.nan_to_num(heightmap, nan=elev_min), path_raw, bits=8,
                   elev_min=elev_min, elev_max=elev_max)
    outputs['heightmap_raw'] = path_raw
    print(f"  Saved: {path_raw}")

    # 07: heightmap after interpolation, before smoothing
    path_filled = os.path.join(output_dir, f"{base_name}_07_heightmap_filled.png")
    save_heightmap(heightmap_filled, path_filled, bits=8,
                   elev_min=elev_min, elev_max=elev_max)
    outputs['heightmap_filled'] = path_filled
    print(f"  Saved: {path_filled}")

    # 08: final heightmap (16-bit and 8-bit)
    path_16bit = os.path.join(output_dir, f"{base_name}_08_heightmap_16bit.png")
    save_heightmap(heightmap_final, path_16bit, bits=16,
                   elev_min=elev_min, elev_max=elev_max)
    outputs['heightmap_16bit'] = path_16bit
    print(f"  Saved: {path_16bit}")

    path_8bit = os.path.join(output_dir, f"{base_name}_08_heightmap_8bit.png")
    save_heightmap(heightmap_final, path_8bit, bits=8,
                   elev_min=elev_min, elev_max=elev_max)
    outputs['heightmap_8bit'] = path_8bit
    print(f"  Saved: {path_8bit}")

    # 09: visualization
    path_viz = os.path.join(output_dir, f"{base_name}_09_visualization.png")
    create_visualization(map_region, heightmap_final, mask_combined, path_viz,
                         elev_min=elev_min, elev_max=elev_max)
    outputs['visualization'] = path_viz
    print(f"  Saved: {path_viz}")

    # 10: 3D mesh generation
    print(f"Generating 3D mesh (scale={mesh_scale}, decimate={mesh_decimate})...")

    # Normalize heightmap to 0-1 for mesh generation
    hm_normalized = (heightmap_final - elev_min) / (elev_max - elev_min)
    hm_normalized = np.clip(hm_normalized, 0, 1)

    vertices, faces = create_mesh(hm_normalized, scale=mesh_scale, decimate=mesh_decimate)
    print(f"  Vertices: {len(vertices):,}, Triangles: {len(faces):,}")

    # Save OBJ
    path_obj = os.path.join(output_dir, f"{base_name}_10_terrain.obj")
    save_obj(path_obj, vertices, faces)
    outputs['mesh_obj'] = path_obj
    size_mb = os.path.getsize(path_obj) / (1024 * 1024)
    print(f"  Saved: {path_obj} ({size_mb:.1f} MB)")

    # Save STL
    path_stl = os.path.join(output_dir, f"{base_name}_10_terrain.stl")
    save_stl_binary(path_stl, vertices, faces)
    outputs['mesh_stl'] = path_stl
    size_mb = os.path.getsize(path_stl) / (1024 * 1024)
    print(f"  Saved: {path_stl} ({size_mb:.1f} MB)")

    print("Done!")
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description='Convert hypsometric map to heightmap for 3D printing/visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s map.png
    %(prog)s map.png --output-dir ./output
    %(prog)s map.png --colormap my_colormap.json
    %(prog)s map.png --color-artifact-threshold 30 --dilate-iterations 3

Hybrid Mask Detection Pipeline:
    1. Explicit mask colors: defined in cmap.json mask_colors with tolerance
    2. Color-space detection: pixels far from palette (by RGB distance)
    3. Morphological cleanup: dilation to catch anti-aliasing fringes
    4. River detection: saved separately for future handling

Output files (numbered by pipeline step):
    01_input.png           - Input image
    02_mask_explicit.png   - Mask colors from cmap.json
    03_mask_color.png      - Color distance artifact mask
    04_mask_combined.png   - Combined + dilated mask
    05_rivers_detected.png - River detection (grayscale)
    06_heightmap_raw.png   - Before interpolation (with holes)
    07_heightmap_filled.png - After interpolation
    08_heightmap_*.png     - Final smoothed (8-bit and 16-bit)
    09_visualization.png   - Debug visualization
    10_terrain.obj/.stl    - 3D mesh files

Notes:
    - Requires cmap.json colormap file (or specify with --colormap)
    - Tune --color-artifact-threshold based on your map (higher = less masking)
    - Tune --dilate-iterations for anti-aliasing cleanup (0 to disable)
        """
    )
    parser.add_argument('input', help='Input map image (PNG, JPG, etc.)')
    parser.add_argument('--output-dir', '-o', default='output',
                        help='Output directory (default: output)')
    parser.add_argument('--smoothing', '-s', type=float, default=SMOOTHING_SIGMA,
                        help=f'Gaussian smoothing sigma (default: {SMOOTHING_SIGMA}, 0 to disable)')
    parser.add_argument('--colormap', '-c', type=str, default='cmap.json',
                        help='JSON file with elevation colors (default: cmap.json)')
    parser.add_argument('--color-artifact-threshold', type=float, default=30.0,
                        help='RGB distance threshold for artifact detection (default: 30)')
    parser.add_argument('--dilate-iterations', type=int, default=3,
                        help='Mask dilation iterations for cleanup (default: 3, 0 to disable)')
    parser.add_argument('--mesh-scale', type=str, default='100,100,20',
                        help='Mesh scale as X,Y,Z (default: 100,100,20)')
    parser.add_argument('--mesh-decimate', type=int, default=4,
                        help='Mesh decimation factor (default: 4, 1=full res)')

    args = parser.parse_args()

    # Parse mesh scale
    mesh_scale = tuple(float(x) for x in args.mesh_scale.split(','))
    if len(mesh_scale) != 3:
        print("Error: --mesh-scale must be X,Y,Z (e.g., 100,100,20)")
        return 1

    # Load colormap (required)
    if not os.path.exists(args.colormap):
        print(f"Error: colormap file '{args.colormap}' not found")
        print("Create a cmap.json file or specify one with --colormap")
        return 1

    print(f"Loading colormap: {args.colormap}")
    elevation_colors, mask_colors, water_colors, elev_min, elev_max = load_colormap(args.colormap)
    print(f"  Elevation range: {elev_min}m to {elev_max}m")
    if mask_colors:
        print(f"  Mask colors: {len(mask_colors)} entries")
    if water_colors:
        print(f"  Water colors: {', '.join(water_colors.keys())}")

    process_map(args.input, args.output_dir, args.smoothing,
                elevation_colors, mask_colors, elev_min, elev_max, water_colors,
                args.color_artifact_threshold, args.dilate_iterations,
                mesh_scale, args.mesh_decimate)


if __name__ == '__main__':
    main()
