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
        elev_min: Minimum elevation in colormap
        elev_max: Maximum elevation in colormap
    """
    with open(path) as f:
        data = json.load(f)

    elevation_colors = [(item['rgb'], item['elevation']) for item in data['elevation_colors']]
    mask_colors = [item['rgb'] for item in data.get('mask_colors', [])]

    elevations = [item['elevation'] for item in data['elevation_colors']]
    elev_min = min(elevations)
    elev_max = max(elevations)

    return elevation_colors, mask_colors, elev_min, elev_max


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

def detect_mask_rgb(image):
    """
    Detect non-terrain pixels using RGB color heuristics.

    Detects:
    - Red borders: R > 150 AND R > G*1.8 AND R > B*1.8
    - Near-white: R > 245 AND G > 245 AND B > 245
    - Near-black: R < 50 AND G < 50 AND B < 50
    - Gray text: max(R,G,B) - min(R,G,B) < 25 AND R < 180

    Args:
        image: RGB or RGBA image as numpy array

    Returns:
        Boolean mask (True = mask this pixel)
    """
    # Handle RGBA
    if image.shape[2] == 4:
        image = image[:, :, :3]

    r = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    b = image[:, :, 2].astype(np.float32)

    # Red borders: R > 150 AND R > G*1.8 AND R > B*1.8
    red_mask = (r > 150) & (r > g * 1.8) & (r > b * 1.8)

    # Near-white: R > 245 AND G > 245 AND B > 245
    white_mask = (r > 245) & (g > 245) & (b > 245)

    # Near-black: R < 50 AND G < 50 AND B < 50
    black_mask = (r < 50) & (g < 50) & (b < 50)

    # Gray text: low saturation AND not too bright
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    gray_mask = (max_rgb - min_rgb < 25) & (r < 180) & (r > 50)

    return red_mask | white_mask | black_mask | gray_mask


def detect_rivers(image, elevation):
    """
    Detect river pixels (blue-ish in land areas).

    Rivers are characterized by:
    - B > R AND B > G (blue dominant)
    - elevation > -100 (in land areas, not ocean)

    Args:
        image: RGB or RGBA image as numpy array
        elevation: 2D array of elevation values

    Returns:
        Boolean mask for river pixels (for future special handling)
    """
    # Handle RGBA
    if image.shape[2] == 4:
        image = image[:, :, :3]

    r = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    b = image[:, :, 2].astype(np.float32)

    # Blue-ish: B > R AND B > G
    blue_dominant = (b > r) & (b > g)

    # In land areas (not deep ocean)
    land_area = elevation > -100

    return blue_dominant & land_area


def detect_elevation_artifacts(heightmap, window_size=5, threshold=300.0):
    """
    Detect artifacts via local elevation variance.

    High local standard deviation indicates text/border causing elevation spikes.

    Args:
        heightmap: 2D array of elevation values
        window_size: Size of local window for std calculation
        threshold: Std threshold in meters (pixels above this are artifacts)

    Returns:
        Boolean mask (True = artifact pixel)
    """
    from scipy.ndimage import uniform_filter

    # Replace NaN with 0 for calculation
    hm = np.nan_to_num(heightmap, nan=0)

    mean = uniform_filter(hm, size=window_size)
    mean_sq = uniform_filter(hm**2, size=window_size)
    local_std = np.sqrt(np.maximum(mean_sq - mean**2, 0))

    return local_std > threshold


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


def colors_to_elevation_only(map_region, color_tree, lut_elevations):
    """
    Convert map colors to elevation values WITHOUT masking.

    Args:
        map_region: RGB or RGBA image as numpy array
        color_tree: KDTree of elevation colors
        lut_elevations: Array of elevation values corresponding to colors

    Returns:
        heightmap: 2D array of elevation values
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

    return heightmap


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
                elevation_colors=None, elev_min=None, elev_max=None,
                elev_artifact_threshold=300.0, dilate_iterations=3,
                mesh_scale=(100.0, 100.0, 20.0), mesh_decimate=4):
    """
    Main processing pipeline with hybrid mask detection.

    Pipeline:
    1. Load image
    2. Color → Elevation mapping
    3. RGB-space mask detection (red borders, white/black text, gray)
    4. River detection (separate layer, not masked)
    5. Elevation-space artifact detection (local std threshold)
    6. Combine masks + morphological cleanup (dilation)
    7. Interpolate masked regions
    8. Smooth, save, and generate 3D meshes

    Args:
        input_path: Path to input map image
        output_dir: Directory for output files
        smoothing: Gaussian smoothing sigma (0 to disable)
        elevation_colors: List of (rgb, elevation) tuples (required)
        elev_min: Minimum elevation for normalization
        elev_max: Maximum elevation for normalization
        elev_artifact_threshold: Elevation std threshold for artifact detection
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
    heightmap_raw = colors_to_elevation_only(map_region, color_tree, lut_elevations)
    print(f"  Raw elevation range: {heightmap_raw.min():.0f}m to {heightmap_raw.max():.0f}m")

    # Step 3: RGB-space mask detection
    print("Detecting masks in RGB space...")
    mask_rgb = detect_mask_rgb(map_region)
    print(f"  RGB mask: {mask_rgb.sum()} pixels ({100*mask_rgb.mean():.1f}%)")

    # Step 4: River detection (separate layer)
    print("Detecting rivers...")
    rivers = detect_rivers(map_region, heightmap_raw)
    print(f"  Rivers detected: {rivers.sum()} pixels ({100*rivers.mean():.1f}%)")

    # Step 5: Elevation-space artifact detection
    print(f"Detecting elevation artifacts (threshold={elev_artifact_threshold}m)...")
    mask_elev = detect_elevation_artifacts(heightmap_raw, threshold=elev_artifact_threshold)
    print(f"  Elevation artifact mask: {mask_elev.sum()} pixels ({100*mask_elev.mean():.1f}%)")

    # Step 6: Combine masks + morphological cleanup
    print(f"Combining masks and cleanup (dilate={dilate_iterations})...")
    mask_combined = combine_and_cleanup_mask(mask_rgb, mask_elev, dilate_iterations)
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
    path_mask_rgb = os.path.join(output_dir, f"{base_name}_02_mask_rgb.png")
    Image.fromarray((mask_rgb * 255).astype(np.uint8), mode='L').save(path_mask_rgb)
    outputs['mask_rgb'] = path_mask_rgb
    print(f"  Saved: {path_mask_rgb}")

    # 03: Elevation artifact mask
    path_mask_elev = os.path.join(output_dir, f"{base_name}_03_mask_elevation.png")
    Image.fromarray((mask_elev * 255).astype(np.uint8), mode='L').save(path_mask_elev)
    outputs['mask_elevation'] = path_mask_elev
    print(f"  Saved: {path_mask_elev}")

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
    %(prog)s map.png --elev-artifact-threshold 300 --dilate-iterations 3

Hybrid Mask Detection Pipeline:
    1. RGB-space detection: red borders, white/black text, gray labels
    2. Elevation-space detection: local std threshold for artifacts
    3. Morphological cleanup: dilation to catch anti-aliasing fringes
    4. River detection: saved separately for future handling

Output files (numbered by pipeline step):
    01_input.png           - Input image
    02_mask_rgb.png        - RGB-space detected mask
    03_mask_elevation.png  - Elevation artifact mask
    04_mask_combined.png   - Combined + dilated mask
    05_rivers_detected.png - River detection (grayscale)
    06_heightmap_raw.png   - Before interpolation (with holes)
    07_heightmap_filled.png - After interpolation
    08_heightmap_*.png     - Final smoothed (8-bit and 16-bit)
    09_visualization.png   - Debug visualization
    10_terrain.obj/.stl    - 3D mesh files

Notes:
    - Requires cmap.json colormap file (or specify with --colormap)
    - Tune --elev-artifact-threshold based on your map (higher = less masking)
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
    parser.add_argument('--elev-artifact-threshold', type=float, default=300.0,
                        help='Elevation std threshold for artifact detection (default: 300)')
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
    elevation_colors, mask_colors, elev_min, elev_max = load_colormap(args.colormap)
    print(f"  Elevation range: {elev_min}m to {elev_max}m")

    process_map(args.input, args.output_dir, args.smoothing,
                elevation_colors, elev_min, elev_max,
                args.elev_artifact_threshold, args.dilate_iterations,
                mesh_scale, args.mesh_decimate)


if __name__ == '__main__':
    main()
