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
import os
import numpy as np
from PIL import Image
from scipy.spatial import KDTree
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


# ============================================================================
# COLOR LOOKUP TABLE (LUT) CONFIGURATION
# ============================================================================
# Standard hypsometric coloring scheme. Adjust RGB values based on your map.
# Format: ([R, G, B], elevation_in_meters)

ELEVATION_COLORS = [
    # Ocean / bathymetry (blue tones)
    ([150, 193, 228], -200),
    ([130, 180, 215], -500),
    ([110, 165, 200], -1000),
    ([90, 150, 185], -2000),
    ([70, 130, 170], -3000),
    
    # Coastal lowlands (green tones)
    ([145, 187, 113], 50),
    ([150, 181, 124], 100),
    ([149, 184, 104], 150),
    ([143, 167, 109], 200),
    ([153, 179, 114], 250),
    
    # Transitional (yellow-green)
    ([170, 195, 120], 300),
    ([180, 205, 125], 400),
    ([187, 209, 127], 500),
    ([189, 212, 132], 600),
    ([184, 214, 154], 700),
    
    # Hills (tan/yellow tones)
    ([200, 200, 140], 800),
    ([210, 195, 145], 1000),
    ([215, 190, 150], 1200),
    ([220, 185, 140], 1500),
    
    # Mountains (brown tones)
    ([200, 170, 130], 1800),
    ([185, 155, 115], 2000),
    ([170, 140, 100], 2500),
    ([155, 125, 95], 3000),
    
    # High peaks (gray/white)
    ([180, 175, 170], 3500),
    ([200, 195, 190], 4000),
    ([220, 215, 212], 4500),
    ([240, 238, 235], 5000),
    
    # Rivers/water bodies within terrain
    ([77, 112, 142], 100),
]

# Colors to mask (non-terrain: text, borders, legends, background)
MASK_COLORS = [
    [255, 255, 255],  # Pure white (background/text halo)
    [254, 254, 254],  # Near-white
    [253, 253, 253],
    [255, 253, 254],
    [0, 0, 0],        # Black (text)
    [50, 50, 50],     # Dark gray (text)
    [100, 100, 100],  # Gray (text)
    [180, 30, 30],    # Red (borders)
    [200, 50, 50],
    [220, 70, 70],
    [190, 40, 40],
]

# Processing parameters
MASK_THRESHOLD = 30       # Color distance threshold for masking
SMOOTHING_SIGMA = 3.0     # Gaussian smoothing sigma
ELEVATION_MIN = -3000     # Minimum elevation for normalization
ELEVATION_MAX = 5000      # Maximum elevation for normalization


def load_image(path):
    """Load image and convert to numpy array."""
    img = Image.open(path)
    return np.array(img)


def build_color_trees(elevation_colors, mask_colors):
    """Build KD-trees for fast color lookup."""
    lut_colors = np.array([c[0] for c in elevation_colors])
    lut_elevations = np.array([c[1] for c in elevation_colors])
    mask_rgb = np.array(mask_colors)
    
    color_tree = KDTree(lut_colors)
    mask_tree = KDTree(mask_rgb)
    
    return color_tree, mask_tree, lut_elevations


def extract_map_region(img_array, margins=None):
    """
    Extract the map region, excluding borders and legends.
    
    Args:
        img_array: Input image as numpy array
        margins: Dict with keys 'top', 'bottom', 'left', 'right' specifying pixels to crop
                 If None, uses automatic detection (not implemented) or defaults
    
    Returns:
        Cropped map region
    """
    if margins is None:
        # Default margins - adjust based on your map
        margins = {
            'top': 60,
            'bottom': 50,
            'left': 0,
            'right': 400  # Exclude legend on right
        }
    
    h, w = img_array.shape[:2]
    return img_array[
        margins['top']:h - margins['bottom'],
        margins['left']:w - margins['right']
    ]


def colors_to_elevation(map_region, color_tree, mask_tree, lut_elevations, 
                        mask_threshold=MASK_THRESHOLD):
    """
    Convert map colors to elevation values.
    
    Returns:
        heightmap: 2D array of elevation values (NaN where masked)
        mask: 2D boolean array indicating masked pixels
    """
    h, w, c = map_region.shape
    pixels = map_region.reshape(-1, 3)
    
    # Find nearest elevation color for each pixel
    distances, indices = color_tree.query(pixels)
    
    # Find distance to nearest mask color
    mask_distances, _ = mask_tree.query(pixels)
    
    # Create mask: pixels closer to mask colors than terrain colors
    mask = mask_distances < np.minimum(distances, mask_threshold)
    
    # Map to elevations
    pixel_elevations = lut_elevations[indices].astype(float)
    pixel_elevations[mask] = np.nan
    
    heightmap = pixel_elevations.reshape(h, w)
    mask_2d = mask.reshape(h, w)
    
    return heightmap, mask_2d


def interpolate_masked_regions(heightmap):
    """Fill masked (NaN) regions using nearest-neighbor interpolation."""
    valid_mask = ~np.isnan(heightmap)
    indices = ndimage.distance_transform_edt(
        ~valid_mask, return_distances=False, return_indices=True
    )
    return heightmap[tuple(indices)]


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


def create_visualization(map_region, heightmap, mask, output_path):
    """Create visualization of the extraction process."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    axes[0, 0].imshow(map_region)
    axes[0, 0].set_title("Original map region")
    axes[0, 0].axis('off')
    
    im1 = axes[0, 1].imshow(heightmap, cmap='terrain', vmin=-1000, vmax=2000)
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


def process_map(input_path, output_dir='.', margins=None, smoothing=SMOOTHING_SIGMA):
    """
    Main processing pipeline.
    
    Args:
        input_path: Path to input map image
        output_dir: Directory for output files
        margins: Optional crop margins dict
        smoothing: Gaussian smoothing sigma (0 to disable)
    
    Returns:
        Dict with paths to output files
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    print(f"Loading image: {input_path}")
    img_array = load_image(input_path)
    print(f"  Shape: {img_array.shape}, dtype: {img_array.dtype}")
    
    print("Building color lookup tables...")
    color_tree, mask_tree, lut_elevations = build_color_trees(ELEVATION_COLORS, MASK_COLORS)
    print(f"  {len(ELEVATION_COLORS)} elevation colors, {len(MASK_COLORS)} mask colors")
    
    print("Extracting map region...")
    map_region = extract_map_region(img_array, margins)
    print(f"  Region shape: {map_region.shape}")
    
    print("Converting colors to elevation...")
    heightmap, mask = colors_to_elevation(map_region, color_tree, mask_tree, lut_elevations)
    print(f"  Masked pixels: {mask.sum()} ({100*mask.mean():.1f}%)")
    print(f"  Elevation range: {np.nanmin(heightmap):.0f}m to {np.nanmax(heightmap):.0f}m")
    
    print("Interpolating masked regions...")
    heightmap_filled = interpolate_masked_regions(heightmap)
    
    if smoothing > 0:
        print(f"Applying Gaussian smoothing (sigma={smoothing})...")
        heightmap_final = gaussian_filter(heightmap_filled, sigma=smoothing)
    else:
        heightmap_final = heightmap_filled
    
    print(f"  Final range: {heightmap_final.min():.0f}m to {heightmap_final.max():.0f}m")
    
    # Save outputs
    outputs = {}
    
    path_16bit = os.path.join(output_dir, f"{base_name}_heightmap_16bit.png")
    save_heightmap(heightmap_final, path_16bit, bits=16)
    outputs['heightmap_16bit'] = path_16bit
    print(f"  Saved: {path_16bit}")
    
    path_8bit = os.path.join(output_dir, f"{base_name}_heightmap_8bit.png")
    save_heightmap(heightmap_final, path_8bit, bits=8)
    outputs['heightmap_8bit'] = path_8bit
    print(f"  Saved: {path_8bit}")
    
    path_viz = os.path.join(output_dir, f"{base_name}_visualization.png")
    create_visualization(map_region, heightmap_final, mask, path_viz)
    outputs['visualization'] = path_viz
    print(f"  Saved: {path_viz}")
    
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
    %(prog)s map.png --smoothing 5.0 --margins 60,50,0,400

Notes:
    - The color LUT is configured for standard hypsometric coloring
    - You may need to sample colors from your specific map and adjust the LUT
    - Use --margins to crop borders, legends, and non-map areas
        """
    )
    parser.add_argument('input', help='Input map image (PNG, JPG, etc.)')
    parser.add_argument('--output-dir', '-o', default='.', help='Output directory')
    parser.add_argument('--smoothing', '-s', type=float, default=SMOOTHING_SIGMA,
                        help=f'Gaussian smoothing sigma (default: {SMOOTHING_SIGMA}, 0 to disable)')
    parser.add_argument('--margins', '-m', type=str, default=None,
                        help='Crop margins as top,bottom,left,right (e.g., 60,50,0,400)')
    
    args = parser.parse_args()
    
    margins = None
    if args.margins:
        parts = [int(x) for x in args.margins.split(',')]
        if len(parts) == 4:
            margins = {'top': parts[0], 'bottom': parts[1], 'left': parts[2], 'right': parts[3]}
    
    process_map(args.input, args.output_dir, margins, args.smoothing)


if __name__ == '__main__':
    main()
