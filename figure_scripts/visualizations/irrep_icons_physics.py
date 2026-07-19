import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, Ellipse, Polygon
import numpy as np
from matplotlib.path import Path

# Configure matplotlib for high-quality output
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'cm',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.transparent': True,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02
})

def draw_isotropic_icon(ax, center=(0.5, 0.5), size=1.0):
    """Draw l=0 (Isotropic) representation as a perfect sphere"""
    cx, cy = center
    scale = size

    # Main sphere with gradient effect
    sphere = Circle((cx, cy), 0.3*scale, fc='#CE93D8', ec='#7B1FA2',
                   lw=2*scale, alpha=0.8, zorder=2)
    ax.add_patch(sphere)

    # Add highlight to give 3D sphere effect
    highlight = Circle((cx - 0.08*scale, cy + 0.08*scale), 0.12*scale,
                      fc='#E1BEE7', ec='none', alpha=0.5, zorder=3)
    ax.add_patch(highlight)

    # Add subtle shadow
    shadow = Circle((cx + 0.05*scale, cy - 0.05*scale), 0.25*scale,
                   fc='#4A148C', ec='none', alpha=0.2, zorder=1)
    ax.add_patch(shadow)

    # Add label
    ax.text(cx, cy + 0.45*scale, r"$\ell=0$", ha='center', va='center',
           fontsize=14*scale, fontweight='bold')
    ax.text(cx, cy - 0.45*scale, r"Isotropic", ha='center', va='center',
           fontsize=12*scale, color='#666')

def draw_deviatoric_icon(ax, center=(0.5, 0.5), size=1.0):
    """Draw l=2 (Deviatoric) representation as an ellipsoid/dumbbell shape"""
    cx, cy = center
    scale = size

    # Main ellipsoid (elongated)
    ellipsoid = Ellipse((cx, cy), 0.4*scale, 0.25*scale,
                       fc='#9FA8DA', ec='#3F51B5', lw=2*scale,
                       alpha=0.8, zorder=2)
    ax.add_patch(ellipsoid)

    # Add lobe details to show directional nature
    # Left lobe
    left_lobe = Circle((cx - 0.15*scale, cy), 0.12*scale,
                      fc='#7986CB', ec='none', alpha=0.6, zorder=3)
    ax.add_patch(left_lobe)

    # Right lobe
    right_lobe = Circle((cx + 0.15*scale, cy), 0.12*scale,
                       fc='#7986CB', ec='none', alpha=0.6, zorder=3)
    ax.add_patch(right_lobe)

    # Add gradient stripes to show deformation
    for i in range(3):
        x_pos = cx + (i - 1) * 0.1*scale
        stripe = Rectangle((x_pos - 0.01*scale, cy - 0.15*scale),
                          0.02*scale, 0.3*scale,
                          fc='#5C6BC0', alpha=0.4, zorder=4)
        ax.add_patch(stripe)

    # Add label
    ax.text(cx, cy + 0.45*scale, r"$\ell=2$", ha='center', va='center',
           fontsize=14*scale, fontweight='bold')
    ax.text(cx, cy - 0.45*scale, r"Deviatoric", ha='center', va='center',
           fontsize=12*scale, color='#666')

def draw_higher_order_icon(ax, center=(0.5, 0.5), size=1.0):
    """Draw l=4 (Higher-order) representation as multi-lobed shape"""
    cx, cy = center
    scale = size

    # Create 4-lobed clover/petal shape
    n_lobes = 4
    lobe_radius = 0.15 * scale
    lobe_distance = 0.2 * scale

    # Draw lobes at cardinal directions
    for i in range(n_lobes):
        angle = i * np.pi / 2
        x = cx + lobe_distance * np.cos(angle)
        y = cy + lobe_distance * np.sin(angle)

        # Each lobe
        lobe = Circle((x, y), lobe_radius,
                     fc='#80CBC4', ec='#00796B',
                     lw=1.5*scale, alpha=0.8, zorder=2)
        ax.add_patch(lobe)

        # Add inner detail
        inner = Circle((x, y), lobe_radius * 0.5,
                      fc='#4DB6AC', ec='none', alpha=0.6, zorder=3)
        ax.add_patch(inner)

    # Central connecting region
    center_region = Circle((cx, cy), 0.12*scale,
                          fc='#A5D6A7', ec='#00695C',
                          lw=1.5*scale, alpha=0.7, zorder=1)
    ax.add_patch(center_region)

    # Add connecting lines to show structure
    for i in range(n_lobes):
        angle = i * np.pi / 2
        x = cx + 0.1 * scale * np.cos(angle)
        y = cy + 0.1 * scale * np.sin(angle)
        x_end = cx + lobe_distance * np.cos(angle)
        y_end = cy + lobe_distance * np.sin(angle)
        ax.plot([x, x_end], [y, y_end],
               c='#00695C', lw=1*scale, alpha=0.5, zorder=0)

    # Add label
    ax.text(cx, cy + 0.5*scale, r"$\ell=4$", ha='center', va='center',
           fontsize=14*scale, fontweight='bold')
    ax.text(cx, cy - 0.5*scale, r"Higher-order", ha='center', va='center',
           fontsize=12*scale, color='#666')

# Alternative l=2 representation (dumbbell shape)
def draw_deviatoric_icon_alt(ax, center=(0.5, 0.5), size=1.0):
    """Alternative l=2 representation as a classic dumbbell shape"""
    cx, cy = center
    scale = size

    # Connection bar
    bar = Rectangle((cx - 0.1*scale, cy - 0.05*scale),
                   0.2*scale, 0.1*scale,
                   fc='#9FA8DA', ec='none', alpha=0.8, zorder=1)
    ax.add_patch(bar)

    # Left bulb
    left_bulb = Circle((cx - 0.2*scale, cy), 0.15*scale,
                      fc='#7986CB', ec='#3F51B5',
                      lw=2*scale, alpha=0.8, zorder=2)
    ax.add_patch(left_bulb)

    # Right bulb
    right_bulb = Circle((cx + 0.2*scale, cy), 0.15*scale,
                       fc='#7986CB', ec='#3F51B5',
                       lw=2*scale, alpha=0.8, zorder=2)
    ax.add_patch(right_bulb)

    # Add label
    ax.text(cx, cy + 0.45*scale, r"$\ell=2$", ha='center', va='center',
           fontsize=14*scale, fontweight='bold')
    ax.text(cx, cy - 0.45*scale, r"Deviatoric", ha='center', va='center',
           fontsize=12*scale, color='#666')

# Create and save each icon separately
def create_icon(icon_func, filename):
    """Create and save a single icon"""
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Draw the icon
    icon_func(ax, center=(0.5, 0.5), size=1.0)

    # Save with tight bounding box
    plt.savefig(filename, dpi=300, transparent=True,
                bbox_inches='tight', pad_inches=0.05)
    plt.close()

# Generate the three icons
print("Generating l=0 (Isotropic) icon...")
create_icon(draw_isotropic_icon, 'irrep_l0_isotropic.png')

print("Generating l=2 (Deviatoric) icon...")
create_icon(draw_deviatoric_icon, 'irrep_l2_deviatoric.png')

print("Generating l=4 (Higher-order) icon...")
create_icon(draw_higher_order_icon, 'irrep_l4_higherorder.png')

# Alternative l=2 representation
print("\nGenerating alternative l=2 (Deviatoric) dumbbell icon...")
create_icon(draw_deviatoric_icon_alt, 'irrep_l2_deviatoric_alt.png')

print("\nAll icons generated successfully!")
print("Files saved:")
print("  - irrep_l0_isotropic.png (sphere)")
print("  - irrep_l2_deviatoric.png (ellipsoid with lobes)")
print("  - irrep_l2_deviatoric_alt.png (dumbbell shape)")
print("  - irrep_l4_higherorder.png (4-lobed clover)")