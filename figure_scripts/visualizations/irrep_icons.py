import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle
import numpy as np

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

def draw_l0_icon(ax, center=(0.5, 0.5), size=1.0):
    """Draw l=0 (scalar) representation as concentric circles"""
    cx, cy = center
    scale = size

    # Outer circle - made smaller
    circle_outer = Circle((cx, cy), 0.25*scale, fc='#CE93D8', ec='#7B1FA2',
                         lw=2*scale, alpha=0.9, zorder=2)
    ax.add_patch(circle_outer)

    # Inner circle - made smaller proportionally
    circle_inner = Circle((cx, cy), 0.15*scale, fc='#E1BEE7', ec='none',
                         alpha=0.7, zorder=3)
    ax.add_patch(circle_inner)

    # Add label - standardized vertical position for alignment
    ax.text(cx, cy + 0.4*scale, r"$\ell=0$", ha='center', va='center',
           fontsize=14*scale, fontweight='bold')
    ax.text(cx, cy - 0.4*scale, r"scalar", ha='center', va='center',
           fontsize=12*scale, color='#666')

def draw_l2_icon(ax, center=(0.5, 0.5), size=1.0):
    """Draw l=2 (rank-2 tensor) representation as 3x3 matrix"""
    cx, cy = center
    scale = size

    matrix_size = 0.1 * scale
    spacing = 0.015 * scale

    # Draw 3x3 matrix
    for row in range(3):
        for col in range(3):
            x_pos = cx + (col - 1) * (matrix_size + spacing)
            y_pos = cy + (row - 1) * (matrix_size + spacing)

            # Highlight symmetric components (upper triangle including diagonal)
            if row <= col:
                rect = Rectangle((x_pos - matrix_size/2, y_pos - matrix_size/2),
                               matrix_size, matrix_size,
                               fc='#9FA8DA', ec='#3F51B5',
                               lw=1*scale, alpha=0.9, zorder=2)
            else:
                rect = Rectangle((x_pos - matrix_size/2, y_pos - matrix_size/2),
                               matrix_size, matrix_size,
                               fc='#C5CAE9', ec='#3F51B5',
                               lw=1*scale, alpha=0.3, zorder=2)
            ax.add_patch(rect)

    # Add label - standardized vertical position for alignment
    ax.text(cx, cy + 0.4*scale, r"$\ell=2$", ha='center', va='center',
           fontsize=14*scale, fontweight='bold')
    ax.text(cx, cy - 0.4*scale, r"rank-2 tensor", ha='center', va='center',
           fontsize=12*scale, color='#666')

def draw_l4_icon(ax, center=(0.5, 0.5), size=1.0):
    """Draw l=4 (rank-4 tensor) representation as complex pattern"""
    cx, cy = center
    scale = size

    # Draw multiple circles with radial lines
    for r in range(3):
        radius = (0.08 + r * 0.06) * scale
        alpha = 0.8 - r * 0.2

        # Circle
        circle = Circle((cx, cy), radius, fc='none', ec='#00796B',
                       lw=2*scale, alpha=alpha, zorder=2-r)
        ax.add_patch(circle)

        # Radial lines
        n_lines = 8 * (r + 1)
        for angle in np.linspace(0, 2*np.pi, n_lines, endpoint=False):
            x_start = cx + radius * 0.6 * np.cos(angle)
            y_start = cy + radius * 0.6 * np.sin(angle)
            x_end = cx + radius * np.cos(angle)
            y_end = cy + radius * np.sin(angle)
            ax.plot([x_start, x_end], [y_start, y_end],
                   c='#00796B', lw=1*scale, alpha=alpha, zorder=1)

    # Add label - standardized vertical position for alignment
    ax.text(cx, cy + 0.4*scale, r"$\ell=4$", ha='center', va='center',
           fontsize=14*scale, fontweight='bold')
    ax.text(cx, cy - 0.4*scale, r"rank-4 tensor", ha='center', va='center',
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
print("Generating l=0 (scalar) icon...")
create_icon(draw_l0_icon, 'irrep_l0_scalar.png')

print("Generating l=2 (rank-2 tensor) icon...")
create_icon(draw_l2_icon, 'irrep_l2_tensor.png')

print("Generating l=4 (rank-4 tensor) icon...")
create_icon(draw_l4_icon, 'irrep_l4_tensor.png')

print("\nAll icons generated successfully!")
print("Files saved:")
print("  - irrep_l0_scalar.png")
print("  - irrep_l2_tensor.png")
print("  - irrep_l4_tensor.png")