import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.spatial.transform import Rotation
from itertools import product, combinations

# Configure for clean CVPR style
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'Arial',
    'mathtext.fontset': 'cm',
    'font.size': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': False,
    'axes.facecolor': 'none',
    'figure.facecolor': 'none'
})

# Professional color scheme from user
COLORS = {
    'atom_corner': '#0A6dAF',   # Deep blue for corners
    'atom_face':   '#A72061',   # Deep magenta-red for face centers
    'atom_inner':  '#926AAD',   # Light purple for inner atoms
    'bond':        '#444444',   # Even darker gray for bonds
    'arrow':       '#926AAD',   # Light purple for arrow
    'box_edge':    '#848385',   # Light gray for edges
    'highlight':   '#AC5326',   # Brown-orange for highlights
    'bg_light':    '#F5EBF4',   # Very light purple background
}

class CrystalStructure:
    def __init__(self, lattice_constant=1.0):
        self.a = lattice_constant
        self.positions = []
        self.types = []  # 0: corner, 1: face, 2: inner
        self.colors = []
        self.rotation_matrix = None  # Initialize rotation matrix

        self._generate_zinc_blende()
        self.positions = np.array(self.positions)

    def _generate_zinc_blende(self):
        """Generate zinc blende structure with all atom types"""
        # 1. Corner atoms
        corners = list(product([0, 1], repeat=3))
        for p in corners:
            self.positions.append(p)
            self.types.append(0)  # Corner
            self.colors.append(COLORS['atom_corner'])

        # Face center atoms
        faces = [
            [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
            [0.5, 0.5, 1], [0.5, 1, 0.5], [1, 0.5, 0.5]
        ]
        for p in faces:
            self.positions.append(p)
            self.types.append(1)  # Face
            self.colors.append(COLORS['atom_face'])

        # Inner tetrahedral atoms (light purple)
        # Located at 1/4 of body diagonals
        inners = [
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75]
        ]
        for p in inners:
            self.positions.append(p)
            self.types.append(2)  # Inner
            self.colors.append(COLORS['atom_inner'])

    def rotate(self, rotation_obj):
        """Apply rotation transformation"""
        # Center the rotation at (0.5, 0.5, 0.5)
        center = np.array([0.5, 0.5, 0.5])
        centered_pos = self.positions - center
        rotated_pos = rotation_obj.apply(centered_pos)
        self.positions = rotated_pos + center

        # Store the rotation matrix for drawing the box
        self.rotation_matrix = rotation_obj.as_matrix()

def draw_bonds(ax, positions, max_length=0.45):
    """Draw bonds between atoms based on distance"""
    num_atoms = len(positions)

    # Create list of bonds to draw
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            p1 = positions[i]
            p2 = positions[j]
            dist = np.linalg.norm(p1 - p2)

            # Zinc blende nearest neighbor distance is ~sqrt(3)/4 * a ≈ 0.433
            if dist < max_length:
                # Calculate bond opacity based on distance
                alpha = max(0.3, 0.8 - (dist / max_length) * 0.3)

                # Draw bond with gradient effect
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                       color=COLORS['bond'], linewidth=7.0, alpha=alpha, zorder=2)

def draw_unit_cell_box(ax, rotation_matrix=None, alpha=0.3):
    """Draw the unit cell box from (0,0,0) to (1,1,1)"""
    # Define vertices of the unit cell
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face (z=0)
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face (z=1)
    ])

    # Apply rotation if provided
    if rotation_matrix is not None:
        # Center at (0.5, 0.5, 0.5) before rotation
        center = np.array([0.5, 0.5, 0.5])
        vertices_centered = vertices - center
        vertices_rotated = np.dot(vertices_centered, rotation_matrix.T)
        vertices = vertices_rotated + center

    # Define edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]

    # Draw edges
    for edge in edges:
        points = vertices[edge]
        ax.plot(points[:, 0], points[:, 1], points[:, 2],
               color=COLORS['bond'], linewidth=6.0, alpha=alpha*1.5, zorder=1)

def draw_crystal(ax, crystal, view_angle=(20, -60), rotation_matrix=None):
    """Draw enhanced crystal structure"""
    # Set viewing angle
    ax.view_init(*view_angle)

    # 1. Draw unit cell box that follows the crystal
    draw_unit_cell_box(ax, rotation_matrix=rotation_matrix, alpha=0.25)

    # 2. Draw bonds
    draw_bonds(ax, crystal.positions, max_length=0.45)

    # 3. Draw atoms
    sizes = {0: 1200, 1: 1100, 2: 1000}  # Extra large atoms for all types

    # Sort atoms by z-coordinate for proper depth rendering
    sort_indices = np.argsort(crystal.positions[:, 2])

    for idx in sort_indices:
        pos = crystal.positions[idx]
        atype = crystal.types[idx]
        color = crystal.colors[idx]
        size = sizes[atype]

        # Draw atom sphere
        ax.scatter(pos[0], pos[1], pos[2],
                  c=[color], s=size,
                  edgecolors='black', linewidth=3.0,
                  alpha=0.9, depthshade=False, zorder=10)

    # 4. Set axis properties - tighter limits to reduce whitespace
    center = np.mean(crystal.positions, axis=0)
    limit = 0.6  # Reduced from 0.8
    ax.set_xlim(center[0] - limit, center[0] + limit)
    ax.set_ylim(center[1] - limit, center[1] + limit)
    ax.set_zlim(center[2] - limit, center[2] + limit)

    # Style settings - ensure transparent background
    ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')

    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

def add_box_and_title(fig, ax_bbox, label_text, math_text):
    """Add rounded box and labels with enhanced styling"""
    # Get position of the 3D axis in figure coordinates
    bbox = ax_bbox.get_position()
    x0, y0, width, height = bbox.x0, bbox.y0, bbox.width, bbox.height

    # Define box padding
    pad = 0.025
    rect = FancyBboxPatch(
        (x0 - pad, y0 - pad), width + 2*pad, height + 2*pad,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor='white', edgecolor=COLORS['box_edge'],
        linewidth=2.0, transform=fig.transFigure, zorder=0
    )
    fig.patches.append(rect)

    # Add inner subtle box for elegance
    rect_inner = FancyBboxPatch(
        (x0 - pad + 0.01, y0 - pad + 0.01), width + 2*pad - 0.02, height + 2*pad - 0.02,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor='none', edgecolor=COLORS['bg_light'],
        linewidth=1.0, transform=fig.transFigure, zorder=0.5
    )
    fig.patches.append(rect_inner)

    # Add text with enhanced styling
    fig.text(x0 + width/2, y0 + height + 0.045, label_text,
             ha='center', va='bottom', fontsize=15, fontweight='bold', color='#333333')
    fig.text(x0 + width/2, y0 + height + 0.01, math_text,
             ha='center', va='bottom', fontsize=17, fontweight='bold', color='#000000')

def create_flowchart_visualization():
    """Create enhanced vertical flowchart visualization"""

    # 1. Prepare data
    crystal_original = CrystalStructure()
    crystal_rotated = CrystalStructure()

    # Define rotation: 30° around Z-axis, 45° around X-axis for better visualization
    rot = Rotation.from_euler('zx', [30, 45], degrees=True)
    crystal_rotated.rotate(rot)

    # 2. Create figure with enhanced layout
    fig = plt.figure(figsize=(8, 10), facecolor='none')

    # 3. Create two 3D subplots with better positioning - make them square
    # left, bottom, width, height
    ax_top = fig.add_axes([0.25, 0.58, 0.5, 0.35], projection='3d')
    ax_btm = fig.add_axes([0.25, 0.08, 0.5, 0.35], projection='3d')

    # Set consistent viewing angles for both crystals
    view_angles = (25, -60)  # Slightly adjusted for better view
    ax_top.view_init(*view_angles)
    ax_btm.view_init(*view_angles)

    # 4. Draw crystals
    draw_crystal(ax_top, crystal_original, view_angle=view_angles, rotation_matrix=crystal_original.rotation_matrix)
    draw_crystal(ax_btm, crystal_rotated, view_angle=view_angles, rotation_matrix=crystal_rotated.rotation_matrix)

    # 5. Add boxes and labels
    add_box_and_title(fig, ax_top, "Original Structure", r"$X$")
    add_box_and_title(fig, ax_btm, "Rotated Structure", r"$\mathcal{R} \cdot X$")

    # 6. Draw enhanced connection arrow
    top_bbox = ax_top.get_position()
    btm_bbox = ax_btm.get_position()

    x_start = top_bbox.x0 - 0.02
    y_start = top_bbox.y0 + top_bbox.height/2
    x_end = btm_bbox.x0 - 0.02
    y_end = btm_bbox.y0 + btm_bbox.height/2

    # Main curved arrow with enhanced styling
    arrow = FancyArrowPatch(
        posA=(x_start, y_start),
        posB=(x_end, y_end),
        connectionstyle="arc3,rad=0.35",
        arrowstyle="-|>,head_width=0.4,head_length=0.6",
        color=COLORS['arrow'],
        linewidth=3.0,
        transform=fig.transFigure,
        zorder=50
    )
    fig.patches.append(arrow)

    # Add secondary decoration arrow
    arrow_decor = FancyArrowPatch(
        posA=(x_start + 0.05, y_start - 0.03),
        posB=(x_end + 0.05, y_end + 0.03),
        connectionstyle="arc3,rad=0.35",
        arrowstyle="->,head_width=0.3,head_length=0.4",
        color=COLORS['highlight'],
        linewidth=1.5,
        alpha=0.7,
        transform=fig.transFigure,
        zorder=49
    )
    fig.patches.append(arrow_decor)

    # 7. Add arrow labels with better positioning
    text_x = x_start - 0.15
    text_y = (y_start + y_end) / 2

    fig.text(text_x, text_y + 0.02, "3D Rotation",
             ha='left', fontsize=14, color='#444444')
    fig.text(text_x, text_y - 0.02, r"$\mathcal{R} \in SO(3)$",
             ha='left', fontsize=14, fontweight='bold', color='#2980b9')

    # Add rotation parameters
    fig.text(text_x - 0.08, text_y - 0.06, r"Rz = 30°, Rx = 45°",
             ha='left', fontsize=11, color='#666666', style='italic')

    # Save with high quality and reduced padding
    plt.savefig('crystal_structures.png', bbox_inches='tight',
                dpi=300, facecolor='none', edgecolor='none', pad_inches=0.02, transparent=True)
    plt.savefig('crystal_structures.svg', format='svg', bbox_inches='tight',
                facecolor='none', edgecolor='none', pad_inches=0.02, transparent=True)

    print("Generated: crystal_structures.png")
    print("Generated: crystal_structures.svg")

    # Also create separate individual views
    create_individual_views(crystal_original, crystal_rotated, view_angles)

def create_individual_views(original, rotated, view_angle):
    """Create separate high-quality individual views"""
    # Original - square format without title
    fig1 = plt.figure(figsize=(8, 8), facecolor='none')
    ax1 = fig1.add_subplot(111, projection='3d')
    draw_crystal(ax1, original, view_angle=view_angle, rotation_matrix=original.rotation_matrix)
    plt.savefig('crystal_original.png', bbox_inches='tight', dpi=300, pad_inches=0.02, transparent=True)
    plt.savefig('crystal_original.svg', format='svg', bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close()

    # Rotated - square format without title
    fig2 = plt.figure(figsize=(8, 8), facecolor='none')
    ax2 = fig2.add_subplot(111, projection='3d')
    draw_crystal(ax2, rotated, view_angle=view_angle, rotation_matrix=rotated.rotation_matrix)
    plt.savefig('crystal_rotated.png', bbox_inches='tight', dpi=300, pad_inches=0.02, transparent=True)
    plt.savefig('crystal_rotated.svg', format='svg', bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close()

    print("Generated: crystal_original.png/svg")
    print("Generated: crystal_rotated.png/svg")

if __name__ == "__main__":
    create_flowchart_visualization()