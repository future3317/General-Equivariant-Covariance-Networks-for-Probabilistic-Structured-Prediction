import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, PathPatch, Rectangle
from matplotlib.path import Path
import numpy as np
import matplotlib.patheffects as path_effects

# ==========================================
# 1. CONFIGURATION & STYLE
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.size': 11,
    'axes.linewidth': 1.0,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Nature/ICML Professional Palette
COLORS = {
    'bg':         '#FFFFFF',
    'input_box':  '#F3F4F6',
    'atom_A':     '#EF5350', # Red
    'atom_B':     '#42A5F5', # Blue
    'backbone':   '#E8F5E9', # Light Green
    'backbone_d': '#2E7D32', # Dark Green
    'tensor_bg':  '#E3F2FD', # Light Blue
    'tensor_d':   '#1565C0', # Dark Blue
    'lie_alg':    '#FFF3E0', # Light Orange
    'lie_border': '#EF6C00', # Dark Orange
    'spd_cone':   '#F3E5F5', # Light Purple
    'spd_border': '#7B1FA2', # Dark Purple
    'arrow':      '#546E7A',
    'text':       '#263238'
}

def add_box_shadow(ax, x, y, w, h, radius=0.1, offset=(0.03, -0.03), alpha=0.1):
    shadow = FancyBboxPatch((x+offset[0], y+offset[1]), w, h, 
                           boxstyle=f"round,pad=0,rounding_size={radius}",
                           fc='black', ec='none', alpha=alpha, zorder=0)
    ax.add_patch(shadow)

# ==========================================
# 2. DRAWING COMPONENTS
# ==========================================

def draw_input_stage(ax, xy):
    x, y = xy
    # Background
    w, h = 2.2, 2.5
    add_box_shadow(ax, x, y, w, h)
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1", 
                         fc=COLORS['input_box'], ec='#B0BEC5', lw=1)
    ax.add_patch(box)
    
    # Crystal Structure
    cx, cy = x + w/2, y + h/2 + 0.1
    atoms = [
        (cx-0.4, cy-0.4, COLORS['atom_A']), (cx+0.4, cy+0.4, COLORS['atom_A']),
        (cx-0.3, cy+0.5, COLORS['atom_B']), (cx+0.5, cy-0.3, COLORS['atom_B']),
        (cx, cy, COLORS['atom_B'])
    ]
    
    # Bonds
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            p1, p2 = np.array(atoms[i][:2]), np.array(atoms[j][:2])
            if np.linalg.norm(p1-p2) < 0.9:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c='#CFD8DC', lw=1.5, zorder=2)
                
    # Draw Atoms
    for ax_x, ax_y, c in atoms:
        circle = Circle((ax_x, ax_y), 0.12, fc=c, ec='white', lw=0.5, zorder=3)
        add_box_shadow(ax, ax_x-0.12, ax_y-0.12, 0.24, 0.24, radius=0.12, offset=(0.02,-0.02))
        ax.add_patch(circle)
        
    # Rotation Symbol
    arc = FancyArrowPatch((cx-0.7, cy+0.7), (cx-0.4, cy+0.9), 
                          connectionstyle="arc3,rad=-0.3", 
                          arrowstyle="->", color=COLORS['text'], lw=1.5)
    ax.add_patch(arc)
    ax.text(cx-0.8, cy+0.85, r"$R \in O(3)$", fontsize=8)

    ax.text(x+w/2, y+0.25, r"\textbf{Input} $X$", ha='center', fontweight='bold')
    ax.text(x+w/2, y+0.1, "Point Cloud", ha='center', fontsize=8, color='#555')

def draw_mace_backbone(ax, xy):
    x, y = xy
    w, h = 2.0, 3.0
    
    # Main Box
    add_box_shadow(ax, x, y, w, h)
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1", 
                         fc=COLORS['backbone'], ec=COLORS['backbone_d'], lw=1.5)
    ax.add_patch(box)
    
    # Internal Layers (Message Passing)
    for i in range(3):
        ly = y + 0.6 + i*0.7
        layer = FancyBboxPatch((x+0.2, ly), w-0.4, 0.4, boxstyle="round,pad=0,rounding_size=0.05", 
                               fc='white', ec=COLORS['backbone_d'], lw=1, alpha=0.9)
        ax.add_patch(layer)
        ax.text(x+w/2, ly+0.2, f"Interaction {i+1}", ha='center', va='center', fontsize=7)
        
    ax.text(x+w/2, y+h-0.3, r"\textbf{MACE Backbone}", ha='center', fontweight='bold', color=COLORS['backbone_d'])
    ax.text(x+w/2, y+0.3, "E(3)-Equivariant\nFeatures", ha='center', fontsize=8, style='italic', color='#444')

def draw_covariance_head(ax, xy):
    x, y = xy
    w, h = 2.8, 2.0

    # Group Box
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1",
                         fc=COLORS['tensor_bg'], ec=COLORS['tensor_d'], lw=1, linestyle='--')
    ax.add_patch(box)

    # Title
    ax.text(x+w/2, y+h-0.25, r"Covariance Head", ha='center', fontweight='bold', fontsize=9, color=COLORS['tensor_d'])

    # Visualizing 2×(l=0)⊕2×(l=2)⊕1×(l=4)
    # Each irreducible representation shown once with multiplicity label

    # Starting positions for each type
    spacing = 0.7
    cx0, cy0 = x + 0.7, y + 0.8  # L=0 position
    cx2, cy2 = cx0 + spacing, y + 0.8  # L=2 position
    cx4, cy4 = cx2 + spacing, y + 0.8  # L=4 position

    # L=0 - Scalar (2 copies)
    # Draw as concentric circles representing spherical symmetry
    circle_outer = Circle((cx0, cy0), 0.2, fc='#CE93D8', ec='#7B1FA2', lw=1.2, alpha=0.8)
    ax.add_patch(circle_outer)
    circle_inner = Circle((cx0, cy0), 0.12, fc='#E1BEE7', ec='none', alpha=0.6)
    ax.add_patch(circle_inner)
    ax.text(cx0, cy0 + 0.35, r"$2\times(\ell=0)$", ha='center', fontsize=8, fontweight='bold')
    ax.text(cx0, cy0 - 0.35, r"scalar", ha='center', fontsize=7, color='#666')

    # Direct sum symbol
    ax.text(cx0 + 0.35, cy0, r"$\oplus$", ha='center', va='center', fontsize=14, color='#555')

    # L=2 - Rank-2 tensor (2 copies)
    # Draw as 3x3 matrix pattern showing symmetry
    matrix_size = 0.07
    for row in range(3):
        for col in range(3):
            x_pos = cx2 + (col - 1) * (matrix_size + 0.015)
            y_pos = cy2 + (row - 1) * (matrix_size + 0.015)
            # Highlight symmetric components (upper triangle including diagonal)
            if row <= col:
                rect = Rectangle((x_pos, y_pos), matrix_size, matrix_size,
                               fc='#9FA8DA', ec='#3F51B5', lw=0.8, alpha=0.9)
            else:
                rect = Rectangle((x_pos, y_pos), matrix_size, matrix_size,
                               fc='#C5CAE9', ec='#3F51B5', lw=0.8, alpha=0.3)
            ax.add_patch(rect)
    ax.text(cx2, cy2 + 0.35, r"$2\times(\ell=2)$", ha='center', fontsize=8, fontweight='bold')
    ax.text(cx2, cy2 - 0.35, r"rank-2 tensor", ha='center', fontsize=7, color='#666')

    # Direct sum symbol
    ax.text(cx2 + 0.35, cy0, r"$\oplus$", ha='center', va='center', fontsize=14, color='#555')

    # L=4 - Rank-4 tensor (1 copy)
    # Draw as a complex pattern representing higher-order structure
    # Use multiple circles with radial lines to show complexity
    for r in range(3):
        radius = 0.08 + r * 0.06
        alpha = 0.7 - r * 0.2
        circle_l4 = Circle((cx4, cy4), radius, fc='none', ec='#00796B',
                          lw=1.5, alpha=alpha)
        ax.add_patch(circle_l4)
        # Add radial lines for each circle
        n_lines = 8 * (r + 1)  # More lines for outer circles
        for angle in np.linspace(0, 2*np.pi, n_lines, endpoint=False):
            x_start = cx4 + radius * 0.6 * np.cos(angle)
            y_start = cy4 + radius * 0.6 * np.sin(angle)
            x_end = cx4 + radius * np.cos(angle)
            y_end = cy4 + radius * np.sin(angle)
            ax.plot([x_start, x_end], [y_start, y_end],
                   c='#00796B', lw=0.8, alpha=alpha)
    ax.text(cx4, cy4 + 0.45, r"$1\times(\ell=4)$", ha='center', fontsize=8, fontweight='bold')
    ax.text(cx4, cy4 - 0.45, r"rank-4 tensor", ha='center', fontsize=7, color='#666')

    # Bottom annotation
    ax.text(x+w/2, y+0.15, r"$\mathrm{Sym}^2(\rho_c) \cong 21$ params", ha='center', fontsize=8)

def draw_geometry_transform(ax, xy):
    """Draws the core Lie Algebra -> Manifold visualization"""
    x, y = xy
    
    # 1. Lie Algebra (Flat Plane)
    # Parallelogram
    lx, ly = x, y + 0.5
    path_coords = [
        (lx, ly), (lx+1.8, ly), (lx+1.4, ly+1.2), (lx-0.4, ly+1.2), (lx, ly)
    ]
    poly = patches.Polygon(path_coords, closed=True, fc=COLORS['lie_alg'], ec=COLORS['lie_border'], lw=1, alpha=0.8)
    ax.add_patch(poly)
    
    # Grid lines on plane
    ax.plot([lx+0.2, lx-0.2+1.4], [ly, ly+1.2], c=COLORS['lie_border'], lw=0.5, alpha=0.5)
    ax.plot([lx+0.9, lx+0.5+1.4], [ly, ly+1.2], c=COLORS['lie_border'], lw=0.5, alpha=0.5)
    ax.plot([lx-0.1, lx+1.7], [ly+0.6, ly+0.6], c=COLORS['lie_border'], lw=0.5, alpha=0.5)

    # Matrix A on the plane
    box_A = FancyBboxPatch((lx+0.5, ly+0.4), 0.5, 0.5, boxstyle="square,pad=0", fc='white', ec='black', lw=1)
    ax.add_patch(box_A)
    ax.text(lx+0.75, ly+0.65, r"$A$", ha='center', va='center', fontweight='bold')
    
    ax.text(lx+0.7, ly-0.25, r"Symmetric Lie Algebra $\mathfrak{gl}_{\mathrm{sym}}(6)$", ha='center', fontsize=9, fontweight='bold')
    ax.text(lx+0.7, ly-0.5, r"(Unconstrained)", ha='center', fontsize=8, color='#555')

    # 2. Transition Arrow (Matrix Exp)
    # Curved arrow jumping from plane to cone
    arr_start = (lx+1.6, ly+0.6)
    arr_end = (x+3.0, ly+0.6)
    arrow = FancyArrowPatch(posA=arr_start, posB=arr_end,
                            connectionstyle="arc3,rad=-0.2",
                            arrowstyle='-|>,head_length=8,head_width=6',
                            color='#2E7D32', lw=2, zorder=10)
    ax.add_patch(arrow)
    ax.text((arr_start[0]+arr_end[0])/2, ly+1.0, r"$\exp(A)$", ha='center', color='#2E7D32', fontweight='bold', fontsize=10, 
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#C8E6C9', alpha=1.0))

    # 3. SPD Manifold (Cone)
    cx, cy = x + 3.5, y
    
    # Draw Cone
    # Base Ellipse
    ellipse = Ellipse((cx, cy+1.6), 1.8, 0.6, fc=COLORS['spd_cone'], ec=COLORS['spd_border'], alpha=0.3, lw=1)
    ax.add_patch(ellipse)
    # Sides
    path_cone = Path([(cx-0.9, cy+1.6), (cx, cy), (cx+0.9, cy+1.6)], 
                     [Path.MOVETO, Path.LINETO, Path.LINETO])
    patch_cone = PathPatch(path_cone, fc=COLORS['spd_cone'], ec=COLORS['spd_border'], alpha=0.5, lw=1.5)
    ax.add_patch(patch_cone)
    
    # Covariance Ellipsoid inside Cone
    sigma_ell = Ellipse((cx, cy+1.0), 0.6, 0.35, angle=-20, fc=COLORS['atom_A'], ec='white', alpha=0.8)
    ax.add_patch(sigma_ell)
    ax.text(cx, cy+1.0, r"$\Sigma$", ha='center', va='center', color='white', fontweight='bold')

    ax.text(cx, cy-0.25, r"SPD Manifold $\mathcal{P}_6$", ha='center', fontsize=9, fontweight='bold')
    ax.text(cx, cy-0.5, r"(Geometry Aware)", ha='center', fontsize=8, color='#555')

def draw_loss_box(ax, xy):
    x, y = xy
    w, h = 4.0, 1.0
    
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.1", 
                         fc='#FFEBEE', ec='#C62828', lw=1.5)
    ax.add_patch(box)
    
    # Equation
    eq = r"$\mathcal{L}_{stable} = \frac{1}{2}\mathrm{Tr}(A) + \frac{1}{2}\Delta \mathbf{c}^\top \exp(-A) \Delta \mathbf{c}$"
    ax.text(x+w/2, y+h/2, eq, ha='center', va='center', fontsize=12, color='#B71C1C')
    ax.text(x+w/2, y+h+0.15, "Invariant NLL Loss", ha='center', fontweight='bold', fontsize=9, color='#C62828')

# ==========================================
# 3. MAIN ASSEMBLY
# ==========================================
fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
ax.axis('off')

# Coordinates
y_main = 2.0
x_input = 0.5
x_backbone = 3.5
x_head = 6.2
x_geo = 9.8

# 1. Draw Components
draw_input_stage(ax, (x_input, 1.8))
draw_mace_backbone(ax, (x_backbone, 1.5))

# Branching logic visualization
# Arrow to Mean Head (Top)
ax.add_patch(FancyArrowPatch((x_backbone+2.0, 3.5), (x_head, 4.8), connectionstyle="arc3,rad=0.1", arrowstyle='-|>', color=COLORS['arrow']))
box_mean = FancyBboxPatch((x_head, 4.4), 2.8, 0.8, boxstyle="round,pad=0.1", fc='#E1BEE7', ec='#8E24AA')
ax.add_patch(box_mean)
ax.text(x_head+1.4, 4.8, r"Mean Head $\mu(X)$", ha='center', fontweight='bold')
ax.text(x_head+1.4, 4.5, r"($\ell=0 \oplus \ell=2$ features)", ha='center', fontsize=8)

# Arrow to Covariance Head (Bottom)
ax.add_patch(FancyArrowPatch((x_backbone+2.0, 2.5), (x_head, 2.5), arrowstyle='-|>', color=COLORS['arrow']))
draw_covariance_head(ax, (x_head, 1.4))

# Arrow Covariance -> Geometry
ax.add_patch(FancyArrowPatch((x_head+2.8, 2.5), (x_geo, 2.5), arrowstyle='-|>', color=COLORS['arrow']))

# Geometry Section
draw_geometry_transform(ax, (x_geo, 1.5))

# Loss Section (Far Right/Top)
draw_loss_box(ax, (11.5, 4.5))

# Connecting Lines to Loss
# From Mean
ax.add_patch(FancyArrowPatch((x_head+2.8, 4.8), (11.5, 5.0), arrowstyle='-|>', color='#BDBDBD', lw=1, linestyle='--'))
# From Sigma (Manifold)
ax.add_patch(FancyArrowPatch((x_geo+3.5, 3.2), (13.5, 4.5), connectionstyle="arc3,rad=-0.2", arrowstyle='-|>', color='#BDBDBD', lw=1, linestyle='--'))

# Arrows between stages
ax.add_patch(FancyArrowPatch((x_input+2.2, 3.0), (x_backbone, 3.0), arrowstyle='-|>', lw=2, color=COLORS['arrow']))

# Annotations for "Equivariant Everywhere"
ax.text(8.0, 0.5, r"\textbf{Key Property:} $f(R \cdot X) = \rho(R) f(X)$ preserved at every stage", 
        ha='center', fontsize=12, bbox=dict(fc='#FFF8E1', ec='#FFC107', boxstyle='round,pad=0.4'))

plt.tight_layout()
plt.savefig('figure1_architecture.pdf')
plt.savefig('figure1_architecture.png')
print("Figure 1 generated successfully.")
plt.show()