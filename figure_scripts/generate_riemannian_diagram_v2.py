"""
===============================================
ICML 2025 - Riemannian Manifold Visualization
===============================================
A professional visualization of the mapping between
Lie Algebra tangent space and SPD manifold for
equivariant uncertainty quantification.

Author: Generated for ICML 2025 Submission
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from typing import Optional, Tuple


# === Configuration ===
class PlotConfig:
    """ICML 2025 style configuration"""

    # Matplotlib settings for ICML submission
    RC_PARAMS = {
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Computer Modern'],
        'mathtext.fontset': 'cm',
        'axes.unicode_minus': False,
        'figure.dpi': 300,
        'font.size': 13,           # Reduced from 14 to 13
        'axes.labelsize': 13,
        'axes.titlesize': 15,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 12,
    }

    # Color palette - professional and accessible
    COLORS = {
        'lie_algebra': '#2980b9',     # Blue for Lie Algebra
        'spd_manifold': '#e67e22',    # Orange for SPD Manifold
        'grid': '#bdc3c7',            # Light gray for grids
        'text': '#2c3e50',            # Dark text
        'invalid': '#7f8c8d',         # Gray for invalid regions
        'accent1': '#34495e',         # Dark blue for accents
        'accent2': '#16a085',         # Teal for secondary elements
        'warning': '#c0392b',         # Red for warnings
        # ICML optimization: Professional color scheme (provided)
        'point_A': '#0A6dAF',         # Deep blue - for point A and ellipsoid Σ
        'point_A_prime': '#A72061',   # Deep magenta-red - for point A' and ellipsoid Σ'
        'action': '#926AAD',          # Light purple - for rotation/action ρ(R)
        'cone_surface': '#F6E4D0',    # Light beige for neutral background
    }

    # Plot dimensions
    FIGURE_SIZE = (14, 4.5)  # Reduced height from 6 to 4.5
    DPI = 300

    # 3D parameters
    LIE_VIEW = {'elev': 35, 'azim': -60}
    SPD_VIEW = {'elev': 25, 'azim': 45}


# === Custom 3D Arrow Class ===
class Arrow3D(FancyArrowPatch):
    """Custom 3D arrow for better visualization"""

    def __init__(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,
                 *args, **kwargs):
        """Initialize 3D arrow with coordinates"""
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None) -> float:
        """Project 3D coordinates to 2D plane"""
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[-1], ys[-1]))
        return np.min(zs)


# === Visualization Utilities ===
class VisualizationUtils:
    """Utility functions for 3D visualization"""

    @staticmethod
    def draw_curved_arrow_3d(ax: Axes3D, p1: np.ndarray, p2: np.ndarray,
                           height: float = 0.5, color: str = 'black',
                           label: Optional[str] = None, fontsize: int = 10,
                           alpha: float = 1.0) -> None:
        """
        Draw a smooth curved arrow in 3D space.

        Args:
            ax: 3D axes object
            p1: Starting point [x, y, z]
            p2: Ending point [x, y, z]
            height: Arrow curve height
            color: Arrow color
            label: Optional label text
            fontsize: Label font size
            alpha: Transparency
        """
        t = np.linspace(0, 1, 100)

        # Bezier curve interpolation
        x = (1 - t) * p1[0] + t * p2[0]
        y = (1 - t) * p1[1] + t * p2[1]
        z = (1 - t) * p1[2] + t * p2[2] + height * np.sin(t * np.pi)

        # Draw curve
        ax.plot(x, y, z, color=color, linewidth=1.0, linestyle='--',
               alpha=alpha, zorder=10) # <--- 添加 zorder=10

        # Add arrowhead
        arrow = Arrow3D([x[-10], x[-1]], [y[-10], y[-1]], [z[-10], z[-1]],
                       mutation_scale=12, lw=1, arrowstyle="-|>",
                       color=color, alpha=alpha, zorder=10) # <--- 添加 zorder=10
        ax.add_artist(arrow)

        # Add label if specified
        if label:
            mid_idx = len(x) // 2
            ax.text(x[mid_idx], y[mid_idx], z[mid_idx] + 1.35, label,
                   fontsize=fontsize, ha='center', color=color,
                   weight='bold', alpha=alpha, zorder=10) # <--- 添加 zorder=10

    @staticmethod
    def draw_geodesic_on_cone(ax: Axes3D, p1: np.ndarray, p2: np.ndarray,
                             color: str = 'black', label: Optional[str] = None,
                             fontsize: int = 10, alpha: float = 1.0) -> None:
        """
        Draw a geodesic curve on the SPD cone surface.

        In the SPD manifold with affine-invariant metric, the geodesic between
        two points Σ₁ and Σ₂ is given by:
        Σ(t) = Σ₁^{1/2} (Σ₁^{-1/2} Σ₂ Σ₁^{-1/2})^t Σ₁^{1/2}

        For visualization, we approximate this as a curve on the cone surface.

        Args:
            ax: 3D axes object
            p1: Starting point [x, y, z] on the cone
            p2: Ending point [x, y, z] on the cone
            color: Curve color
            label: Optional label text
            fontsize: Label font size
            alpha: Transparency
        """
        # Convert to cylindrical coordinates
        r1 = np.sqrt(p1[0]**2 + p1[1]**2)
        theta1 = np.arctan2(p1[1], p1[0])
        r2 = np.sqrt(p2[0]**2 + p2[1]**2)
        theta2 = np.arctan2(p2[1], p2[0])

        # Create geodesic path
        t = np.linspace(0, 1, 100)

        # Interpolate in the log-Euclidean space (approximation)
        # This gives a more natural curve on the cone surface
        log_r1 = np.log(r1 + 1e-6)  # Add small epsilon to avoid log(0)
        log_r2 = np.log(r2 + 1e-6)

        # Handle angle wrapping
        if theta2 - theta1 > np.pi:
            theta2 -= 2 * np.pi
        elif theta1 - theta2 > np.pi:
            theta2 += 2 * np.pi

        # Interpolate in log-space
        log_r = (1 - t) * log_r1 + t * log_r2
        theta = (1 - t) * theta1 + t * theta2

        # Convert back to Cartesian coordinates
        r = np.exp(log_r)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = r  # On the cone surface

        # Draw the geodesic curve
        ax.plot(x, y, z, color=color, linewidth=2.0,
               alpha=alpha, zorder=10)

        # Add arrowhead at the end
        arrow = Arrow3D([x[-10], x[-1]], [y[-10], y[-1]], [z[-10], z[-1]],
                       mutation_scale=15, lw=2, arrowstyle="-|>",
                       color=color, alpha=alpha, zorder=10)
        ax.add_artist(arrow)

        # Add label if specified
        if label:
            mid_idx = len(x) // 2
            # Offset the label slightly away from the surface
            offset = 0.2
            norm = np.array([x[mid_idx], y[mid_idx], 0])
            norm = norm / (np.linalg.norm(norm) + 1e-6)
            ax.text(x[mid_idx] + offset*norm[0],
                   y[mid_idx] + offset*norm[1],
                   z[mid_idx] + offset*norm[2] + 0.1,
                   label,
                   fontsize=fontsize, ha='center', color=color,
                   weight='bold', alpha=alpha, zorder=10)

    @staticmethod
    def draw_grid_plane(ax: Axes3D, xlim: Tuple[float, float],
                       ylim: Tuple[float, float], z: float = 0,
                       color: str = 'gray', alpha: float = 0.3,
                       grid_only: bool = True) -> None:
        """
        Draw a grid plane in 3D space.

        Args:
            ax: 3D axes object
            xlim: X-axis limits (min, max)
            ylim: Y-axis limits (min, max)
            z: Z-coordinate of the plane
            color: Grid color
            alpha: Transparency
            grid_only: If True, only draw wireframe
        """
        x = np.linspace(xlim[0], xlim[1], 11)
        y = np.linspace(ylim[0], ylim[1], 11)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, z)

        if grid_only:
            ax.plot_wireframe(X, Y, Z, color=color, alpha=alpha,
                            linewidth=1.0)  # Thicker lines (was 0.5)
        else:
            ax.plot_surface(X, Y, Z, color=color, alpha=alpha * 0.5,
                           shade=False)

    @staticmethod
    def draw_point_with_label(ax: Axes3D, point: np.ndarray,
                            label: str, color: str = 'blue',
                            size: int = 80, offset: float = 0.3) -> None:
        """
        Draw a point with a label in 3D space.

        Args:
            ax: 3D axes object
            point: Point coordinates [x, y, z]
            label: Label text
            color: Point color
            size: Point size
            offset: Label offset from point
        """
        # Draw point with high zorder to ensure it's on top
        ax.scatter([point[0]], [point[1]], [point[2]],
                  c=color, s=size, edgecolors='white',
                  linewidth=1.0, depthshade=False)
        ax.text(point[0], point[1] - offset, point[2], label,
               fontsize=13, color=color, weight='bold')

# === Manifold Visualizers ===
class LieAlgebraVisualizer:
    """Visualizer for Lie Algebra tangent space"""

    def __init__(self, ax: Axes3D, config: PlotConfig):
        """Initialize visualizer with axes and configuration"""
        self.ax = ax
        self.config = config
        self.ax.set_box_aspect((1, 1, 0.2))

    def draw_plane(self) -> None:
        """Draw the tangent space plane with light fill and visible grid"""
        # Add light surface fill first
        x = np.linspace(-2, 1.8, 11)  # Match grid x range
        y = np.linspace(-2, 2, 11)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        self.ax.plot_surface(X, Y, Z, color='#f5f5f5', alpha=0.2, shade=False)  # Very light gray

        # Draw grid lines on top with preserved color
        VisualizationUtils.draw_grid_plane(
            self.ax, (-2, 1.8), (-2, 2), z=0,
            color='#404040', alpha=0.6
        )

    def draw_rotated_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw points A and A' with 90-degree rotation.

        Returns:
            Tuple of (p1, p2) coordinates
        """
        # Define initial point (slightly above the plane)
        p1 = np.array([-1.2, -0.5, 0.05])

        # Apply 90-degree rotation
        theta_rot = np.pi / 2
        rot_mat = np.array([
            [np.cos(theta_rot), -np.sin(theta_rot)],
            [np.sin(theta_rot), np.cos(theta_rot)]
        ])
        p2_xy = rot_mat @ p1[:2]
        p2 = np.array([p2_xy[0], p2_xy[1], 0.05])  # Also slightly above the plane

        # Draw points with color coding for visual correspondence
        VisualizationUtils.draw_point_with_label(
            self.ax, p1, r"$A$", color=self.config.COLORS['point_A'], size=80, offset=0.4
        )
        VisualizationUtils.draw_point_with_label(
            self.ax, p2, r"$A'$", color=self.config.COLORS['point_A_prime'], size=80, offset=-0.3
        )

        return p1, p2

    def draw_rotation_arrow(self, p1: np.ndarray, p2: np.ndarray) -> None:
        """Draw curved arrow showing rotation"""
        VisualizationUtils.draw_curved_arrow_3d(
            self.ax, p1, p2, height=0.6,
            color=self.config.COLORS['action'],  # Use purple to match geodesic
            label=r"Linear action: $\rho(R)$",
            fontsize=14  # Reduced font size
        )

    def add_labels(self) -> None:
        """Add descriptive labels"""
        color = self.config.COLORS['text']
        self.ax.text2D(
            0.5, 0.08,
            r"Tangent Space $\mathcal{T}$ (Lie Algebra $\mathfrak{g}$)",
            transform=self.ax.transAxes, ha='center', fontsize=15,
            color=color, weight='bold'
        )
        self.ax.text2D(
            0.5, 0.03,
            r"Decomposition: $2\times(l=0) \oplus 2\times(l=2) \oplus 1\times(l=4)$",
            transform=self.ax.transAxes, ha='center', fontsize=12,
            color='#404040'  # Darker gray (was 'gray')
        )

    def finalize(self) -> None:
        """Set final view properties"""
        self.ax.view_init(**self.config.LIE_VIEW)
        # Ensure proper rendering order
        self.ax.set_axis_off()
        # Force redraw to ensure proper layering
        self.ax.figure.canvas.draw_idle()


class SPDManifoldVisualizer:
    """Visualizer for SPD manifold"""

    def __init__(self, ax: Axes3D, config: PlotConfig):
        """Initialize visualizer with axes and configuration"""
        self.ax = ax
        self.config = config
        self.ax.set_box_aspect((1, 1, 0.8))
        self.rotation_angle = np.pi / 2  # 90 degrees

    def draw_cone(self) -> None:
        """Draw a minimalist SPD cone for ICML submission"""
        # Generate mesh for cone
        r = np.linspace(0, 2.2, 50)
        theta = np.linspace(0, 2*np.pi, 80)  # Lower resolution for cleaner look
        R, Theta = np.meshgrid(r, theta)
        Xc = R * np.cos(Theta)
        Yc = R * np.sin(Theta)
        Zc = R

        # 1. Surface with professional gradient effect
        # Create subtle gray gradient from base to apex
        norm = plt.Normalize(Zc.min(), Zc.max())
        # Use grayscale colormap for professional look
        colors = plt.cm.Greys(norm(Zc))
        # Map to light gray range (0.7 to 0.95) for subtle effect
        colors = colors * 0.3 + 0.7  # Scale and shift to light gray range
        # Set uniform transparency
        colors[..., 3] = 0.15  # Alpha = 0.15 for subtle visibility

        self.ax.plot_surface(
            Xc, Yc, Zc,
            facecolors=colors,  # Use professional gray gradient
            rstride=4, cstride=4,  # Moderate mesh density
            linewidth=0.5,  # Thicker wireframe lines for visibility
            edgecolor=(0.5, 0.5, 0.5, 0.3),  # Medium gray wireframe with transparency
            antialiased=True,
            shade=True,  # Enable shading for depth
            lightsource=plt.matplotlib.colors.LightSource(azdeg=45, altdeg=45),
            zorder=1
        )

        # 2. Add thicker contour rings for better visibility
        n_levels = 5
        levels = np.linspace(0.5, 2.0, n_levels)
        for level in levels:
            theta = np.linspace(0, 2*np.pi, 100)
            x_ring = level * np.cos(theta)
            y_ring = level * np.sin(theta)
            z_ring = np.full_like(x_ring, level)

            self.ax.plot(
                x_ring, y_ring, z_ring,
                color=(0.4, 0.4, 0.4),  # Medium gray
                alpha=0.25,  # Higher visibility
                linewidth=0.8,  # Thicker lines
                zorder=2
            )

    def draw_invalid_region(self) -> None:
        """Draw non-SPD region indicator"""
        # Draw dotted grid below
        VisualizationUtils.draw_grid_plane(
            self.ax, (-2, 2), (-2, 2), z=-0.2,
            color=self.config.COLORS['invalid'], alpha=0.2,
            grid_only=True
        )

        # Add warning symbols
        self.ax.text(1.2, 1.2, -0.2, r"$\times$",
                    color=self.config.COLORS['warning'], fontsize=21,
                    ha='center', va='center')
        self.ax.text(-1.2, -1.2, -0.2, r"$\times$",
                    color=self.config.COLORS['warning'], fontsize=21,
                    ha='center', va='center')
        self.ax.text(0, -2.4, -0.2, "Non-SPD Region",
                    color=self.config.COLORS['warning'], fontsize=10,
                    ha='center', alpha=0.8)

    def draw_ellipsoids(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw two rotated ellipsoids representing covariance matrices.

        Returns:
            Tuple of (s1, s2) center positions
        """
        # Define positions
        s1_r, s1_theta = 1.2, 0.5
        s1_h = 1.2
        s1 = np.array([s1_r * np.cos(s1_theta), s1_r * np.sin(s1_theta), s1_h])

        s2_theta = s1_theta + self.rotation_angle
        s2 = np.array([s1_r * np.cos(s2_theta), s1_r * np.sin(s2_theta), s1_h])

        # Initial tilt for first ellipsoid - make orientation more visible
        initial_tilt = np.pi / 6  # 30 degrees for clearer orientation

        # Enhanced rotation for second ellipsoid - make it very clear
        # The ellipsoid itself should rotate by 90 degrees, matching the group action
        second_ellipsoid_rotation = initial_tilt + np.pi/2  # Full 90-degree rotation

        # Draw ellipsoids with color coding to match left plot points
        self._draw_single_ellipsoid(s1, initial_tilt, r"$\Sigma$",
                                  base_color=self.config.COLORS['point_A'])
        self._draw_single_ellipsoid(s2, second_ellipsoid_rotation, r"$\Sigma'$",
                                  base_color=self.config.COLORS['point_A_prime'])

        return s1, s2

    def _draw_single_ellipsoid(self, center: np.ndarray, rotation: float,
                             label: str, base_color: str = None) -> None:
        """Draw a beautiful ellipsoid at given position with rotation"""
        # Use provided color or default
        if base_color is None:
            base_color = '#ff6b35'  # Default orange color

        # Generate higher resolution sphere for smoother ellipsoid
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        # Shape into ellipsoid (anisotropic)
        x = x * 0.15  # Short axis
        y = y * 0.45  # Long axis
        z = z * 0.15  # Short axis

        # Apply rotation around Z axis with smoother rotation
        cos_a, sin_a = np.cos(rotation), np.sin(rotation)
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a
        z_rot = z

        # Translate to center
        x_final = x_rot + center[0]
        y_final = y_rot + center[1]
        z_final = z_rot + center[2]

        # Create color variations based on the base color
        # Convert hex to RGB
        from matplotlib import colors
        base_rgb = colors.to_rgb(base_color)

        # Create gradient effect
        colors = np.zeros((z.shape[0], z.shape[1], 4))
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                # Vary the brightness based on z position
                brightness = 0.6 + 0.4 * (z[i, j] - z.min()) / (z.max() - z.min())
                colors[i, j] = (*base_rgb, 0.8)  # Use base color with fixed alpha

        # Draw surface with base color
        surf = self.ax.plot_surface(
            x_final, y_final, z_final,
            facecolors=colors,
            alpha=0.8,
            rstride=1,
            cstride=1,
            antialiased=True,
            shade=True,
            lightsource=plt.matplotlib.colors.LightSource(azdeg=45, altdeg=45)
        )

        # Add subtle wireframe for better 3D perception
        self.ax.plot_wireframe(
            x_final, y_final, z_final,
            color='#8b4513',  # Saddle brown for contrast
            alpha=0.2,
            linewidth=0.3,
            rstride=4,
            cstride=4
        )

        # Add highlight edges on the major axis for emphasis
        # Draw the major axis ellipse
        theta_highlight = np.linspace(0, 2*np.pi, 100)
        x_major = 0.15 * np.cos(theta_highlight)
        y_major = 0.45 * np.sin(theta_highlight)
        z_major = np.zeros_like(x_major)

        # Rotate the major axis
        x_major_rot = x_major * cos_a - y_major * sin_a
        y_major_rot = x_major * sin_a + y_major * cos_a

        # Translate
        x_major_final = x_major_rot + center[0]
        y_major_final = y_major_rot + center[1]
        z_major_final = z_major + center[2]

        # Draw with base color
        self.ax.plot(
            x_major_final, y_major_final, z_major_final,
            color=base_color,
            alpha=0.9,
            linewidth=2
        )

        # Add an elegant shadow effect beneath the ellipsoid
        # Create a soft, diffuse shadow with gradient

        # Calculate shadow footprint on the ground plane
        shadow_scale = 0.9
        shadow_offset = 0.05  # Raise shadow higher for better visibility

        # Create multiple shadow layers for soft effect
        n_layers = 3
        for i in range(n_layers):
            layer_scale = shadow_scale + i * 0.05
            alpha = 0.15 * (1 - i / n_layers)  # Fade outer layers

            x_shadow = x_final * layer_scale + center[0]
            y_shadow = y_final * layer_scale + center[1]
            z_shadow = np.full_like(x_final, shadow_offset)

            # Draw shadow layer using base color with transparency
            self.ax.plot_surface(
                x_shadow, y_shadow, z_shadow,
                color=base_rgb,  # Use ellipsoid's base color for shadow
                alpha=alpha * 0.3,  # Very subtle
                shade=False,
                lightsource=None
            )

        # 椭圆轮廓线已删除，保持简洁

        # Don't draw label here - store it for later
        if not hasattr(self, '_ellipsoid_labels'):
            self._ellipsoid_labels = []
        self._ellipsoid_labels.append((center, label))

    def draw_transformation_arrow(self, s1: np.ndarray, s2: np.ndarray) -> None:
        """Draw both Euclidean interpolation (invalid) and geodesic (valid) paths"""

        # 1. First draw Euclidean interpolation (invalid path - leaves SPD cone)
        t = np.linspace(0, 1, 50)
        x_euclid = (1 - t) * s1[0] + t * s2[0]
        y_euclid = (1 - t) * s1[1] + t * s2[1]
        z_euclid = (1 - t) * s1[2] + t * s2[2]

        # Check if path goes outside cone (z < sqrt(x^2 + y^2))
        for i in range(len(t)):
            r = np.sqrt(x_euclid[i]**2 + y_euclid[i]**2)
            if z_euclid[i] < r * 0.95:  # Slightly below cone surface
                # Path goes outside SPD cone
                self.ax.plot(x_euclid, y_euclid, z_euclid,
                           color='red', alpha=0.7, linewidth=2,
                           linestyle='--', zorder=10,
                           label=r"Euclidean (Non-SPD)")

                # Add warning marker where path exits cone
                self.ax.plot(x_euclid[i], y_euclid[i], z_euclid[i],
                           'rx', markersize=10, markeredgewidth=2,
                           alpha=0.8, zorder=11)
                break
        else:
            # If entire path is inside (unlikely), still draw it
            self.ax.plot(x_euclid, y_euclid, z_euclid,
                       color='red', alpha=0.7, linewidth=2,
                       linestyle='--', zorder=10,
                       label=r"Euclidean (Non-SPD)")

        # 2. Draw geodesic (valid path - stays on cone surface)
        # Note: No label here to avoid overlap. Will add separate label later.

    def add_labels(self) -> None:
        """Add descriptive labels"""
        color = self.config.COLORS['text']
        # Store these to draw later after everything else
        self.labels_2d = [
            (0.5, 0.08, r"SPD Manifold $\mathcal{P}_6$ (Covariance Matrices $\Sigma$)",
             color, 15, 'bold'),
            (0.5, 0.03, r"Conjugate action: $\Sigma' = \rho(R)\Sigma\rho(R)^\top$",
             '#404040', 12, 'normal')  # Darker gray (was 'gray')
        ]

    def finalize(self) -> None:
        """Set final view properties"""
        self.ax.view_init(**self.config.SPD_VIEW)
        self.ax.set_axis_off()

        # Ellipsoid labels removed - will be added manually in PPT

        # Draw 2D labels last so they appear on top
        if hasattr(self, 'labels_2d'):
            for x, y, text, color, fontsize, weight in self.labels_2d:
                self.ax.text2D(
                    x, y, text,
                    transform=self.ax.transAxes, ha='center',
                    fontsize=fontsize+2, color=color, weight=weight
                )


# === Main Figure Generator ===
class RiemannianManifoldFigure:
    """Main class to generate the complete ICML figure"""

    def __init__(self, config: PlotConfig = None):
        """Initialize with configuration"""
        self.config = config or PlotConfig()
        plt.rcParams.update(self.config.RC_PARAMS)

    def generate(self) -> None:
        """Generate the complete figure"""
        # Create figure with subplots
        fig = plt.figure(
            figsize=self.config.FIGURE_SIZE,
            constrained_layout=True
        )
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.02)  # Reduced wspace from 0.1 to 0.02

        # Create subplots
        ax_lie = fig.add_subplot(gs[0], projection='3d')
        ax_spd = fig.add_subplot(gs[1], projection='3d')

        # Initialize visualizers
        lie_viz = LieAlgebraVisualizer(ax_lie, self.config)
        spd_viz = SPDManifoldVisualizer(ax_spd, self.config)

        # Draw Lie Algebra visualization
        lie_viz.draw_plane()
        # Disable auto-scaling to prevent z-order issues
        ax_lie.set_xlim3d(-2, 1.8)
        ax_lie.set_ylim3d(-2, 2)
        ax_lie.set_zlim3d(-0.5, 1.5)
        p1, p2 = lie_viz.draw_rotated_points()
        lie_viz.draw_rotation_arrow(p1, p2)
        lie_viz.add_labels()
        lie_viz.finalize()

        # Draw SPD Manifold visualization
        spd_viz.draw_cone()
        # spd_viz.draw_invalid_region()  # Removed for cleaner ICML-style visualization
        s1, s2 = spd_viz.draw_ellipsoids()
        spd_viz.draw_transformation_arrow(s1, s2)
        spd_viz.add_labels()
        spd_viz.finalize()

        # Store global annotations
        self._add_matrix_exponential_annotation(fig)

        # Draw all annotations at the very end
        self._draw_global_annotations(fig)

        # Save figure
        self._save_figure(fig)

    def _add_matrix_exponential_annotation(self, fig) -> None:
        """Add matrix exponential annotation between plots"""
        # Store annotations to add at the very end
        self._global_annotations = {
            'arrows': [
                FancyArrowPatch(
                    (0.44, 0.55), (0.56, 0.55),
                    transform=fig.transFigure,
                    arrowstyle='-|>', mutation_scale=25,
                    color=self.config.COLORS['accent1'], lw=2
                )
            ],
            'texts': [
                (0.5, 0.58, r"Diffeomorphism: $\exp: \mathfrak{g} \to \mathcal{P}$",
                 13, self.config.COLORS['text'], 'bold'),
                (0.5, 0.47, r"$\Sigma = \exp(A)$  and  $A = \log(\Sigma)$",
                 14, self.config.COLORS['text'], 'bold')
            ]
        }

    def _draw_global_annotations(self, fig) -> None:
        """Draw all global annotations at the very end"""
        if hasattr(self, '_global_annotations'):
            # Draw arrows first
            if 'arrows' in self._global_annotations:
                for arrow in self._global_annotations['arrows']:
                    fig.add_artist(arrow)

            # Then draw texts
            if 'texts' in self._global_annotations:
                for x, y, text, fontsize, color, weight in self._global_annotations['texts']:
                    fig.text(x, y, text,
                            ha='center', fontsize=fontsize,
                            color=color, fontweight=weight)

    def _save_figure(self, fig) -> None:
        """Save figure in multiple formats"""
        base_name = "riemannian_diagram_icml_4k"
        formats = ['png', 'pdf']

        for fmt in formats:
            filename = f"{base_name}.{fmt}"
            fig.savefig(
                filename,
                dpi=self.config.DPI,
                bbox_inches='tight',
                format=fmt,
                transparent=True if fmt == 'png' else False
            )
            print(f"Generated: {filename}")


# === Entry Point ===
def generate_icml_figure_v3():
    """Main function to generate the figure"""
    figure_generator = RiemannianManifoldFigure()
    figure_generator.generate()


if __name__ == "__main__":
    # Optional: Allow command-line customization
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ICML Riemannian Manifold Visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config', type=str, default=None,
        help="Configuration file (not implemented)"
    )
    parser.add_argument(
        '--dpi', type=int, default=300,
        help="Output DPI"
    )
    parser.add_argument(
        '--formats', nargs='+', default=['png', 'pdf'],
        help="Output formats"
    )

    args = parser.parse_args()

    # Create custom config if needed
    if args.dpi != 300:
        config = PlotConfig()
        config.DPI = args.dpi
    else:
        config = None

    # Generate figure
    if config:
        generator = RiemannianManifoldFigure(config)
    else:
        generator = RiemannianManifoldFigure()

    # Override save formats if specified
    if args.formats != ['png', 'pdf']:
        generator._save_figure = lambda fig: None  # Disable default save
        fig = plt.figure(figsize=PlotConfig.FIGURE_SIZE)
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.02)  # Same reduced spacing
        ax_lie = fig.add_subplot(gs[0], projection='3d')
        ax_spd = fig.add_subplot(gs[1], projection='3d')

        lie_viz = LieAlgebraVisualizer(ax_lie, generator.config)
        spd_viz = SPDManifoldVisualizer(ax_spd, generator.config)

        lie_viz.draw_plane()
        p1, p2 = lie_viz.draw_rotated_points()
        lie_viz.draw_rotation_arrow(p1, p2)
        lie_viz.add_labels()
        lie_viz.finalize()

        spd_viz.draw_cone()
        # spd_viz.draw_invalid_region()  # Removed for cleaner ICML-style visualization
        s1, s2 = spd_viz.draw_ellipsoids()
        spd_viz.draw_transformation_arrow(s1, s2)
        spd_viz.add_labels()
        spd_viz.finalize()

        generator._add_matrix_exponential_annotation(fig)

        base_name = "riemannian_diagram_icml_4k"
        for fmt in args.formats:
            fig.savefig(
                f"{base_name}.{fmt}",
                dpi=args.dpi,
                bbox_inches='tight',
                format=fmt,
                transparent=True if fmt == 'png' else False
            )
            print(f"Generated: {base_name}.{fmt}")
    else:
        generator.generate()