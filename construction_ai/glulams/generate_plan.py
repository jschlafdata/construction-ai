"""
mpl-tools==0.4.1
numpy==2.3.4
matplotlib==3.10.7
"""

from typing import Dict, Any, Tuple, List, Sequence, Literal, Optional
from numpy.typing import NDArray

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import (
    Arc, 
    FancyArrowPatch, 
    Polygon, 
    Rectangle, 
    Wedge,
    FancyBboxPatch
)
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


_HAS_INSET = True ## from mpl_toolkits.axes_grid1.inset_locator import inset_axes (setting as requirement.)

# ------------------- #
# ------------------- #

import math

MEASURE_DENOM = 16  # 1/16"

def round_to_increment(x: float, denom: int = MEASURE_DENOM) -> float:
    """Round to nearest 1/denom inch."""
    return round(x * denom) / denom

def residual_to_increment(x: float, denom: int = MEASURE_DENOM) -> float:
    """How far x is from the nearest 1/denom increment (in ticks)."""
    return abs(x * denom - round(x * denom))

def format_inches_frac(x: float, denom: int = MEASURE_DENOM) -> str:
    """Format inches as a reduced fraction to 1/denom (e.g., 12 3/16")."""
    q = round_to_increment(x, denom)
    sgn = "-" if q < 0 else ""
    a = abs(q)
    whole = int(math.floor(a))
    num = int(round((a - whole) * denom))
    if num == denom:
        whole += 1
        num = 0
    if num == 0:
        return f'{sgn}{whole}"'
    g = math.gcd(num, denom)
    num //= g
    den = denom // g
    return f'{sgn}{whole} {num}/{den}"' if whole else f'{sgn}{num}/{den}"'


# ------------------- #
# ------------------- #

def round_to_sixteenth(value: float) -> float:
    """
    Round a measurement to the nearest 1/16 inch for practical construction.
    
    Args:
        value: Measurement in inches
        
    Returns:
        Value rounded to nearest 1/16"
        
    Examples:
        round_to_sixteenth(10.234) -> 10.25 (10 1/4")
        round_to_sixteenth(5.47) -> 5.5 (5 1/2")
        round_to_sixteenth(3.92) -> 3.9375 (3 15/16")
    """
    sixteenths = round(value * 16)
    return sixteenths / 16.0

def format_inches(value: float) -> str:
    return format_inches_frac(value, MEASURE_DENOM)

def find_buildable_parameters(
    arch_width: float,
    target_arch_height: float,
    target_kerf_spacing: float,
    target_support_spacing: float,
    num_boards: int = 2,
    board_thickness: float = 0.75
) -> Dict[str, float]:
    """
    Find adjusted parameters that produce buildable measurements (1/16" increments).
    
    This function adjusts the FLEXIBLE parameters (height, kerf spacing, support spacing)
    to values that naturally produce measurements in 1/16" increments throughout the design.
    
    FIXED (cannot change):
        - arch_width: The span
        - num_boards: Number of boards for caps
        - board_thickness: Thickness of each board
        - support_width: 2x4 width (3.5")
        - kerf_width: Saw blade width (0.125")
    
    FLEXIBLE (will be adjusted):
        - arch_height: Will be adjusted to nearest 1/16" that works well
        - kerf_spacing: Will be adjusted to produce integer number of kerfs
        - support_spacing: Will be adjusted to produce buildable support heights
    
    Args:
        arch_width: Fixed span in inches
        target_arch_height: Desired height (will be adjusted)
        target_kerf_spacing: Desired kerf spacing (will be adjusted)
        target_support_spacing: Desired support spacing (will be adjusted)
        num_boards: Number of boards for caps
        board_thickness: Thickness of each board
        
    Returns:
        Dictionary with adjusted parameters:
        {
            'arch_height': float,      # Adjusted to 1/16"
            'kerf_spacing': float,     # Adjusted for even distribution
            'support_spacing': float,  # Adjusted for even distribution
            'cap_thickness': float,    # Calculated from boards
            'adjustments_made': str    # Description of changes
        }
    """
    # Fixed parameters
    cap_thickness = num_boards * board_thickness
    
    # Step 1: Adjust arch height to nearest 1/16"
    # Try values near the target and pick one that produces good support heights
    height_candidates = [
        target_arch_height - 0.125,  # -1/8"
        target_arch_height - 0.0625, # -1/16"
        target_arch_height,
        target_arch_height + 0.0625, # +1/16"
        target_arch_height + 0.125,  # +1/8"
    ]
    
    best_height = target_arch_height
    best_score = float('inf')
    
    for h in height_candidates:
        h_rounded = round_to_sixteenth(h)
        if h_rounded <= 0:
            continue
        
        # Calculate radius for this height
        R = (arch_width**2 / 4 + h_rounded**2) / (2 * h_rounded)
        
        # Test how well support heights round at various positions
        score = 0
        for x_frac in [0.2, 0.4, 0.6, 0.8]:
            x = arch_width/2 * x_frac
            theta_max = np.arcsin(arch_width / (2 * R))
            if x < R * np.sin(theta_max):
                theta = np.arcsin(x / R)
                support_h = R - R * np.cos(theta)
                rounded_h = round_to_sixteenth(support_h)
                score += abs(support_h - rounded_h)
        
        if score < best_score:
            best_score = score
            best_height = h_rounded
    
    arch_height = best_height
    
    # Step 2: Adjust kerf spacing for even distribution
    # Calculate radius with adjusted height
    R = (arch_width**2 / 4 + arch_height**2) / (2 * arch_height)
    theta_max = np.arcsin(arch_width / (2 * R))
    arc_length = 2 * R * theta_max
    
    # Find number of kerfs that fits well
    num_kerfs_target = arc_length / target_kerf_spacing
    num_kerfs = max(3, round(num_kerfs_target))  # At least 3 kerfs
    
    # Adjust spacing to distribute evenly
    kerf_spacing = arc_length / num_kerfs
    kerf_spacing = round_to_sixteenth(kerf_spacing)
    
    # Step 3: Adjust support spacing for even distribution
    num_supports_target = arch_width / target_support_spacing
    num_supports = max(3, round(num_supports_target))  # At least 3 supports
    
    # Adjust spacing to distribute evenly across span
    support_spacing = arch_width / (num_supports - 1)  # -1 because includes both ends
    support_spacing = round_to_sixteenth(support_spacing)
    
    # Generate adjustment report
    adjustments = []
    if abs(arch_height - target_arch_height) > 0.001:
        adjustments.append(f"Height: {target_arch_height:.3f}\" → {arch_height:.4f}\"")
    if abs(kerf_spacing - target_kerf_spacing) > 0.001:
        adjustments.append(f"Kerf spacing: {target_kerf_spacing:.3f}\" → {kerf_spacing:.4f}\"")
    if abs(support_spacing - target_support_spacing) > 0.001:
        adjustments.append(f"Support spacing: {target_support_spacing:.3f}\" → {support_spacing:.4f}\"")
    
    adjustment_text = "; ".join(adjustments) if adjustments else "No adjustments needed"
    
    return {
        'arch_height': arch_height,
        'kerf_spacing': kerf_spacing,
        'support_spacing': support_spacing,
        'cap_thickness': cap_thickness,
        'num_kerfs': num_kerfs,
        'num_supports': num_supports,
        'adjustments_made': adjustment_text
    }


VISUAL_STYLES: Dict[str, Dict[str, Any]] = {
    "blueprint": {
        # Figure styling
        "fig_bg": "#0D47A1",
        "ax_bg": "#0D47A1",
        
        # Board/wood elements
        "board_face": "#BBDEFB",
        "board_edge": "#FFFFFF",
        "board_alpha": 0.8,
        
        # Kerf cuts
        "kerf_face": "#1565C0",
        "kerf_edge": "#FFFFFF",
        "kerf_alpha": 0.75,
        
        # Dimensions
        "dim_color": "#FFFFFF",
        "dim_linewidth": 1.5,
        "dim_alpha": 0.9,
        "dim_arrow_style": '<->',
        "dim_arrow_mutation": 15,
        "dim_leader_style": ':',
        "dim_leader_width": 1.0,
        
        # Text
        "dim_fontsize": 10,
        "dim_fontweight": 'bold',
        "label_fontsize": 11,
        "title_fontsize": 14,
        "title_color": "#FFFFFF",
        "info_fontsize": 9,
        
        # Text backgrounds - IMPROVED for no overlaps
        "text_bg": "#0D47A1",
        "text_bg_alpha": 0.95,  # More opaque so labels are clearly readable
        "text_box_style": 'round,pad=0.5',  # Slightly more padding
        "text_box_linewidth": 1.8,  # Thicker border for clarity
        
        # Grid
        "grid_color": "#1976D2",
        "grid_alpha": 0.4,
        "grid_linewidth": 0.5,
        "grid_style": '-',
        
        # Spines/borders
        "spine_color": "#FFFFFF",
        "spine_linewidth": 1.5,
        
        # Additional colors
        "arc_outer": "#BBDEFB",
        "arc_inner": "#90CAF9",
        "arc_alpha": 0.8,
        "chord_color": "#FFFFFF",
        "chord_style": "--",
        "chord_alpha": 0.7,
    },
}

# ============================================================================
# DEFAULT STYLE
# ============================================================================

DEFAULT_STYLE: str = "blueprint"

# ============================================================================
# STYLE ACCESSOR
# ============================================================================

class VisualStyle:
    """
    Centralized visual style manager.
    
    Usage:
        style = VisualStyle("blueprint")
        ax.set_facecolor(style.ax_bg)
        ax.plot(..., color=style.dim_color, linewidth=style.dim_linewidth)
    """
    
    def __init__(self, style_name: str = DEFAULT_STYLE):
        if style_name not in VISUAL_STYLES:
            raise ValueError(
                f"Unknown style '{style_name}'. Available: {list(VISUAL_STYLES.keys())}"
            )
        self._style = VISUAL_STYLES[style_name]
        self.name = style_name
    
    def __getattr__(self, key: str) -> Any:
        """Allow dot notation access to style properties"""
        if key.startswith('_'):
            return object.__getattribute__(self, key)
        if key in self._style:
            return self._style[key]
        raise AttributeError(f"Style property '{key}' not found in '{self.name}' theme")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style access with default"""
        return self._style.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access"""
        return self._style[key]
    
    def keys(self):
        """Get all available style keys"""
        return self._style.keys()
    
    def items(self):
        """Get all style items"""
        return self._style.items()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def add_kerf_thumbnail_inset(
    fig: Figure,
    ax: Axes,
    plan: Dict[str, Any],
    *,
    style: VisualStyle,
    title: str,
    loc: Literal["top", "bottom"],
    width_frac: float = 0.30,   # % of the main axes width
    height_frac: float = 0.10,  # % of the main axes height
    x_pad_frac: float = 0.03,   # gap from main axes right side
    y_shift_frac: float = 0.08, # vertical shift from center for stacking
):
    """
    Create a clean inset on the right for a compact kerf board thumbnail.
    """
    # Inset placement (axes-relative)
    if loc == "top":
        y_center = 0.72
    else:
        y_center = 0.28

    w = width_frac
    h = height_frac
    x0 = 1.0 + x_pad_frac - w
    y0 = y_center - h/2.0

    if _HAS_INSET:
        ax_in = inset_axes(ax, width=f"{w*100:.1f}%", height=f"{h*100:.1f}%",
                           bbox_to_anchor=(x0, y0, w, h), bbox_transform=ax.transAxes, borderpad=0)
    else:
        # fallback: figure-based coords near right; adjust if needed
        bbox = ax.get_position()
        left = bbox.x1 + x_pad_frac * (bbox.width)
        bottom = bbox.y0 + y0 * bbox.height
        ax_in = fig.add_axes([left, bottom, w * bbox.width, h * bbox.height])

    ax_in.set_xticks([]); ax_in.set_yticks([]); ax_in.set_frame_on(False)

    # Draw simplified board (re-using your palette)
    L   = float(plan["meta"]["board_length"])
    T   = float(plan["meta"]["board_thickness"])
    kw  = float(plan["meta"]["kerf_width"])
    xs  = np.asarray(plan["kerf_positions"], float)
    nd  = int(plan["num_kerfs"])
    kd  = float(plan["kerf_depth"])

    # Fit to inset axes
    pad = 0.02 * L
    ax_in.set_xlim(-L/2 - pad, L/2 + pad)
    ax_in.set_ylim(-0.35*T, 1.2*T)

    board = FancyBboxPatch(
        (-L/2, 0), L, T,
        boxstyle="round,pad=0.25",
        facecolor=style.board_face, edgecolor=style.board_edge,
        linewidth=1.2, alpha=style.board_alpha
    )
    ax_in.add_patch(board)

    # Downsample kerfs to keep it sparse
    step = max(1, int(np.ceil(nd / 12)))
    for i, x in enumerate(xs):
        if i % step: 
            continue
        ax_in.add_patch(
            Rectangle((x - kw/2.0, 0), kw, min(kd, 0.7*T),
                      facecolor=style.kerf_face, edgecolor=style.kerf_edge,
                      linewidth=1.0, alpha=style.kerf_alpha)
        )

    # One tidy length dimension only
    draw_horizontal_dimension(
        ax_in, x_left=-L/2, x_right=L/2, y_line=T,
        label=f"{title}: {format_inches(L)}",
        style=style, where="above", offset=6.0, leaders=True, label_offset_mult=1.2
    )

def get_style(style_name: str = DEFAULT_STYLE) -> VisualStyle:
    """
    Get a VisualStyle object for the specified theme.
    
    Args:
        style_name: "blueprint" (only option)
    
    Returns:
        VisualStyle object with all theme properties accessible via dot notation
    
    Example:
        style = get_style("blueprint")
        fig.patch.set_facecolor(style.fig_bg)
        ax.plot(..., color=style.dim_color)
    """
    return VisualStyle(style_name)

def apply_figure_style(fig, ax, style: VisualStyle):
    """
    Apply consistent styling to a matplotlib figure and axes.
    
    Args:
        fig: matplotlib Figure
        ax: matplotlib Axes
        style: VisualStyle object
    """
    # Figure background
    fig.patch.set_facecolor(style.fig_bg)
    
    # Axes background
    ax.set_facecolor(style.ax_bg)
    
    # Grid
    ax.grid(
        True,
        linewidth=style.grid_linewidth,
        alpha=style.grid_alpha,
        color=style.grid_color,
        linestyle=style.grid_style
    )
    ax.set_axisbelow(True)
    
    # Spines
    for spine in ax.spines.values():
        spine.set_edgecolor(style.spine_color)
        spine.set_linewidth(style.spine_linewidth)
    
    # Tick parameters
    ax.tick_params(colors=style.title_color, labelsize=style.info_fontsize)

def get_text_bbox_props(style: VisualStyle) -> Dict[str, Any]:
    """
    Get standard text bbox properties for the given style.
    
    Returns dict suitable for use as bbox parameter in ax.text()
    """
    return {
        'boxstyle': style.text_box_style,
        'facecolor': style.text_bg,
        'edgecolor': style.dim_color,
        'alpha': style.text_bg_alpha,
        'linewidth': style.text_box_linewidth,
    }

def list_available_styles() -> list:
    """Get list of all available style names"""
    return list(VISUAL_STYLES.keys())


def print_style_info(style_name: str = None):
    """Print information about a style or all styles"""
    if style_name:
        if style_name not in VISUAL_STYLES:
            print(f"Unknown style: {style_name}")
            print(f"Available: {list_available_styles()}")
            return
        styles_to_show = {style_name: VISUAL_STYLES[style_name]}
    else:
        styles_to_show = VISUAL_STYLES
    
    for name, props in styles_to_show.items():
        print(f"\n{'='*60}")
        print(f"Style: {name.upper()}")
        print('='*60)
        for key, value in sorted(props.items()):
            print(f"  {key:25s} : {value}")


def draw_horizontal_dimension(
    ax: Axes,
    *,
    x_left: float,
    x_right: float,
    y_line: float,
    label: str,
    style: VisualStyle,
    where: str = "above",
    offset: float = 10.0,  # INCREASED default
    leaders: bool = True,
    label_offset_mult: float = 1.5,  # INCREASED default
):
    """
    Draw horizontal dimension with arrows.
    Aggressive spacing prevents overlaps.
    """
    y_dim = y_line + (offset if where == "above" else -offset)
    
    # Main dimension line with arrows
    arrow = FancyArrowPatch(
        (x_left, y_dim), (x_right, y_dim),
        arrowstyle=style.dim_arrow_style,
        color=style.dim_color,
        linewidth=style.dim_linewidth,
        alpha=style.dim_alpha,
        mutation_scale=style.dim_arrow_mutation,
    )
    ax.add_patch(arrow)
    
    # Leader lines at both ends
    if leaders:
        leader_width = style.dim_linewidth * style.dim_leader_width
        ax.plot([x_left, x_left], [y_dim, y_line], 
                color=style.dim_color, 
                linewidth=leader_width, 
                linestyle=style.dim_leader_style, 
                alpha=style.dim_alpha * 0.9)
        ax.plot([x_right, x_right], [y_dim, y_line], 
                color=style.dim_color, 
                linewidth=leader_width, 
                linestyle=style.dim_leader_style, 
                alpha=style.dim_alpha * 0.9)
    
    # Label with aggressive offset to prevent overlap
    xm = 0.5 * (x_left + x_right)
    va_pos = "bottom" if where == "above" else "top"
    
    # INCREASED vertical offset for label
    label_y_offset = 1.5 * label_offset_mult  # DOUBLED from 0.8
    y_text = y_dim + label_y_offset if where == "above" else y_dim - label_y_offset
    
    ax.text(
        xm, y_text, label, 
        ha="center", 
        va=va_pos, 
        color=style.dim_color, 
        fontsize=style.dim_fontsize, 
        fontweight=style.dim_fontweight,
        bbox=get_text_bbox_props(style),
        zorder=100  # MUCH higher to always be on top
    )

def draw_vertical_dimension(
    ax: Axes,
    *,
    anchor: str,
    x_edge: float,
    x_target: float,
    y_bottom: float,
    y_top: float,
    label: str,
    style: VisualStyle,
    x_offset: float = 4.0,  # INCREASED default
    text_offset: float = 2.5,  # INCREASED default
    label_offset_mult: float = 1.5,  # INCREASED default
):
    """
    Draw vertical dimension with arrows.
    Aggressive spacing prevents overlaps.
    """
    x_dim = x_target - x_offset if anchor == "left" else x_target + x_offset
    y_mid = 0.5 * (y_bottom + y_top)
    y_text = y_mid + text_offset * label_offset_mult

    # Vertical dimension line with arrows
    arrow = FancyArrowPatch(
        (x_dim, y_bottom), (x_dim, y_top),
        arrowstyle=style.dim_arrow_style,
        color=style.dim_color,
        linewidth=style.dim_linewidth,
        alpha=style.dim_alpha,
        mutation_scale=style.dim_arrow_mutation,
    )
    ax.add_patch(arrow)

    # Horizontal pointer from edge
    pointer_width = style.dim_linewidth * style.dim_leader_width
    ax.plot([x_edge, x_dim], [y_mid, y_mid], 
            color=style.dim_color, 
            linewidth=pointer_width, 
            linestyle=style.dim_leader_style, 
            alpha=style.dim_alpha * 0.9)

    # Label with aggressive offset to prevent overlap
    ha = "left" if anchor == "left" else "right"
    
    # INCREASED horizontal offset for label
    label_x_offset = 1.0 * label_offset_mult  # DOUBLED from 0.5
    x_text = x_edge - label_x_offset if anchor == "left" else x_edge + label_x_offset
    
    ax.text(
        x_text, y_text, label, 
        ha=ha, 
        va="bottom", 
        color=style.dim_color, 
        fontsize=style.dim_fontsize, 
        fontweight=style.dim_fontweight,
        bbox=get_text_bbox_props(style),
        zorder=100  # MUCH higher to always be on top
    )

def draw_chord(ax: Axes, x_left: float, x_right: float, y: float, style: VisualStyle):
    """Draw a chord line using global style"""
    ax.plot(
        [x_left, x_right], [y, y], 
        linestyle=style.chord_style, 
        linewidth=style.dim_linewidth, 
        alpha=style.chord_alpha, 
        color=style.chord_color
    )

def draw_box(
    ax: Axes, 
    x_left: float, 
    x_right: float, 
    y_bottom: float, 
    y_top: float, 
    style: VisualStyle
):
    """Draw a rectangular box outline using global style"""
    ax.plot([x_left, x_right], [y_bottom, y_bottom], 
            linestyle=style.chord_style, linewidth=style.dim_linewidth, 
            alpha=style.chord_alpha, color=style.chord_color)
    ax.plot([x_left, x_right], [y_top, y_top], 
            linestyle=style.chord_style, linewidth=style.dim_linewidth, 
            alpha=style.chord_alpha, color=style.chord_color)
    ax.plot([x_left, x_left], [y_bottom, y_top], 
            linestyle=style.chord_style, linewidth=style.dim_linewidth, 
            alpha=style.chord_alpha, color=style.chord_color)
    ax.plot([x_right, x_right], [y_bottom, y_top], 
            linestyle=style.chord_style, linewidth=style.dim_linewidth, 
            alpha=style.chord_alpha, color=style.chord_color)

def draw_box_partial(
    ax: Axes,
    x_left: float,
    x_right: float,
    y_bottom: float,
    y_top: float,
    style: VisualStyle,
    *,
    show_bottom: bool,
    show_top: bool,
):
    """Draw a partial rectangular box outline using global style"""
    # Sides (always drawn)
    ax.plot([x_left, x_left], [y_bottom, y_top], 
            linestyle=style.chord_style, linewidth=style.dim_linewidth, 
            alpha=style.chord_alpha, color=style.chord_color)
    ax.plot([x_right, x_right], [y_bottom, y_top], 
            linestyle=style.chord_style, linewidth=style.dim_linewidth, 
            alpha=style.chord_alpha, color=style.chord_color)
    
    # Optional bottom/top
    if show_bottom:
        ax.plot([x_left, x_right], [y_bottom, y_bottom], 
                linestyle=style.chord_style, linewidth=style.dim_linewidth, 
                alpha=style.chord_alpha, color=style.chord_color)
    if show_top:
        ax.plot([x_left, x_right], [y_top, y_top], 
                linestyle=style.chord_style, linewidth=style.dim_linewidth, 
                alpha=style.chord_alpha, color=style.chord_color)

def add_info_box(
    ax: Axes,
    text: str,
    style: VisualStyle,
    *,
    position: tuple = (0.02, 0.98),
    ha: str = 'left',
    va: str = 'top',
):
    """
    Add an information text box to the plot using global style.
    """
    ax.text(
        position[0], position[1], text,
        transform=ax.transAxes,
        fontsize=style.info_fontsize,
        horizontalalignment=ha,
        verticalalignment=va,
        bbox=dict(
            boxstyle='round,pad=0.8', 
            facecolor=style.text_bg,
            edgecolor=style.dim_color,
            alpha=style.text_bg_alpha,
            linewidth=style.text_box_linewidth
        ),
        color=style.title_color,
        family='monospace',
        zorder=100  # High zorder so it's always on top
    )

SPACING_CONFIG = {
    "lamination_arcs": {
        "horizontal_offset": 8.0,      # Space between feature and dimension line
        "vertical_offset": 10.0,       # Space for vertical dimensions
        "label_offset_mult": 1.2,      # Extra space for labels
        "stagger_vertical": True,      # Stagger multiple vertical dims
        "vertical_step": 8.0,          # Step size for staggered dims
    },
    
    "cap_assembly": {
        "horizontal_offset": 9.0,
        "vertical_offset": 12.0,
        "label_offset_mult": 1.3,
        "stagger_vertical": True,
        "vertical_step": 10.0,
    },
    
    "kerf_layout": {
        "horizontal_offset_main": 10.0,  # Board length
        "horizontal_offset_detail": 7.0,  # Kerf width
        "horizontal_offset_spacing": 7.0, # Spacing between kerfs
        "vertical_offset": 12.0,
        "label_offset_mult": 1.2,
    },
}

def get_spacing(figure_type: str, key: str, default: float = 8.0) -> float:
    """Get spacing value for a specific figure type and key"""
    config = SPACING_CONFIG.get(figure_type, {})
    return config.get(key, default)

def calculate_circular_arch(
    span: float, 
    rise: float, 
    radius_offset: float = 0, 
    num_points: int = 100
) -> Tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """
    Calculate points for a circular arch.
    All units in inches.
    """
    outer_radius: float = (rise**2 + (span/2)**2) / (2 * rise)
    radius: float = outer_radius - radius_offset
    center_y: float = outer_radius - rise
    theta_max: float = np.arcsin((span/2) / outer_radius)
    theta: NDArray[np.float64] = np.linspace(-theta_max, theta_max, num_points)
    x: NDArray[np.float64] = radius * np.sin(theta)
    y: NDArray[np.float64] = radius * np.cos(theta) - center_y
    return x, y, outer_radius

def compute_arc_geometry(span: float, rise: float, thickness: float, num_points: int = 1000) -> Dict[str, Any]:
    """
    Compute full geometry for a circular arc 'board' (band) with thickness.
    Returns a dict with radii, center, arcs, band polygon, and both chords (outer & inner).
    All units are in inches. Angles in radians.
    """
    # --- circle geometry (outer) ---
    R_outer = (rise**2 + (span / 2.0)**2) / (2.0 * rise)
    center_y = R_outer - rise
    theta_max = np.arcsin((span / 2.0) / R_outer)

    # --- radii ---
    R_inner = R_outer - thickness
    if R_inner <= 0:
        raise ValueError("thickness too large for given span/rise; inner radius <= 0")

    # --- sample theta symmetrically ---
    theta = np.linspace(-theta_max, theta_max, num_points)

    # --- outer/inner arc points (data coords) ---
    x_outer = R_outer * np.sin(theta)
    y_outer = R_outer * np.cos(theta) - center_y

    x_inner = R_inner * np.sin(theta)
    y_inner = R_inner * np.cos(theta) - center_y

    # --- band polygon (outer forward, inner reversed) ---
    band_x = np.concatenate([x_outer, x_inner[::-1]])
    band_y = np.concatenate([y_outer, y_inner[::-1]])

    # --- chords (tip-to-tip for each arc) ---
    # outer tips at theta = ±theta_max
    xL_out, xR_out = -span / 2.0, span / 2.0
    y_out = 0.0
    width_out = xR_out - xL_out

    # inner tips at theta = ±theta_max for R_inner
    half_span_inner = R_inner * np.sin(theta_max)
    xL_in, xR_in = -half_span_inner, half_span_inner
    y_in = R_inner * np.cos(theta_max) - center_y
    width_in = xR_in - xL_in

    geom: Dict[str, Any] = {
        "params": {"span": span, "rise": rise, "thickness": thickness, "num_points": num_points},
        "circle": {"R_outer": R_outer, "R_inner": R_inner, "center_y": center_y, "theta_max": float(theta_max)},
        "arcs": {
            "outer": {"x": x_outer, "y": y_outer},
            "inner": {"x": x_inner, "y": y_inner},
        },
        "band": {"x": band_x, "y": band_y},
        "chords": {
            "outer": {"x_left": xL_out, "x_right": xR_out, "y": y_out, "width": width_out},
            "inner": {"x_left": xL_in, "x_right": xR_in, "y": y_in, "width": width_in},
        },
    }
    return geom

# ============================================================================
# ANNOTATION FUNCTIONS (using global style)
# ============================================================================

def _annotate_base_dims(ax: Axes, geom: Dict[str, Any], style: VisualStyle) -> None:
    """
    Add dimensions to the base laminated arch drawing using global style
    """
    span     = geom["params"]["span"]
    rise     = geom["params"]["rise"]
    pad_x    = max(24, 0.1 * span)
    x_left   = -span/2 - pad_x
    x_right  =  span/2 + pad_x

    R_outer   = geom["circle"]["R_outer"]
    R_inner   = geom["circle"]["R_inner"]
    center_y  = geom["circle"]["center_y"]
    theta_max = geom["circle"]["theta_max"]

    # crowns and tips (data y)
    y_crown_outer = R_outer - center_y
    y_crown_inner = R_inner - center_y
    y_tips_inner  = R_inner * np.cos(theta_max) - center_y

    # chords (tip-to-tip widths)
    ch_out = geom["chords"]["outer"]
    ch_in  = geom["chords"]["inner"]

    # Outer tip-to-tip
    draw_horizontal_dimension(
        ax,
        x_left=ch_out["x_left"], x_right=ch_out["x_right"], y_line=ch_out["y"],
        label=format_inches(ch_out["width"]), 
        style=style,
        where="below", offset=15.0, leaders=True, label_offset_mult=2.0,
    )

    # Inner tip-to-tip (MUCH FURTHER DOWN to avoid overlap)
    draw_horizontal_dimension(
        ax,
        x_left=ch_in["x_left"], x_right=ch_in["x_right"], y_line=ch_in["y"],
        label=format_inches(ch_in["width"]), 
        style=style,
        where="below", offset=30.0, leaders=True, label_offset_mult=2.0,
    )

    # Total height: baseline (0) → outer crown
    draw_vertical_dimension(
        ax,
        anchor="left", x_edge=x_left, x_target=0.0,
        y_bottom=0.0, y_top=y_crown_outer,
        label=format_inches(y_crown_outer - 0.0),
        style=style,
        x_offset=5.0, text_offset=3.0, label_offset_mult=2.0,
    )

    # Lam thickness at crown (STAGGERED LEFT to avoid overlap)
    draw_vertical_dimension(
        ax,
        anchor="left", x_edge=x_left - 12.0, x_target=-12.0,
        y_bottom=y_crown_inner, y_top=y_crown_outer,
        label=format_inches(y_crown_outer - y_crown_inner),
        style=style,
        x_offset=5.0, text_offset=3.0, label_offset_mult=2.0,
    )

# ============================================================================
# LAMINATION HELPER FUNCTIONS
# ============================================================================

def _muted_colors(n: int, alpha: float = 0.35, desat: float = 0.55):
    """Return n muted RGBA colors"""
    base = plt.get_cmap("tab10").colors
    cols = [base[i % len(base)] for i in range(n)]
    muted = []
    for r, g, b in cols:
        r_m = r * (1 - desat) + 1.0 * desat
        g_m = g * (1 - desat) + 1.0 * desat
        b_m = b * (1 - desat) + 1.0 * desat
        muted.append((r_m, g_m, b_m, alpha))
    return muted

def _darker(rgb: tuple[float, float, float, float] | tuple[float, float, float], k: float = 0.75):
    """Darken an RGB(A) color for borders"""
    if len(rgb) == 4:
        r, g, b, _ = rgb
    else:
        r, g, b = rgb
    return (r * k, g * k, b * k, 1.0)

def total_thickness_from_spec(lams: Sequence[Tuple[int, float]]) -> float:
    if not lams:
        raise ValueError("lams must be non-empty")
    tot = 0.0
    for count, t in lams:
        if count <= 0 or t <= 0:
            raise ValueError("each (count, thickness) must be positive")
        tot += count * t
    return tot

def expand_and_sort_spec(lams: Sequence[Tuple[int, float]]) -> List[float]:
    expanded: List[float] = []
    for count, t in lams:
        expanded.extend([t] * count)
    expanded.sort()
    return expanded

def lamination_rows_from_spec(geom: Dict[str, Any], lams: Sequence[Tuple[int, float]]) -> List[Dict[str, Any]]:
    span = geom["params"]["span"]
    R_inner = geom["circle"]["R_inner"]
    theta_max = geom["circle"]["theta_max"]
    center_y = geom["circle"]["center_y"]

    seq = expand_and_sort_spec(lams)
    rows: List[Dict[str, Any]] = []
    cur_inner = R_inner
    for i, t in enumerate(seq):
        R_in = cur_inner
        R_out = cur_inner + t
        R_mid = (R_in + R_out) * 0.5
        half_span_i = R_mid * np.sin(theta_max)
        chord_i = 2.0 * half_span_i
        arc_i = 2.0 * theta_max * R_mid
        y_tip_i = R_mid * np.cos(theta_max) - center_y
        rows.append({
            "index": i,
            "thickness": t,
            "R_in": R_in,
            "R_out": R_out,
            "R_mid": R_mid,
            "chord_length": chord_i,
            "arc_length": arc_i,
            "x_left": -half_span_i,
            "x_right": half_span_i,
            "y_tip": y_tip_i,
        })
        cur_inner = R_out
    return rows

def _draw_height_rail(
    ax: Axes,
    *,
    x_line: float,
    y0: float, y1: float, y2: float, y3: float,
    style: VisualStyle,
    rung_half: float = 10.0,       # half-length of the small horizontal rungs
    rung_lw: float = 2.0,
    rail_lw: float = 2.0,
    col_shift: float = 18.0,       # gap from rail to label column
    line_alpha: float = 0.85,
    labels: tuple[str, str, str, str, str] = (
        'Base', 'Bottom Cap', 'Base Rise', 'Top Cap', 'TOTAL'
    ),
):
    """
    Clean cumulative rail:
      rail from y0->y3 with rungs at y0, y1, y2, y3 and a stacked label column:
        Base 0", Bottom Cap Δy0→y1, Base Rise Δy1→y2, Top Cap Δy2→y3, TOTAL Δy0→y3
    """
    c = style.dim_color
    bg = style.text_bg

    # main rail
    ax.plot([x_line, x_line], [y0, y3],
            color=c, linewidth=rail_lw, alpha=line_alpha, zorder=12, clip_on=False)

    # rung helper
    def rung(y: float):
        ax.plot([x_line - rung_half, x_line + rung_half], [y, y],
                color=c, linewidth=rung_lw, alpha=line_alpha, zorder=13, clip_on=False)

    # rungs at each cumulative level
    for y in (y0, y1, y2, y3):
        rung(y)

    # text builder
    L0, L1, L2, L3, LT = labels
    base  = 0.0
    bcap  = (y1 - y0)
    rise  = (y2 - y1)
    tcap  = (y3 - y2)
    total = (y3 - y0)

    rows = [
        (y0, f'{format_inches(base)}  {L0}'),
        (y1, f'{format_inches(bcap)}  {L1}'),
        (y2, f'{format_inches(rise)}  {L2}'),
        (y3, f'{format_inches(tcap)}  {L3}'),
        (y3 + (y3 - y2) * 0.10, f'{format_inches(total)}  {LT}'),  # TOTAL just above top rung
    ]

    # single aligned column (left-justified), short leader from rail to text
    x_txt = x_line + col_shift
    for y, txt in rows:
        # leader
        ax.plot([x_line + rung_half, x_txt - 6.0], [y, y],
                color=c, linewidth=1.4, alpha=line_alpha, zorder=13, clip_on=False)
        # label
        ax.text(
            x_txt, y, txt,
            fontsize=11, color=c, fontweight='bold',
            va='center', ha='left', zorder=14, clip_on=False,
            bbox=dict(
                boxstyle='round,pad=0.45',
                facecolor=bg,
                edgecolor=c,
                alpha=0.95,
                linewidth=style.text_box_linewidth,
            ),
        )


def enrich_lamination_surface_lengths(
    lam_rows: List[Dict[str, Any]], 
    geom: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Add inner/outer/mid arc & chord lengths"""
    theta_max = geom["circle"]["theta_max"]
    out: List[Dict[str, Any]] = []
    for row in lam_rows:
        R_in  = row["R_in"]
        R_out = row["R_out"]
        R_mid = row["R_mid"]
        new_row = dict(row)
        new_row.update({
            "arc_length_min":   2.0 * theta_max * R_in,
            "arc_length_max":   2.0 * theta_max * R_out,
            "arc_length_mid":     2.0 * theta_max * R_mid,
            "chord_length_min": 2.0 * R_in  * np.sin(theta_max),
            "chord_length_max": 2.0 * R_out * np.sin(theta_max),
            "chord_length_mid":   2.0 * R_mid * np.sin(theta_max),
        })
        out.append(new_row)
    return out

def top_bottom(lam_rows):
    if not lam_rows:
        raise ValueError("no laminations")
    return {"bottom": lam_rows[0], "top": lam_rows[-1]}

# ============================================================================
# FIGURE 1: LAMINATION ARCS (with global style)
# ============================================================================

def plot_lamination_arcs_from_spec(
    span: float,
    rise: float,
    lams: Sequence[Tuple[int, float]],
    *,
    style_name: str = "modern",
    num_points: int = 1000,
    draw_band: bool = True,
    draw_midlines: bool = False,
    draw_strips: bool = True,
    fig_y_pad: int = 5,
    fig_x_pad: int = 3
) -> Tuple[Figure, Axes, Dict[str, Any]]:
    """
    FIGURE 1: Lamination arcs with strips showing each layer
    """
    style = get_style(style_name)
    
    total_thickness = total_thickness_from_spec(lams)
    geom = compute_arc_geometry(span, rise, total_thickness, num_points=num_points)

    lam_rows_base = lamination_rows_from_spec(geom, lams)
    lam_rows = enrich_lamination_surface_lengths(lam_rows_base, geom)

    fig, ax = plt.subplots(figsize=(15, 9))
    apply_figure_style(fig, ax, style)

    if draw_band:
        band_x, band_y = geom["band"]["x"], geom["band"]["y"]
        ax.add_patch(Polygon(np.c_[band_x, band_y], closed=True, linewidth=0.5,
                             fill=True, alpha=0.10, edgecolor='none'))

    if draw_strips:
        theta_max = geom["circle"]["theta_max"]
        center_y = geom["circle"]["center_y"]
        theta = np.linspace(-theta_max, theta_max, num_points)

        faces = _muted_colors(len(lam_rows), alpha=0.35, desat=0.55)
        for i, row in enumerate(lam_rows):
            R_in, R_out = row["R_in"], row["R_out"]
            x_out = R_out * np.sin(theta); y_out = R_out * np.cos(theta) - center_y
            x_in  = R_in  * np.sin(theta); y_in  = R_in  * np.cos(theta) - center_y
            strip_x = np.concatenate([x_out, x_in[::-1]])
            strip_y = np.concatenate([y_out, y_in[::-1]])

            face = faces[i]
            edge = _darker(face, k=0.70)
            ax.add_patch(Polygon(np.c_[strip_x, strip_y], closed=True,
                                 facecolor=face, edgecolor=edge, linewidth=0.8,
                                 joinstyle="miter", capstyle="butt", zorder=3))

    if draw_midlines:
        theta_max = geom["circle"]["theta_max"]
        center_y = geom["circle"]["center_y"]
        theta = np.linspace(-theta_max, theta_max, num_points)
        for row in lam_rows:
            R_mid = row["R_mid"]
            xi = R_mid * np.sin(theta)
            yi = R_mid * np.cos(theta) - center_y
            ax.plot(xi, yi, linewidth=0.8, alpha=0.9, color=style.chord_color)

    # Chords using global style
    ch_out, ch_in = geom["chords"]["outer"], geom["chords"]["inner"]
    draw_chord(ax, ch_out["x_left"], ch_out["x_right"], ch_out["y"], style)
    draw_chord(ax, ch_in["x_left"], ch_in["x_right"], ch_in["y"], style)

    # Viewport
    ax.set_aspect("equal", adjustable="box")
    pad = max(24, 0.1 * span)
    ax.set_xlim(-span/2 - pad - fig_x_pad, span/2 + pad + fig_x_pad)
    ymin = min(-0.15 * rise, ch_in["y"]) - total_thickness * 0.2
    ax.set_ylim(ymin-fig_y_pad, rise * 1.3 + total_thickness + fig_y_pad)
    
    # Labels with style
    ax.set_xlabel("inches", fontsize=style.label_fontsize, color=style.title_color, fontweight='bold')
    ax.set_ylabel("inches", fontsize=style.label_fontsize, color=style.title_color, fontweight='bold')
    ax.set_title("Laminated Arch Structure", fontsize=style.title_fontsize, 
                 color=style.title_color, fontweight='bold', pad=20)

    out: Dict[str, Any] = {
        "geom": geom,
        "laminations": lam_rows,
        "total_thickness": total_thickness,
        "lam_spec_sorted": expand_and_sort_spec(lams),
    }
    
    # Dimensions
    _annotate_base_dims(ax, geom, style)
    
    return fig, ax, out

# ============================================================================
# MATING CAP FUNCTIONS
# ============================================================================

def mating_arch_from_row(
    base_row: Dict[str, Any],
    geom: Dict[str, Any],
    thickness: float,
    clearance: float = 0.0,
    placement: Literal["top", "bottom"] = "top",
    recommend: Literal["auto", "min", "mid", "max"] = "auto",
) -> Dict[str, Any]:
    """Create a concentric mating arch"""
    if thickness <= 0:
        raise ValueError("thickness must be > 0")
    if clearance < 0:
        raise ValueError("clearance must be >= 0")

    theta_max = geom["circle"]["theta_max"]
    center_y  = geom["circle"]["center_y"]

    R_in_base  = base_row.get("R_in")
    R_out_base = base_row.get("R_out")
    if R_in_base is None or R_out_base is None:
        t_base = base_row.get("thickness")
        R_mid_base = base_row["R_mid"]
        if t_base is None:
            raise KeyError("base_row must include R_in/R_out or both R_mid and thickness")
        R_in_base  = R_mid_base - 0.5 * t_base
        R_out_base = R_mid_base + 0.5 * t_base

    if placement == "top":
        R_in  = R_out_base + clearance
        R_out = R_in + thickness
        fit_surface = "inner"
    elif placement == "bottom":
        R_out = R_in_base - clearance
        R_in  = R_out - thickness
        fit_surface = "outer"
    else:
        raise ValueError('placement must be "top" or "bottom"')

    if R_in <= 0 or R_out <= 0 or R_in >= R_out:
        raise ValueError("Invalid radii; check thickness/clearance relative to base row and geometry.")

    def half_span(R: float) -> float:
        return R * np.sin(theta_max)
    def y_tip(R: float) -> float:
        return R * np.cos(theta_max) - center_y

    xL_in,  xR_in  = -half_span(R_in),  half_span(R_in)
    xL_mid, xR_mid = -half_span((R_in + R_out) * 0.5), half_span((R_in + R_out) * 0.5)
    xL_out, xR_out = -half_span(R_out), half_span(R_out)

    widths = {
        "inner": xR_in - xL_in,
        "mid":   xR_mid - xL_mid,
        "outer": xR_out - xL_out,
    }

    arc_lengths = {
        "inner": 2.0 * theta_max * R_in,
        "mid":   2.0 * theta_max * ((R_in + R_out) * 0.5),
        "outer": 2.0 * theta_max * R_out,
    }

    chord_lengths = {
        "inner": 2.0 * half_span(R_in),
        "mid":   2.0 * half_span((R_in + R_out) * 0.5),
        "outer": 2.0 * half_span(R_out),
    }

    endpoints = {
        "inner": {"x_left": xL_in,  "x_right": xR_in,  "y_tip": y_tip(R_in)},
        "mid":   {"x_left": xL_mid, "x_right": xR_mid, "y_tip": y_tip((R_in + R_out) * 0.5)},
        "outer": {"x_left": xL_out, "x_right": xR_out, "y_tip": y_tip(R_out)},
    }

    if recommend == "auto":
        rec_key = "outer" if placement == "top" else "inner"
    else:
        rec_key = recommend

    key_map = {"min": "inner", "mid": "mid", "max": "outer"}
    rec_key = key_map.get(rec_key, rec_key)
    if rec_key not in arc_lengths:
        raise ValueError('recommend must be "auto", "min", "mid", "max", "inner", "outer"')

    return {
        "params": {
            "placement": placement,
            "thickness": thickness,
            "clearance": clearance,
            "fit_surface": fit_surface,
            "recommend_key": rec_key,
        },
        "circle": {
            "theta_max": float(theta_max),
            "center_y": center_y,
            "R_in": R_in,
            "R_mid": 0.5 * (R_in + R_out),
            "R_out": R_out,
        },
        "arc_lengths": arc_lengths,
        "chord_lengths": chord_lengths,
        "endpoints": endpoints,
        "widths": widths,
        "recommended": {
            "arc_length":   arc_lengths[rec_key],
            "chord_length": chord_lengths[rec_key],
            "surface":      rec_key,
        },
    }

# ============================================================================
# FIGURES 2 & 3: MATING CAP OVER BASE (with global style)
# ============================================================================

def plot_mating_cap_over_base(
    out: Dict[str, Any],
    *,
    placement: Literal["top", "bottom"],
    cap_thickness: float,
    clearance: float = 0.0,
    visual_gap: float = 0.0,
    style_name: str = "modern",
    num_points: int | None = None,
    draw_base: bool = True,
    draw_cap_chords: bool = True,
    draw_supports: bool = True,
    support_spacing: float = 12.0,
    support_width: float = 3.5,
    fig_y_pad: int = 5,
    fig_x_pad: int = 3
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    """
    FIGURES 2 & 3: Top and bottom cap views with support structures.
    
    Args:
        out: Output from plot_lamination_arcs_from_spec
        placement: "top" or "bottom" cap
        cap_thickness: Thickness of the cap (inches)
        clearance: Gap between cap and base (inches)
        visual_gap: Visual offset for cap display
        style_name: Visual style to use
        num_points: Number of points for arc plotting
        draw_base: Whether to draw the base lamination
        draw_cap_chords: Whether to draw cap dimension lines
        draw_supports: Whether to draw 2x4 support structures
        support_spacing: Spacing between support centers (inches)
        support_width: Width of 2x4 supports (inches, actual dimension)
        fig_y_pad: Y-axis padding
        fig_x_pad: X-axis padding
    
    Notes:
        - Support heights are calculated automatically from geometry:
          * Top cap: height from baseline (y=0) to cap surface at each position
          * Bottom cap: height from baseline (y=0) to cap surface at each position
        - Supports follow the natural curvature of the mold
    """
    style = get_style(style_name)
    
    geom = out["geom"]
    lam_rows = out["laminations"]
    total_thickness = out["total_thickness"]

    span = geom["params"]["span"]
    rise = geom["params"]["rise"]
    pad_x = max(24, 0.1 * span)

    base_row = lam_rows[-1] if placement == "top" else lam_rows[0]
    cap = mating_arch_from_row(
        base_row=base_row, geom=geom, thickness=cap_thickness,
        clearance=clearance, placement=placement, recommend="auto",
    )

    theta_max = geom["circle"]["theta_max"]
    center_y  = geom["circle"]["center_y"]
    if num_points is None:
        num_points = geom["params"]["num_points"]
    theta = np.linspace(-theta_max, theta_max, num_points)

    band_x, band_y = geom["band"]["x"], geom["band"]["y"]

    # Cap band polygon
    R_in_cap, R_out_cap = cap["circle"]["R_in"], cap["circle"]["R_out"]
    x_out_cap = R_out_cap * np.sin(theta); y_out_cap = R_out_cap * np.cos(theta) - center_y
    x_in_cap  = R_in_cap  * np.sin(theta); y_in_cap  = R_in_cap  * np.cos(theta) - center_y

    # Visual offset
    if visual_gap:
        y_out_cap = y_out_cap + visual_gap
        y_in_cap  = y_in_cap  + visual_gap

    cap_x = np.concatenate([x_out_cap, x_in_cap[::-1]])
    cap_y = np.concatenate([y_out_cap, y_in_cap[::-1]])

    # Plot
    fig, ax = plt.subplots(figsize=(15, 9))
    apply_figure_style(fig, ax, style)
    
    # Draw 2x4 supports FIRST (behind everything)
    support_heights_list = []
    if draw_supports:
        # Calculate support positions
        step = round_to_increment(max(support_spacing, 1e-6), MEASURE_DENOM)
        m = int(np.floor((span / 2.0) / step))
        support_positions = np.arange(-m, m + 1, dtype=float) * step
        
        # Calculate baseline and surface for each cap type
        if placement == "top":
            # Top cap: placed OVER the laminations (inverted, arch pointing down)
            # The OUTER surface sits on supports
            # Baseline is at the PEAK (center top) of the outer arc
            # Supports extend DOWN from peak to where surface is
            # CENTER: height = 0 (surface at peak)
            # EDGES: height = max (surface far from peak)
            
            R_support_surface = R_out_cap  # Outer surface
            half_span_outer = R_out_cap * np.sin(theta_max)
            
            # Baseline is at the PEAK of outer arc (center highest point)
            baseline_y = R_support_surface - center_y + visual_gap
            
        else:
            # Bottom cap (U-shaped): supports are INSIDE the cap
            # They extend from the INNER surface down to the BOTTOM chord line
            R_support_surface = R_in_cap  # Touch inner surface (inside the cap)
            half_span_inner = R_in_cap * np.sin(theta_max)
            
            # Baseline is at the bottom chord of the inner arc
            baseline_y = R_in_cap * np.cos(theta_max) - center_y + visual_gap
        
        for x_pos in support_positions:
            if placement == "top":
                # Top cap: supports within outer arc span
                if abs(x_pos) > half_span_outer:
                    continue  # Skip supports outside the arc
                
                # Top cap: determine which edge of the support touches the surface
                # Left side of arc: use LEFT edge of support (outer edge)
                # Right side of arc: use RIGHT edge of support (outer edge)
                # Center support: use center (x_pos as-is)
                if x_pos < 0:  # Left side
                    x_contact = x_pos - support_width/2  # Left edge of support (outer)
                elif x_pos > 0:  # Right side
                    x_contact = x_pos + support_width/2  # Right edge of support (outer)
                else:  # Center
                    x_contact = x_pos
                
                # Calculate y-position of OUTER surface at the contact point
                theta_pos = np.arcsin(np.clip(x_contact / R_support_surface, -1, 1))
                y_outer_surface = R_support_surface * np.cos(theta_pos) - center_y + visual_gap
                
                # Support height is distance from peak (baseline) down to surface
                support_height = baseline_y - y_outer_surface
                
                # Draw support FROM surface UP to baseline (peak)
                # Rectangle bottom corner is at y_outer_surface
                # Rectangle extends upward by support_height
                if support_height > 0:
                    support_heights_list.append(support_height)
                    support_rect = Rectangle(
                        (x_pos - support_width/2, y_outer_surface),
                        support_width,
                        support_height,
                        facecolor=style.kerf_face,
                        edgecolor=style.kerf_edge,
                        linewidth=1.5,
                        alpha=0.6,
                        zorder=1
                    )
                    ax.add_patch(support_rect)
                    
                    # Add height label at top of support
                    label_y = y_outer_surface + support_height + 0.5
                    ax.text(x_pos, label_y, format_inches(support_height),
                           ha='center', va='bottom',
                           fontsize=8, color=style.dim_color,
                           fontweight='bold', alpha=0.9)
                
            else:
                # Bottom cap: support must be within inner surface span
                if abs(x_pos) > half_span_inner:
                    continue  # Skip supports outside the arc
                
                # Bottom cap: determine which edge of the support touches the surface
                # Left side of arc: use RIGHT edge of support (inner edge, closer to center)
                # Right side of arc: use LEFT edge of support (inner edge, closer to center)
                # Center support: use center (x_pos as-is)
                if x_pos < 0:  # Left side
                    x_contact = x_pos + support_width/2  # Right edge of support (inner)
                elif x_pos > 0:  # Right side
                    x_contact = x_pos - support_width/2  # Left edge of support (inner)
                else:  # Center
                    x_contact = x_pos
                
                # Calculate y-position on the cap inner surface at the contact point
                theta_pos = np.arcsin(np.clip(x_contact / R_support_surface, -1, 1))
                y_surface = R_support_surface * np.cos(theta_pos) - center_y + visual_gap
                
                # Support extends from baseline up to surface
                support_height = y_surface - baseline_y
                
                # Draw support FROM baseline UP to surface
                if support_height > 0:
                    support_heights_list.append(support_height)
                    support_rect = Rectangle(
                        (x_pos - support_width/2, baseline_y),
                        support_width,
                        support_height,
                        facecolor=style.kerf_face,
                        edgecolor=style.kerf_edge,
                        linewidth=1.5,
                        alpha=0.6,
                        zorder=1
                    )
                    ax.add_patch(support_rect)
                    
                    # Add height label at BOTTOM of support (more readable for bottom cap)
                    label_y = baseline_y - 0.5
                    ax.text(x_pos, label_y, format_inches(support_height),
                           ha='center', va='top',
                           fontsize=8, color=style.dim_color,
                           fontweight='bold', alpha=0.9)
    
    # Use style colors for base and cap
    if draw_base:
        ax.add_patch(Polygon(np.c_[band_x, band_y], closed=True,
                             facecolor=style.board_face, 
                             edgecolor=style.board_edge, 
                             linewidth=0.8,
                             alpha=style.board_alpha * 0.5,
                             zorder=2))
    
    # Cap uses arc colors from style
    ax.add_patch(Polygon(np.c_[cap_x, cap_y], closed=True,
                         facecolor=style.arc_outer, 
                         edgecolor=style.arc_inner, 
                         linewidth=1.0,
                         alpha=style.arc_alpha,
                         zorder=3))

    # Cap chords and dimensions
    if draw_cap_chords:
        half_span = lambda R: R * np.sin(theta_max)
        y_tip     = lambda R: R * np.cos(theta_max) - center_y + visual_gap
        y_peak    = lambda R: R - center_y + visual_gap
        x_left_edge  = -span/2 - pad_x
        x_right_edge =  span/2 + pad_x

        if placement == "top":
            xL_out, xR_out = -half_span(R_out_cap), half_span(R_out_cap)
            y_bottom = y_tip(R_out_cap)
            y_top    = y_peak(R_out_cap)
            
            draw_box_partial(
                ax, xL_out, xR_out, y_bottom, y_top, style,
                show_bottom=False, show_top=True
            )

            outer_chord_len = 2.0 * half_span(R_out_cap)
            draw_horizontal_dimension(
                ax,
                x_left=xL_out, x_right=xR_out, y_line=y_top,
                label=format_inches(outer_chord_len),
                style=style,
                where="above", offset=15.0, leaders=True, label_offset_mult=2.0
            )

            draw_vertical_dimension(
                ax, anchor="left", x_edge=x_left_edge, x_target=xL_out,
                y_bottom=y_bottom, y_top=y_top,
                label=format_inches(y_top - y_bottom),
                style=style,
                x_offset=6.0, text_offset=4.0, label_offset_mult=2.0,
            )

        elif placement == "bottom":
            xL_in, xR_in = -half_span(R_in_cap), half_span(R_in_cap)
            y_in = y_tip(R_in_cap)
            draw_chord(ax, xL_in, xR_in, y_in, style)

            y_top_inner = y_peak(R_in_cap)
            ax.plot([0.0, 0.0], [y_in, y_top_inner], 
                   linestyle=style.chord_style, 
                   linewidth=style.dim_linewidth, 
                   alpha=style.chord_alpha, 
                   color=style.chord_color)

            inner_chord_len = 2.0 * half_span(R_in_cap)
            draw_horizontal_dimension(
                ax,
                x_left=xL_in, x_right=xR_in, y_line=y_in,
                label=format_inches(inner_chord_len),
                style=style,
                where="below", offset=15.0, leaders=True, label_offset_mult=2.0
            )

            draw_vertical_dimension(
                ax, anchor="right", x_edge=x_right_edge, x_target=0.0,
                y_bottom=y_in, y_top=y_top_inner,
                label=format_inches(y_top_inner - y_in),
                style=style,
                x_offset=6.0, text_offset=4.0, label_offset_mult=2.0,
            )

    # Collect Y data (including supports if drawn)
    ys = [cap_y]
    if draw_base:
        ys.append(band_y)
    if draw_supports:
        # Extend viewport to include baseline
        ys.append(np.array([0.0]))

    y_min = min(np.min(arr) for arr in ys)
    y_max = max(np.max(arr) for arr in ys)

    # Viewport
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-span/2 - pad_x - fig_x_pad, span/2 + pad_x + fig_x_pad)
    ax.set_ylim(y_min - fig_y_pad, y_max + fig_y_pad + 10)
    
    # Labels
    cap_title = f"{'Top' if placement == 'top' else 'Bottom'} Cap Assembly"
    if draw_supports:
        cap_title += f" with 2×4 Supports"
    ax.set_xlabel("inches", fontsize=style.label_fontsize, color=style.title_color, fontweight='bold')
    ax.set_ylabel("inches", fontsize=style.label_fontsize, color=style.title_color, fontweight='bold')
    ax.set_title(cap_title, fontsize=style.title_fontsize, 
                 color=style.title_color, fontweight='bold', pad=20)

    return fig, ax, {"cap": cap, "base": out, "supports": {
        "enabled": draw_supports,
        "spacing": support_spacing,
        "width": support_width,
        "heights": support_heights_list if draw_supports else [],
        "num_supports": len(support_heights_list) if draw_supports else 0,
    }}


def draw_kerf_board_thumbnail(
    ax: Axes,
    plan: Dict[str, Any],
    *,
    origin: Tuple[float, float],
    width_plot_units: float,
    style: VisualStyle,
    label: str,
    max_slots: int = 15,
):
    """
    Draw a simplified, to-scale-ish kerf board thumbnail (unbent) near the combined figure.
    - Scales horizontally to `width_plot_units` in plot coordinates.
    - Vertical scale is modest to stay compact.
    - Shows every Nth slot if there are many.
    """
    L   = float(plan["meta"]["board_length"])
    T   = float(plan["meta"]["board_thickness"])
    kw  = float(plan["meta"]["kerf_width"])
    xs  = np.asarray(plan["kerf_positions"], float)
    nd  = int(plan["num_kerfs"])

    # Compute uniform scale to fit requested width
    sx = width_plot_units / L
    sy = max(min(sx * 0.25, 0.15), 0.05)  # thin visual, readable but compact

    ox, oy = origin
    # Outer board rect
    ax.add_patch(
        FancyBboxPatch(
            (ox, oy), L * sx, T * sy,
            boxstyle="round,pad=0.25",
            facecolor=style.board_face,
            edgecolor=style.board_edge,
            linewidth=1.2,
            alpha=style.board_alpha,
            zorder=6,
        )
    )

    # Draw a subset of slots to avoid clutter
    step = max(1, int(np.ceil(nd / max_slots)))
    kd = float(plan["kerf_depth"])  # draw a short notch
    for i, x in enumerate(xs):
        if i % step != 0:
            continue
        kx0 = ox + (x - kw/2.0) * sx
        kwp = kw * sx
        kh  = min(kd * sy, T * sy * 0.7)
        ax.add_patch(
            Rectangle(
                (kx0, oy), kwp, kh,
                facecolor=style.kerf_face,
                edgecolor=style.kerf_edge,
                linewidth=1.0,
                alpha=style.kerf_alpha,
                zorder=7,
            )
        )

    # Underline showing full board length with label
    cx0, cx1 = ox, ox + L * sx
    yline = oy - (T * sy * 0.50)
    ax.plot([cx0, cx1], [yline, yline],
            color=style.chord_color, linestyle=style.chord_style,
            linewidth=style.dim_linewidth, alpha=0.9, zorder=6)
    draw_horizontal_dimension(
        ax,
        x_left=cx0, x_right=cx1, y_line=yline,
        label=f"{label}: {format_inches(L)}",
        style=style, where="below", offset=5.0, leaders=True, label_offset_mult=1.2,
    )

def draw_height_brackets(
    ax: Axes,
    *,
    x_line: float,
    y0: float, y1: float, y2: float, y3: float,
    style: VisualStyle,
    gap: float = 8.0,
):
    """
    Draw three labeled height segments to the RIGHT of the main cumulative line:
      [y0→y1] Bottom Cap Thickness
      [y1→y2] Base Rise
      [y2→y3] Top Cap Thickness
    Uses small horizontal offsets to avoid overlap.
    """
    tick = 5.0
    off1, off2, off3 = 10.0, 16.0, 22.0  # stagger labels

    def _seg(y_a, y_b, text, off):
        # bracket
        ax.plot([x_line, x_line], [y_a, y_b],
                color=style.dim_color, linewidth=1.8, alpha=0.95, zorder=12)
        ax.plot([x_line - tick, x_line + tick], [y_a, y_a],
                color=style.dim_color, linewidth=1.8, alpha=0.95, zorder=12)
        ax.plot([x_line - tick, x_line + tick], [y_b, y_b],
                color=style.dim_color, linewidth=1.8, alpha=0.95, zorder=12)
        # label
        cy = 0.5 * (y_a + y_b)
        ax.text(
            x_line + off, cy,
            f"{format_inches(y_b - y_a)}  {text}",
            fontsize=11, color=style.dim_color, fontweight="bold",
            va="center", ha="left", alpha=1.0,
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor=style.text_bg,
                edgecolor=style.dim_color,
                alpha=0.95,
                linewidth=style.text_box_linewidth,
            ),
            zorder=13,
        )

    _seg(y0, y1, "Bottom Cap Thickness", off1)
    _seg(y1, y2, "Base Rise",            off2)
    _seg(y2, y3, "Top Cap Thickness",    off3)

def draw_height_brackets_clean(
    ax: Axes,
    *,
    x_line: float,
    y0: float, y1: float, y2: float, y3: float,
    style: VisualStyle,
    gap: float = 8.0,
    tick: float = 5.0,
    label_offsets: tuple[float, float, float] = (12.0, 18.0, 24.0),
    label_names: tuple[str, str, str] = ("Bottom Cap Thickness", "Base Rise", "Top Cap Thickness"),
):
    """
    Draw three labeled height segments on the RIGHT of the main cumulative line:
      [y0→y1] -> label_names[0]
      [y1→y2] -> label_names[1]
      [y2→y3] -> label_names[2]

    - Uses staggered horizontal offsets so labels don't collide
    - Adds short leaders from each label to the segment midpoint
    - Uses a halo (path effect) so text stays readable
    """
    lw = 1.9
    col = style.dim_color
    bg = style.text_bg

    def _segment(y_a, y_b, text, off):
        # main bracket spine
        ax.plot([x_line, x_line], [y_a, y_b],
                color=col, linewidth=lw, alpha=0.95, zorder=12, clip_on=False)

        # end ticks
        ax.plot([x_line - tick, x_line + tick], [y_a, y_a],
                color=col, linewidth=lw, alpha=0.95, zorder=12, clip_on=False)
        ax.plot([x_line - tick, x_line + tick], [y_b, y_b],
                color=col, linewidth=lw, alpha=0.95, zorder=12, clip_on=False)

        # label at midpoint with leader
        y_mid = 0.5 * (y_a + y_b)
        x_lab = x_line + off
        ax.annotate(
            "", xy=(x_line + tick, y_mid), xytext=(x_lab - 4.0, y_mid),
            arrowprops=dict(arrowstyle="-", linestyle="-", color=col, linewidth=1.4, alpha=0.9),
            zorder=13, clip_on=False
        )

        txt = f"{format_inches(y_b - y_a)}  {text}"
        t = ax.text(
            x_lab, y_mid, txt,
            fontsize=11, color=col, fontweight="bold",
            va="center", ha="left", alpha=1.0, zorder=14, clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor=bg,
                edgecolor=col,
                alpha=0.95,
                linewidth=style.text_box_linewidth,
            ),
        )
        # halo so the text reads cleanly over geometry
        t.set_path_effects([pe.withStroke(linewidth=2.0, foreground=bg)])
    
    off1, off2, off3 = label_offsets
    n1, n2, n3 = label_names
    _segment(y0, y1, n1, off1)
    _segment(y1, y2, n2, off2)
    _segment(y2, y3, n3, off3)

def plot_both_caps_combined(
    out: Dict[str, Any],
    *,
    cap_thickness: float,
    clearance: float = 0.0,
    style_name: str = "modern",
    draw_supports: bool = True,
    support_spacing: float = 12.0,
    support_width: float = 3.5,
    vertical_spacing: float = 0.0,  # vertical gap between assemblies
    fig_y_pad: int = 5,
    fig_x_pad: int = 3,
    show_kerf_thumbnails: bool = True,
    kerf_thumb_width: float = 28.0,
    plan_top: Optional[Dict[str, Any]] = None,
    plan_bot: Optional[Dict[str, Any]] = None,
    show_left_labels: bool = False,      # optional left-side tags
    peak_x: float = 0.0,                 # x of the crown/peak to point at
    right_margin: float = 18.0,          # gap from arch bbox to first number
    num_col_gap: float = 16.0,           # extra gap beyond end of dotted line
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    """
    Combined view with caps/chords/supports and minimalist right-side height readout.
    For each referenced height (y1, y2, y3) draw ONE dotted leader from crown (x=peak_x)
    to the right, and place the numeric value just to the right of that line.
    Also show TOTAL at y3 as a second dotted leader with a number above.
    """

    # -----------------------
    # Local helpers (scoped)
    # -----------------------
    def _round_to_sixteenth(x: float) -> float:
        try:
            return round_to_increment(x, 16)  # if your global helper exists
        except Exception:
            return round(x * 16.0) / 16.0

    def _number_tag(ax: Axes, x: float, y: float, text: str, style: VisualStyle):
        ax.text(
            x, y, text,
            fontsize=11, color=style.dim_color, fontweight='bold',
            va='center', ha='left', zorder=14, clip_on=False,
            bbox=dict(
                boxstyle='round,pad=0.45',
                facecolor=style.text_bg,
                edgecolor=style.dim_color,
                alpha=0.95,
                linewidth=style.text_box_linewidth,
            ),
        )

    def _dotted_to_right(ax: Axes, y: float, x_start: float, x_end: float, style: VisualStyle):
        ax.plot([x_start, x_end], [y, y],
                color=style.dim_color, linewidth=1.6, alpha=0.9, linestyle=":",
                zorder=11, clip_on=False)

    # -----------------------
    # Begin main function
    # -----------------------
    style = get_style(style_name)

    geom = out["geom"]
    lam_rows = out["laminations"]
    span = geom["params"]["span"]
    rise = geom["params"]["rise"]
    pad_x = max(24, 0.1 * span)

    # Build caps
    top_cap = mating_arch_from_row(lam_rows[-1], geom, cap_thickness, clearance=clearance, placement="top", recommend="auto")
    bottom_cap = mating_arch_from_row(lam_rows[0],  geom, cap_thickness, clearance=clearance, placement="bottom", recommend="auto")

    theta_max = geom["circle"]["theta_max"]
    center_y  = geom["circle"]["center_y"]
    theta     = np.linspace(-theta_max, theta_max, geom["params"]["num_points"])

    band_x = geom["band"]["x"]
    band_y = geom["band"]["y"]

    # Figure
    fig, ax = plt.subplots(figsize=(20, 24))
    apply_figure_style(fig, ax, style)

    # Colors
    if style_name == "blueprint":
        bottom_cap_color = '#4a90e2'
        base_color       = '#95a5a6'
        top_cap_color    = '#00bcd4'
        bottom_label_color = '#4a90e2'
        base_label_color   = '#ecf0f1'
        top_label_color    = '#00bcd4'
    else:
        bottom_cap_color = '#3498db'
        base_color       = '#95a5a6'
        top_cap_color    = '#e74c3c'
        bottom_label_color = '#3498db'
        base_label_color   = '#333333'
        top_label_color    = '#e74c3c'

    # Vertical offsets
    bottom_cap_offset = 0.0
    base_offset       = vertical_spacing
    top_cap_offset    = 2 * vertical_spacing

    # Base band
    base_y_offset = band_y + base_offset
    ax.add_patch(Polygon(np.c_[band_x, base_y_offset], closed=True,
                         facecolor=base_color, edgecolor=style.board_edge,
                         linewidth=0.8, alpha=style.board_alpha, zorder=2))

    # Lamination rows
    for row in lam_rows:
        x_in = row["R_in"] * np.sin(theta);  y_in = row["R_in"] * np.cos(theta) - center_y + base_offset
        x_out= row["R_out"]* np.sin(theta);  y_out= row["R_out"]* np.cos(theta) - center_y + base_offset
        if row.get("color_index", 0) % 2 == 0:
            ax.plot(x_in,  y_in,  color=style.arc_inner, linewidth=0.5, alpha=0.6, zorder=2)
        else:
            ax.plot(x_out, y_out, color=style.arc_outer, linewidth=0.5, alpha=0.6, zorder=2)

    # Draw both caps with supports and chord dimensions
    for placement, cap_data, y_offset, label_text, cap_color, label_color in [
        ("bottom", bottom_cap, bottom_cap_offset, "BOTTOM CAP", bottom_cap_color, bottom_label_color),
        ("top",    top_cap,    top_cap_offset,    "TOP CAP",    top_cap_color,    top_label_color),
    ]:
        R_in_cap  = cap_data["circle"]["R_in"]
        R_out_cap = cap_data["circle"]["R_out"]

        # Cap polygon
        x_out = R_out_cap * np.sin(theta); y_out = R_out_cap * np.cos(theta) - center_y + y_offset
        x_in  = R_in_cap  * np.sin(theta); y_in  = R_in_cap  * np.cos(theta)  - center_y + y_offset
        cap_x = np.concatenate([x_out, x_in[::-1]])
        cap_y = np.concatenate([y_out, y_in[::-1]])
        ax.add_patch(Polygon(np.c_[cap_x, cap_y], closed=True,
                             facecolor=cap_color, edgecolor=style.arc_inner,
                             linewidth=1.0, alpha=0.7, zorder=3))

        # Cap chord dimensions
        half_span_cap = lambda R: R * np.sin(theta_max)
        y_tip_cap  = lambda R, off: R * np.cos(theta_max) - center_y + off
        y_peak_cap = lambda R, off: R - center_y + off

        if placement == "top":
            xL_out, xR_out = -half_span_cap(R_out_cap), half_span_cap(R_out_cap)
            y_cap_top = y_peak_cap(R_out_cap, y_offset)
            draw_horizontal_dimension(
                ax, x_left=xL_out, x_right=xR_out, y_line=y_cap_top,
                label=format_inches(2.0 * half_span_cap(R_out_cap)),
                style=style, where="above", offset=8.0, leaders=True, label_offset_mult=1.5
            )
        else:
            xL_in, xR_in = -half_span_cap(R_in_cap), half_span_cap(R_in_cap)
            y_cap_bottom = y_tip_cap(R_in_cap, y_offset)
            draw_horizontal_dimension(
                ax, x_left=xL_in, x_right=xR_in, y_line=y_cap_bottom,
                label=format_inches(2.0 * half_span_cap(R_in_cap)),
                style=style, where="below", offset=8.0, leaders=True, label_offset_mult=1.5
            )

        # Supports (rounded to 1/16")
        if draw_supports:
            num_supports = int(span / support_spacing) + 1
            support_positions = np.linspace(-span/2, span/2, num_supports)

            if placement == "top":
                R_support_surface = R_out_cap
                half_span_outer = R_out_cap * np.sin(theta_max)
                baseline_y = R_support_surface - center_y + y_offset

                for x_pos in support_positions:
                    if abs(x_pos) > half_span_outer:
                        continue
                    x_contact = x_pos - support_width/2 if x_pos < 0 else (x_pos + support_width/2 if x_pos > 0 else x_pos)
                    theta_pos = np.arcsin(np.clip(x_contact / R_support_surface, -1, 1))
                    y_surface = R_support_surface * np.cos(theta_pos) - center_y + y_offset
                    support_height = _round_to_sixteenth(baseline_y - y_surface)
                    if support_height > 0:
                        ax.add_patch(Rectangle(
                            (x_pos - support_width/2, y_surface),
                            support_width, support_height,
                            facecolor=style.kerf_face, edgecolor=style.kerf_edge,
                            linewidth=1.5, alpha=0.6, zorder=1
                        ))
                        ax.text(x_pos, y_surface + support_height + 0.5,
                                format_inches(support_height),
                                ha='center', va='bottom', fontsize=9,
                                color=style.dim_color, fontweight='bold', alpha=0.95)
            else:
                R_support_surface = R_in_cap
                half_span_inner = R_in_cap * np.sin(theta_max)
                baseline_y = R_in_cap * np.cos(theta_max) - center_y + y_offset

                for x_pos in support_positions:
                    if abs(x_pos) > half_span_inner:
                        continue
                    x_contact = x_pos + support_width/2 if x_pos < 0 else (x_pos - support_width/2 if x_pos > 0 else x_pos)
                    theta_pos = np.arcsin(np.clip(x_contact / R_support_surface, -1, 1))
                    y_surface = R_support_surface * np.cos(theta_pos) - center_y + y_offset
                    support_height = _round_to_sixteenth(y_surface - baseline_y)
                    if support_height > 0:
                        ax.add_patch(Rectangle(
                            (x_pos - support_width/2, baseline_y),
                            support_width, support_height,
                            facecolor=style.kerf_face, edgecolor=style.kerf_edge,
                            linewidth=1.5, alpha=0.6, zorder=1
                        ))
                        ax.text(x_pos, baseline_y - 0.5,
                                format_inches(support_height),
                                ha='center', va='top', fontsize=9,
                                color=style.dim_color, fontweight='bold', alpha=0.95)

        # Optional left labels
        if show_left_labels:
            label_x = -span/2 - pad_x + 8
            label_y = y_offset + rise/2
            ax.text(label_x, label_y, label_text,
                    fontsize=13, color=label_color, fontweight='bold',
                    rotation=90, va='center', ha='center', alpha=1.0)

    # Optional base label on left
    if show_left_labels:
        ax.text(-span/2 - pad_x + 8, base_offset + rise/2, "BASE\nLAMINATION",
                fontsize=13, color=base_label_color, fontweight='bold',
                rotation=90, va='center', ha='center', alpha=1.0)

    # Dimension helpers
    half_span = lambda R: R * np.sin(theta_max)
    y_tip  = lambda R, off: R * np.cos(theta_max) - center_y + off
    y_peak = lambda R, off: R - center_y + off

    # Base span
    R_base = lam_rows[-1]["R_out"]
    xL_base, xR_base = -half_span(R_base), half_span(R_base)
    y_base_bottom = y_tip(R_base, base_offset)
    draw_horizontal_dimension(
        ax, x_left=xL_base, x_right=xR_base, y_line=y_base_bottom,
        label=f"Base span: {format_inches(span)}",
        style=style, where="below", offset=10.0, leaders=True, label_offset_mult=1.6
    )

    # Heights (peaks)
    R_in_bottom = bottom_cap["circle"]["R_in"]
    R_out_top   = top_cap["circle"]["R_out"]
    y0 = y_tip(R_in_bottom, bottom_cap_offset)   # absolute bottom
    y1 = y_peak(R_in_bottom, bottom_cap_offset)  # bottom-cap crown
    y2 = y_peak(R_base,      base_offset)        # base crown
    y3 = y_peak(R_out_top,   top_cap_offset)     # top-cap crown

    # Right-side placement for numbers (to the right of arch)
    x_numbers = span/2 + pad_x + right_margin
    x_text    = x_numbers + num_col_gap

    # Values
    val_bottom_cap = y1 - y0
    val_base_rise  = y2 - y1
    val_top_cap    = y3 - y2
    val_total      = y3 - y0

    # Draw three simple dotted lines (to peaks) + numbers to the RIGHT
    _dotted_to_right(ax, y1, peak_x, x_numbers, style); _number_tag(ax, x_text, y1, format_inches(val_bottom_cap), style)
    _dotted_to_right(ax, y2, peak_x, x_numbers, style); _number_tag(ax, x_text, y2, format_inches(val_base_rise),  style)
    _dotted_to_right(ax, y3, peak_x, x_numbers, style); _number_tag(ax, x_text, y3, format_inches(val_top_cap),   style)
    # TOTAL: second number at y3, offset slightly above to avoid collision
    _dotted_to_right(ax, y3, peak_x, x_numbers, style); _number_tag(ax, x_text, y3 + max(6.0, 0.06*(y3-y0)), format_inches(val_total), style)

    # Kerf thumbnails (to the far right)
    if show_kerf_thumbnails:
        thumb_x = x_text + kerf_thumb_width + 40.0
        if plan_top is not None:
            draw_kerf_board_thumbnail(
                ax,
                plan_top,
                origin=(thumb_x, (y2 + y3) * 0.5 - 4.0),
                width_plot_units=kerf_thumb_width,
                style=style,
                label="Top Cap Board",
                max_slots=15,
            )
        if plan_bot is not None:
            draw_kerf_board_thumbnail(
                ax,
                plan_bot,
                origin=(thumb_x, (y0 + y1) * 0.5 - 4.0),
                width_plot_units=kerf_thumb_width,
                style=style,
                label="Bottom Cap Board",
                max_slots=15,
            )
        ax.set_xlim(-span/2 - pad_x - fig_x_pad, thumb_x + kerf_thumb_width + 30.0)
    else:
        ax.set_xlim(-span/2 - pad_x - fig_x_pad, span/2 + pad_x + fig_x_pad + 25)

    # Viewport
    y_min = y0 - fig_y_pad - 10
    y_max = y3 + fig_y_pad + 10
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylim(y_min, y_max)

    # Labels
    ax.set_xlabel("inches", fontsize=style.label_fontsize, color=style.title_color, fontweight='bold')
    ax.set_ylabel("inches", fontsize=style.label_fontsize, color=style.title_color, fontweight='bold')
    ax.set_title("Complete Mold Assembly: Base + Top & Bottom Caps with Supports",
                 fontsize=style.title_fontsize, color=style.title_color, fontweight='bold', pad=20)

    # Return bundle
    real_total_height = rise + (2 * cap_thickness)
    return fig, ax, {
        "top_cap": top_cap,
        "bottom_cap": bottom_cap,
        "base": out,
        "total_height": real_total_height,
        "base_rise": rise,
        "cap_thickness": cap_thickness,
        "visual_spacing": vertical_spacing
    }

def calculate_kerf_cuts(
    board_length: float,
    board_thickness: float,
    bend_radius: float,
    kerf_spacing: float,
    kerf_width: float = 0.125,
) -> Tuple[int, float, NDArray[np.float64], NDArray[np.float64]]:
    """Calculate kerf cut parameters"""
    if board_length <= 0 or board_thickness <= 0 or bend_radius <= 0:
        raise ValueError("board_length, board_thickness, and bend_radius must be positive")
    if kerf_spacing <= kerf_width:
        raise ValueError("kerf_spacing must be > kerf_width")
    
    actual_spacing = kerf_spacing
    available_length = board_length - 2 * actual_spacing
    num_kerfs = max(0, int(available_length / actual_spacing) + 1)
    
    kerf_positions: NDArray[np.float64] = np.linspace(
        -board_length/2 + actual_spacing, 
        board_length/2 - actual_spacing, 
        num_kerfs
    ) if num_kerfs > 0 else np.array([], dtype=float)
    
    kerf_depth: float = board_thickness * (1 - bend_radius / (bend_radius + actual_spacing))
    kerf_depths: NDArray[np.float64] = np.full(num_kerfs, kerf_depth)
    
    return num_kerfs, kerf_depth, kerf_positions, kerf_depths

def kerf_plan_from_cap(
    cap: Dict[str, Any], 
    *, 
    kerf_spacing: float, 
    kerf_width: float = 0.125,
    num_boards: int = 1,
    board_thickness: float | None = None
) -> Dict[str, Any]:
    """
    Create kerf plan from cap results.
    
    Args:
        cap: Cap calculation results from mating_arch_from_row()
        kerf_spacing: Spacing between kerf cuts (inches)
        kerf_width: Width of each kerf cut (inches)
        num_boards: Number of boards to laminate together (default: 1)
        board_thickness: Thickness per board (inches). If None, uses cap thickness / num_boards
    
    Examples:
        # Single 1.5" board
        plan = kerf_plan_from_cap(cap, kerf_spacing=12.0)
        
        # Two 0.75" boards laminated
        plan = kerf_plan_from_cap(cap, kerf_spacing=12.0, num_boards=2, board_thickness=0.75)
        
        # Three 0.5" boards laminated  
        plan = kerf_plan_from_cap(cap, kerf_spacing=12.0, num_boards=3, board_thickness=0.5)
    """
    board_length = float(cap["recommended"]["arc_length"])
    cap_thickness = float(cap["params"]["thickness"])
    bend_radius = float(cap["circle"]["R_in"])
    
    # Calculate actual board thickness to use
    if board_thickness is None:
        # Divide cap thickness evenly among boards
        actual_board_thickness = cap_thickness / num_boards
    else:
        # Use specified thickness
        actual_board_thickness = board_thickness
        # Verify it makes sense
        total_thickness = actual_board_thickness * num_boards
        if abs(total_thickness - cap_thickness) > 0.01:
            import warnings
            warnings.warn(
                f"Board thickness mismatch: {num_boards} × {actual_board_thickness}\" = "
                f"{total_thickness}\" but cap thickness is {cap_thickness}\""
            )

    num_kerfs, kerf_depth, kerf_positions, kerf_depths = calculate_kerf_cuts(
        board_length=board_length,
        board_thickness=actual_board_thickness,
        bend_radius=bend_radius,
        kerf_spacing=kerf_spacing,
        kerf_width=kerf_width,
    )

    return {
        "num_kerfs": num_kerfs,
        "kerf_depth": kerf_depth,
        "kerf_positions": kerf_positions,
        "kerf_depths": kerf_depths,
        "meta": {
            "placement": cap["params"]["placement"],
            "board_length": board_length,
            "board_thickness": actual_board_thickness,
            "num_boards": num_boards,
            "total_cap_thickness": cap_thickness,
            "bend_radius": bend_radius,
            "kerf_spacing": kerf_spacing,
            "kerf_width": kerf_width,
        },
    }

# ============================================================================
# FIGURES 4 & 5: KERF BOARD LAYOUTS (with global style)
# ============================================================================

def plot_kerf_board_layout(
    plan: Dict[str, Any],
    *,
    style_name: str = "modern",
    label_all: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    FIGURES 4 & 5: Kerf cut layouts for top and bottom caps
    """
    style = get_style(style_name)
    
    L   = float(plan["meta"]["board_length"])
    T   = float(plan["meta"]["board_thickness"])
    kw  = float(plan["meta"]["kerf_width"])
    ks  = float(plan["meta"]["kerf_spacing"])
    kd  = float(plan["kerf_depth"])
    xs  = np.asarray(plan["kerf_positions"], float)
    nd  = int(plan["num_kerfs"])

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    apply_figure_style(fig, ax, style)

    # Board rectangle
    x0, x1 = -L/2, L/2
    board_rect = FancyBboxPatch(
        (x0, 0), L, T,
        boxstyle="round,pad=0.3",
        facecolor=style.board_face,
        edgecolor=style.board_edge,
        linewidth=2.0,
        alpha=style.board_alpha
    )
    ax.add_patch(board_rect)

    # Wood grain for modern style
    if style_name == "modern":
        for i in range(5):
            y_grain = T * (0.2 + 0.15 * i)
            ax.plot([x0, x1], [y_grain, y_grain], 
                   color=style.board_edge, alpha=0.1, linewidth=0.5)

    # Kerf slots
    for i, x in enumerate(xs):
        kx0, kx1 = x - kw/2.0, x + kw/2.0
        
        kerf_rect = Rectangle(
            (kx0, 0), kw, kd,
            facecolor=style.kerf_face,
            edgecolor=style.kerf_edge,
            linewidth=1.5,
            alpha=style.kerf_alpha
        )
        ax.add_patch(kerf_rect)
        
        # Smart numbering
        if nd > 10 and i % 5 == 0 and not label_all:
            ax.text(x, -0.5, f"#{i+1}", ha='center', va='top', 
                   fontsize=7, color=style.dim_color, alpha=0.6)

    # Dimensions
    pad_x = 0.06 * L
    x_left_edge  = x0 - pad_x
    x_right_edge = x1 + pad_x

    # Overall board length (HIGHEST - way above everything)
    draw_horizontal_dimension(
        ax,
        x_left=x0, x_right=x1, y_line=T,
        label=f"Board Length: {format_inches(L)}",
        style=style,
        where="above", offset=25.0, leaders=True, label_offset_mult=2.0
    )

    # Representative kerf
    rep_idx = int(np.argmin(np.abs(xs))) if nd > 0 else None
    if rep_idx is not None:
        x_rep = xs[rep_idx]
        
        # Kerf depth (LEFT side)
        draw_vertical_dimension(
            ax,
            anchor="left", x_edge=x_left_edge, x_target=x_rep,
            y_bottom=0.0, y_top=kd,
            label=f"Depth: {format_inches(kd)}",
            style=style,
            x_offset=6.0, text_offset=4.0, label_offset_mult=2.0
        )
        
        # Kerf width (MIDDLE height - above board but below board length)
        draw_horizontal_dimension(
            ax,
            x_left=x_rep - kw/2.0, x_right=x_rep + kw/2.0, y_line=kd,
            label=f"Width: {format_inches(kw)}",
            style=style,
            where="above", offset=15.0, leaders=True, label_offset_mult=2.0
        )

    # Spacing between center kerfs (LOWEST - inside the board area)
    if nd >= 2:
        xs_sorted = np.sort(xs)
        left_idx  = np.where(xs_sorted < 0)[0]
        right_idx = np.where(xs_sorted > 0)[0]
        
        if len(left_idx) and len(right_idx):
            xl = xs_sorted[left_idx[-1]]
            xr = xs_sorted[right_idx[0]]
        else:
            xl, xr = xs_sorted[0], xs_sorted[1]
        
        x_left_edge_of_right  = xr - kw/2.0
        x_right_edge_of_left  = xl + kw/2.0
        spacing_edge_to_edge  = x_left_edge_of_right - x_right_edge_of_left
        
        # Place LOWER in the board to avoid kerf width label
        y_spacing_line = T * 0.25
        draw_horizontal_dimension(
            ax,
            x_left=x_right_edge_of_left, x_right=x_left_edge_of_right, y_line=y_spacing_line,
            label=f"Spacing: {format_inches(spacing_edge_to_edge)}",
            style=style,
            where="above", offset=6.0, leaders=True, label_offset_mult=2.0
        )

    # Info box
    num_boards = plan["meta"].get("num_boards", 1)
    total_cap_thickness = plan["meta"].get("total_cap_thickness", T)
    
    if num_boards > 1:
        info_text = (
            f"Total Kerfs: {nd}\n"
            f"Boards: {num_boards} × {format_inches(T)} = {format_inches(total_cap_thickness)}\n"
            f"Bend Radius: {format_inches(plan['meta']['bend_radius'])}"
        )
    else:
        info_text = (
            f"Total Kerfs: {nd}\n"
            f"Board Thickness: {format_inches(T)}\n"
            f"Bend Radius: {format_inches(plan['meta']['bend_radius'])}"
        )
    add_info_box(ax, info_text, style)

    # Viewport - INCREASED to fit all labels
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_left_edge - 12.0, x_right_edge + 12.0)
    ax.set_ylim(-18.0, T + 60.0)
    
    # Labels
    placement = plan["meta"]["placement"].title()
    ax.set_xlabel("Distance (inches)", fontsize=style.label_fontsize, 
                  color=style.title_color, fontweight='bold')
    ax.set_ylabel("Height (inches)", fontsize=style.label_fontsize, 
                  color=style.title_color, fontweight='bold')
    ax.set_title(f"Kerf Cut Layout — {placement} Cap (Unbent Board)", 
                 fontsize=style.title_fontsize, color=style.title_color, 
                 fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig, ax

def fit_friendly_parameters(
    *,
    span: float,
    rise_guess: float,
    lam_spec: Sequence[Tuple[int, float]],
    cap_thickness: float,
    kerf_spacing_guess: float,
    support_spacing_guess: float,
    kerf_width: float = 0.125,
    board_thickness: float = 0.75,
    denom: int = MEASURE_DENOM,
    # Search windows (inches)
    rise_window: float = 0.50,           # ±0.50" around guess
    kerf_window: float = 1.00,           # ±1.00" around guess
    support_window: float = 2.00,        # ±2.00" around guess
) -> Dict[str, float]:
    """
    Coarse grid local search around your guesses.
    Objective: minimize fractional 'residual to 1/16' on key cut/mark dimensions.
    """
    # Precompute fixed total thickness
    total_thk = total_thickness_from_spec(lam_spec)

    # Step equals 1/denom
    step = 1.0 / float(denom)

    # Grids
    rises = np.arange(rise_guess - rise_window, rise_guess + rise_window + 1e-9, step)
    kerfs = np.arange(kerf_spacing_guess - kerf_window, kerf_spacing_guess + kerf_window + 1e-9, step)
    supps = np.arange(support_spacing_guess - support_window, support_spacing_guess + support_window + 1e-9, step)

    best = None
    best_cost = float("inf")

    for rise in rises:
        if rise <= 0:
            continue
        # Base geometry for this rise
        geom = compute_arc_geometry(span, rise, total_thk, num_points=600)
        lam_rows = enrich_lamination_surface_lengths(lamination_rows_from_spec(geom, lam_spec), geom)
        # Top uses top row; bottom uses bottom row
        cap_top = mating_arch_from_row(lam_rows[-1], geom, cap_thickness, placement="top")
        cap_bot = mating_arch_from_row(lam_rows[0],  geom, cap_thickness, placement="bottom")

        # Key outputs to land on grid
        L_top = float(cap_top["recommended"]["arc_length"])   # board length for top cap
        L_bot = float(cap_bot["recommended"]["arc_length"])   # board length for bottom cap
        chord_top = float(cap_top["chord_lengths"]["outer"])  # top cap outer chord
        chord_bot = float(cap_bot["chord_lengths"]["inner"])  # bottom cap inner chord

        for ks in kerfs:
            if ks <= kerf_width + step:
                continue
            # Depth check (use bottom-cap inner radius—worst case)
            R_in_bot = float(cap_bot["circle"]["R_in"])
            kd = board_thickness * (1.0 - R_in_bot / (R_in_bot + ks))
            if kd >= board_thickness * 0.97:  # too deep to be practical
                continue

            for ss in supps:
                if ss <= step:
                    continue

                # Residuals to 1/16 for important things
                R = lambda x: residual_to_increment(x, denom)

                # Weights: emphasize board lengths & chords; keep others gentle
                cost = (
                    4.0 * R(L_top) ** 2
                    + 4.0 * R(L_bot) ** 2
                    + 2.0 * R(chord_top) ** 2
                    + 2.0 * R(chord_bot) ** 2
                    + 1.0 * R(kd) ** 2
                    # tiny bias to stay near guesses
                    + 0.05 * (rise - rise_guess) ** 2
                    + 0.01 * (ks   - kerf_spacing_guess) ** 2
                    + 0.01 * (ss   - support_spacing_guess) ** 2
                )

                if cost < best_cost:
                    best_cost = cost
                    best = {"arch_height": float(rise), "kerf_spacing": float(ks), "support_spacing": float(ss)}

    # Fallback to guesses if nothing better found
    if best is None:
        best = {
            "arch_height": float(rise_guess),
            "kerf_spacing": round_to_increment(kerf_spacing_guess, denom),
            "support_spacing": round_to_increment(support_spacing_guess, denom),
        }
    else:
        # Snap the fitted values to 1/16 as the final step
        best = {k: round_to_increment(v, denom) for k, v in best.items()}

    return best



if __name__ == '__main__':
    # Existing hard constraints
    num_boards = 2
    board_thickness = 0.625
    cap_thickness = (num_boards * board_thickness)
    arch_width  = 108.0   # NOT FLEXIBLE
    arch_height = 28.0    # FLEXIBLE
    kerf_spacing = 12.0   # FLEXIBLE
    support_width = 3.5   # fixed (2x4)
    support_spacing = 12.0  # FLEXIBLE
    kerf_width = 0.125    # fixed (saw blade)

    # >>> NEW: fit the flexible parameters to 1/16-in grid <<<
    fit = fit_friendly_parameters(
        span=arch_width,
        rise_guess=arch_height,
        lam_spec=[(4, 0.75), (4, 0.5)],
        cap_thickness=cap_thickness,
        kerf_spacing_guess=kerf_spacing,
        support_spacing_guess=support_spacing,
        kerf_width=kerf_width,
        board_thickness=board_thickness,
    )

    arch_height   = fit["arch_height"]
    kerf_spacing  = fit["kerf_spacing"]
    support_spacing = fit["support_spacing"]
    print("Fitted:", fit)  # optional

    # Now proceed as before — all labels/locations will be 1/16"-friendly.
    fig1, ax1, out = plot_lamination_arcs_from_spec(
        arch_width, arch_height, [(4, 0.75), (6, 0.5)],
        style_name="blueprint"
    )

    fig2, ax2, out_top = plot_mating_cap_over_base(
        out, 
        placement="top", 
        cap_thickness=cap_thickness,
        style_name="blueprint",
        draw_supports=True,
        support_spacing=support_spacing, 
        support_width=support_width
    )

    fig3, ax3, out_bot = plot_mating_cap_over_base(
        out, 
        placement="bottom", 
        cap_thickness=cap_thickness,
        style_name="blueprint",
        draw_supports=True,
        support_spacing=support_spacing,
        support_width=support_width
    )

    plan_top = kerf_plan_from_cap(
        out_top["cap"], kerf_spacing=kerf_spacing,
        num_boards=num_boards, board_thickness=board_thickness
    )
    plan_bot = kerf_plan_from_cap(
        out_bot["cap"], kerf_spacing=kerf_spacing,
        num_boards=num_boards, board_thickness=board_thickness
    )

    fig_combined, ax_combined, out_combined = plot_both_caps_combined(
        out,
        cap_thickness=cap_thickness,
        style_name="blueprint",
        draw_supports=True,
        support_spacing=support_spacing,
        support_width=support_width,
        show_kerf_thumbnails=False,
        kerf_thumb_width=28.0,
        plan_top=plan_top,
        plan_bot=plan_bot,
    )


    fig4, ax4 = plot_kerf_board_layout(plan_top, style_name="blueprint")
    fig5, ax5 = plot_kerf_board_layout(plan_bot, style_name="blueprint")