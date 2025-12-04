# ========================================================================
# VISUALIZATION CONFIGURATION MODULE
# ========================================================================
"""
Centralized styling and output configuration for mission analysis plots.

Configuration domains:
    - Directory management: Timestamped output folders
    - Color schemes: Mission phases, performance metrics, UI elements
    - Typography: Font families, sizes, weights
    - Layout: Dimensions, spacing, margins
    - Line styles: Widths, dash patterns
    - Export settings: Format, resolution, output paths

Provides consistent publication-quality styling across all visualization modules.
"""

from typing import Dict, Any
from datetime import datetime
from pathlib import Path

# ========================================================================
# SECTION 1: OUTPUT DIRECTORY MANAGEMENT
# ========================================================================

# Global run directory: Singleton for timestamped output folder
_CURRENT_RUN_DIR = None


def get_or_create_run_directory(base_dir: str = None, phase: str = None) -> str:
    """
    Access or create timestamped output directory with phase subfolders.
    
    Directory structure: Images/YYYY-MM-DD_HH-MM-SS/Phase/
    Singleton pattern: One timestamped folder per execution session.
    
    Parameters:
        base_dir: str - base path (default: workspace/Images)
        phase: str - mission phase {'Climb', 'Cruise', 'Descent', 'Optimized', 'CG'}
                     If None, returns base run directory
    
    Returns:
        str: path to phase subdirectory or base run directory
    """
    global _CURRENT_RUN_DIR
    
    # Singleton initialization
    if _CURRENT_RUN_DIR is None:
        # Base directory resolution
        if base_dir is None:
            workspace_root = Path(__file__).parent
            base_dir = workspace_root / "Images"
        else:
            base_dir = Path(base_dir)
        
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamped run directory: YYYY-MM-DD_HH-MM-SS
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = base_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        
        _CURRENT_RUN_DIR = str(run_dir)
        print(f"[EXPORT] Output directory: {run_dir}")
    
    # Phase subdirectory creation
    if phase:
        phase_dir = Path(_CURRENT_RUN_DIR) / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        return str(phase_dir)
    
    return _CURRENT_RUN_DIR


# ========================================================================
# SECTION 2: COLOR PALETTE
# ========================================================================

class Colors:
    """Standardized color scheme for mission analysis visualization."""
    
    # Mission phase identification
    CLIMB = 'royalblue'          # Climb phase trajectories
    CRUISE = 'green'             # Cruise phase trajectories
    DESCENT = 'crimson'          # Descent phase trajectories
    
    # Performance variable encoding
    THRUST = 'blue'              # T [N]: thrust
    DRAG = 'red'                 # D [N]: drag
    FUEL_FLOW = 'darkgreen'      # ṁ [kg/s]: fuel flow rate
    MASS = 'purple'              # m [kg]: aircraft mass
    LEVER = 'orange'             # δ [-]: throttle lever
    AIRSPEED = 'teal'            # V [m/s]: true airspeed
    FUEL_CONSUMED = 'crimson'    # Δm_fuel [kg]: fuel consumed
    
    # UI and reference elements
    GRID = 'lightgray'           # Grid lines
    BACKGROUND = 'rgba(240, 240, 245, 0.5)'  # Plot background
    TRANSITION_LINE = 'gray'     # Phase transitions
    WARNING = 'orange'           # Warning indicators
    CRITICAL = 'red'             # Critical limits
    
    # Envelope and optimization
    ENVELOPE_LIMIT = 'crimson'   # Flight envelope boundaries
    OPTIMAL_PATH = 'yellow'      # Optimal climb trajectory
    OPTIMAL_PATH_DESCENT = 'darkviolet'  # Optimal descent trajectory
    
    # Cost function visualization
    J_VALUES_CLIMB = 'darkviolet'    # J [kg/m]: climb cost density
    J_VALUES_DESCENT = 'darkorange'  # J [kg/m]: descent cost density


# ========================================================================
# SECTION 3: TYPOGRAPHY
# ========================================================================

class Typography:
    """Font configuration for text elements."""
    
    FONT_FAMILY = 'Arial, sans-serif'  # Primary font family
    
    # Title font sizes [pt]
    MAIN_TITLE_SIZE = 18        # Main figure title
    SUBPLOT_TITLE_SIZE = 14     # Subplot titles
    
    # Text element sizes [pt]
    AXIS_LABEL_SIZE = 14        # Axis labels (x, y)
    AXIS_TICK_SIZE = 12         # Tick mark labels
    LEGEND_SIZE = 11            # Legend entries
    ANNOTATION_SIZE = 10        # Annotations and notes
    HOVER_SIZE = 11             # Hover tooltip text
    
    # Font styling
    TITLE_WEIGHT = 'bold'       # Title font weight


# ========================================================================
# SECTION 4: LAYOUT GEOMETRY
# ========================================================================

class Layout:
    """Figure dimensions and spacing parameters."""
    
    # Figure dimensions [px]
    STANDARD_WIDTH = 1200       # Standard plot width
    STANDARD_HEIGHT = 1000      # Standard plot height
    DASHBOARD_WIDTH = 1200      # Dashboard width
    DASHBOARD_HEIGHT = 1600     # Dashboard height (multi-panel)
    WIDE_WIDTH = 1400           # Wide format width
    TALL_HEIGHT = 1200          # Tall format height
    
    # Subplot spacing [normalized 0-1]
    VERTICAL_SPACING = 0.12     # Vertical gap between subplots
    HORIZONTAL_SPACING = 0.12   # Horizontal gap between subplots
    
    # Margins [px]
    MARGIN_LEFT = 80            # Left margin
    MARGIN_RIGHT = 150          # Right margin (legend space)
    MARGIN_TOP = 100            # Top margin (title space)
    MARGIN_BOTTOM = 80          # Bottom margin


# ========================================================================
# SECTION 5: LINE STYLING
# ========================================================================

class LineStyles:
    """Line width and pattern parameters."""
    
    # Line widths [px]
    THICK = 3                   # Thick lines (primary data)
    MEDIUM = 2                  # Medium lines (secondary data)
    THIN = 1                    # Thin lines (reference/grid)
    
    # Line patterns
    SOLID = 'solid'             # Solid line
    DASH = 'dash'               # Dashed line
    DOT = 'dot'                 # Dotted line
    DASHDOT = 'dashdot'         # Dash-dot line


# ========================================================================
# SECTION 6: PLOT CONFIGURATION TEMPLATES
# ========================================================================

def get_standard_layout(title: str, subtitle: str = "", height: int = None, width: int = None) -> Dict[str, Any]:
    """
    Generate standard Plotly layout configuration.
    
    Configuration includes: title formatting, dimensions, template, fonts,
    background colors, and hover behavior.
    
    Parameters:
        title: str - main title text
        subtitle: str - subtitle with metrics (optional)
        height: int - figure height [px] (default STANDARD_HEIGHT)
        width: int - figure width [px] (default STANDARD_WIDTH)
    
    Returns:
        dict: Plotly layout configuration
    """
    if height is None:
        height = Layout.STANDARD_HEIGHT
    if width is None:
        width = Layout.STANDARD_WIDTH
    
    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += f"<br><sup>{subtitle}</sup>"
    
    return {
        'title': {
            'text': title_text,
            'x': 0.5,
            'xanchor': 'center',
            'font': {
                'size': Typography.MAIN_TITLE_SIZE,
                'family': Typography.FONT_FAMILY
            }
        },
        'height': height,
        'width': width,
        'template': 'plotly_white',
        'font': {
            'family': Typography.FONT_FAMILY,
            'size': Typography.HOVER_SIZE
        },
        'paper_bgcolor': 'white',
        'plot_bgcolor': Colors.BACKGROUND,
        'hovermode': 'closest'
    }


def get_standard_legend() -> Dict[str, Any]:
    """
    Generate standard legend configuration.
    
    Returns:
        dict: Plotly legend configuration (horizontal, top-centered)
    """
    return {
        'orientation': 'h',
        'yanchor': 'bottom',
        'y': 1.02,
        'xanchor': 'center',
        'x': 0.5,
        'bgcolor': 'rgba(255, 255, 255, 0.9)',
        'bordercolor': 'gray',
        'borderwidth': 1,
        'font': {'size': Typography.LEGEND_SIZE}
    }


def get_axis_config(title: str) -> Dict[str, Any]:
    """
    Generate standard axis configuration.
    
    Parameters:
        title: str - axis label text
    
    Returns:
        dict: Plotly axis configuration (title, ticks, grid)
    """
    return {
        'title': {
            'text': title,
            'font': {'size': Typography.AXIS_LABEL_SIZE}
        },
        'tickfont': {'size': Typography.AXIS_TICK_SIZE},
        'gridcolor': Colors.GRID
    }


def get_table_header_style() -> Dict[str, Any]:
    """
    Generate table header styling.
    
    Returns:
        dict: Header style (blue background, white text, centered)
    """
    return {
        'fill_color': Colors.CLIMB,
        'align': 'center',
        'font': {
            'color': 'white',
            'size': Typography.AXIS_LABEL_SIZE,
            'family': Typography.FONT_FAMILY
        }
    }


def get_table_cell_style() -> Dict[str, Any]:
    """
    Generate table cell styling.
    
    Returns:
        dict: Cell style (centered, standard font, fixed height)
    """
    return {
        'align': 'center',
        'font': {
            'size': Typography.HOVER_SIZE,
            'family': Typography.FONT_FAMILY
        },
        'height': 35
    }


# ========================================================================
# SECTION 7: HOVER TOOLTIP TEMPLATES
# ========================================================================

class HoverTemplates:
    """Standardized hover tooltip formats for interactive plots."""
    
    @staticmethod
    def fuel(phase: str, x_label: str, x_unit: str) -> str:
        """
        Fuel data hover template.
        
        Format: Phase | x-variable | Fuel [kg]
        """
        return (
            f"<b>{phase}</b><br>"
            f"{x_label}: %{{x:.2f}} {x_unit}<br>"
            f"Fuel: %{{y:.1f}} kg<br>"
            "<extra></extra>"
        )
    
    @staticmethod
    def altitude(phase: str, x_label: str, x_unit: str) -> str:
        """
        Altitude data hover template.
        
        Format: Phase | x-variable | Altitude [m]
        """
        return (
            f"<b>{phase}</b><br>"
            f"{x_label}: %{{x:.2f}} {x_unit}<br>"
            f"Altitude: %{{y:.0f}} m<br>"
            "<extra></extra>"
        )


# ========================================================================
# SECTION 8: EXPORT CONFIGURATION
# ========================================================================

class ExportConfig:
    """Plot export parameters for file output."""
    
    IMAGE_FORMAT = 'png'        # Output format
    IMAGE_SCALE = 2             # Resolution multiplier (2× for high-DPI)
    IMAGE_HEIGHT = 800          # Export height [px]
    IMAGE_WIDTH = 1200          # Export width [px]
    
    @staticmethod
    def get_plotly_config() -> Dict[str, Any]:
        """
        Generate Plotly export configuration.
        
        Returns:
            dict: Plotly config with toolbar and export settings
        """
        return {
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': ExportConfig.IMAGE_FORMAT,
                'filename': 'mission_analysis_plot',
                'height': ExportConfig.IMAGE_HEIGHT,
                'width': ExportConfig.IMAGE_WIDTH,
                'scale': ExportConfig.IMAGE_SCALE
            }
        }
