"""
Visualization Configuration Module

Centralized configuration for consistent styling across all mission analysis visualizations.
Provides publication-quality settings for scientific papers.

Author: Mission Analysis System
"""

from typing import Dict, Any
import os
from datetime import datetime
from pathlib import Path

# ============================================================================
# SHARED RUN DIRECTORY - Used by all plotting modules
# ============================================================================

# Global variable to store the output directory for the current run (shared across all modules)
_CURRENT_RUN_DIR = None


def get_or_create_run_directory(base_dir: str = None, phase: str = None) -> str:
    """
    Get or create a timestamped directory for the current run with phase subfolders.
    This function is shared across all plotting modules to ensure a single timestamped folder.
    Creates directory structure: Images/YYYY-MM-DD_HH-MM-SS/Phase/
    
    Args:
        base_dir: Base directory path. If None, uses Images folder in workspace.
        phase: Phase name ('Climb', 'Cruise', 'Descent', 'Optimized'). If None, returns base run directory.
    
    Returns:
        Path to the phase subdirectory (or base run directory if phase is None)
    """
    global _CURRENT_RUN_DIR
    
    # If already created for this run, use existing path
    if _CURRENT_RUN_DIR is None:
        # Use provided base_dir or default to Images folder
        if base_dir is None:
            # Get workspace root (where main.py is located)
            workspace_root = Path(__file__).parent
            base_dir = workspace_root / "Images"
        else:
            base_dir = Path(base_dir)
        
        # Create Images directory if it doesn't exist
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = base_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Store for this run
        _CURRENT_RUN_DIR = str(run_dir)
        
        print(f"[EXPORT] Created output directory: {run_dir}")
    
    # If phase is specified, create and return phase subfolder
    if phase:
        phase_dir = Path(_CURRENT_RUN_DIR) / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        return str(phase_dir)
    
    return _CURRENT_RUN_DIR


# ============================================================================
# COLOR SCHEME - Consistent across all plots
# ============================================================================

class Colors:
    """Standard color palette for mission phases and data types."""
    
    # Mission Phases
    CLIMB = 'royalblue'
    CRUISE = 'green'
    DESCENT = 'crimson'
    
    # Performance Metrics
    THRUST = 'blue'
    DRAG = 'red'
    FUEL_FLOW = 'darkgreen'
    WEIGHT = 'purple'
    LEVER = 'orange'
    AIRSPEED = 'teal'
    FUEL_CONSUMED = 'crimson'
    
    # UI Elements
    GRID = 'lightgray'
    BACKGROUND = 'rgba(240, 240, 245, 0.5)'
    TRANSITION_LINE = 'gray'
    WARNING = 'orange'
    CRITICAL = 'red'
    
    # Envelope Limits
    ENVELOPE_LIMIT = 'crimson'
    OPTIMAL_PATH = 'yellow'  # For climb
    OPTIMAL_PATH_DESCENT = 'darkviolet'  # For descent
    
    # 3D Visualization
    J_VALUES_CLIMB = 'darkviolet'
    J_VALUES_DESCENT = 'darkorange'


# ============================================================================
# TYPOGRAPHY - Fonts and sizes
# ============================================================================

class Typography:
    """Standard typography settings for all text elements."""
    
    FONT_FAMILY = 'Arial, sans-serif'
    
    # Title sizes
    MAIN_TITLE_SIZE = 18
    SUBPLOT_TITLE_SIZE = 14
    
    # Text sizes
    AXIS_LABEL_SIZE = 14
    AXIS_TICK_SIZE = 12
    LEGEND_SIZE = 11
    ANNOTATION_SIZE = 10
    HOVER_SIZE = 11
    
    # Font weights
    TITLE_WEIGHT = 'bold'


# ============================================================================
# LAYOUT - Dimensions and spacing
# ============================================================================

class Layout:
    """Standard layout dimensions and spacing."""
    
    # Figure dimensions
    STANDARD_WIDTH = 1200
    STANDARD_HEIGHT = 1000
    DASHBOARD_WIDTH = 1200
    DASHBOARD_HEIGHT = 1400
    WIDE_WIDTH = 1400
    TALL_HEIGHT = 1200
    
    # Spacing
    VERTICAL_SPACING = 0.12
    HORIZONTAL_SPACING = 0.12
    
    # Margins
    MARGIN_LEFT = 80
    MARGIN_RIGHT = 150
    MARGIN_TOP = 100
    MARGIN_BOTTOM = 80


# ============================================================================
# LINE STYLES - Consistent across all plots
# ============================================================================

class LineStyles:
    """Standard line styling parameters."""
    
    # Line widths
    THICK = 3
    MEDIUM = 2
    THIN = 1
    
    # Dash styles
    SOLID = 'solid'
    DASH = 'dash'
    DOT = 'dot'
    DASHDOT = 'dashdot'


# ============================================================================
# PLOT TEMPLATES
# ============================================================================

def get_standard_layout(title: str, subtitle: str = "", height: int = None, width: int = None) -> Dict[str, Any]:
    """
    Get standard layout configuration for Plotly figures.
    
    Args:
        title: Main title text
        subtitle: Optional subtitle with key metrics
        height: Optional custom height (defaults to STANDARD_HEIGHT)
        width: Optional custom width (defaults to STANDARD_WIDTH)
    
    Returns:
        Dictionary with layout configuration
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
    """Get standard legend configuration."""
    return {
        'orientation': 'h',
        'yanchor': 'bottom',
        'y': 1.02,
        'xanchor': 'center',
        'x': 0.5,
        'bgcolor': 'rgba(255, 255, 255, 0.9)',
        'bordercolor': 'gray',
        'borderwidth': 1,
        'font': {
            'size': Typography.LEGEND_SIZE
        }
    }


def get_axis_config(title: str) -> Dict[str, Any]:
    """
    Get standard axis configuration.
    
    Args:
        title: Axis title text
    
    Returns:
        Dictionary with axis configuration
    """
    return {
        'title': {
            'text': title,
            'font': {
                'size': Typography.AXIS_LABEL_SIZE
            }
        },
        'tickfont': {
            'size': Typography.AXIS_TICK_SIZE
        },
        'gridcolor': Colors.GRID
    }


def get_table_header_style() -> Dict[str, Any]:
    """Get standard table header styling."""
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
    """Get standard table cell styling."""
    return {
        'align': 'center',
        'font': {
            'size': Typography.HOVER_SIZE,
            'family': Typography.FONT_FAMILY
        },
        'height': 35
    }


# ============================================================================
# PHASE-SPECIFIC CONFIGURATIONS
# ============================================================================

class PhaseColors:
    """Pre-configured settings for each mission phase."""
    
    @staticmethod
    def get_climb_trace_style(name: str = 'Climb') -> Dict[str, Any]:
        """Get standard trace style for climb phase."""
        return {
            'name': name,
            'line': {
                'color': Colors.CLIMB,
                'width': LineStyles.THICK
            },
            'mode': 'lines'
        }
    
    @staticmethod
    def get_cruise_trace_style(name: str = 'Cruise') -> Dict[str, Any]:
        """Get standard trace style for cruise phase."""
        return {
            'name': name,
            'line': {
                'color': Colors.CRUISE,
                'width': LineStyles.THICK
            },
            'mode': 'lines'
        }
    
    @staticmethod
    def get_descent_trace_style(name: str = 'Descent') -> Dict[str, Any]:
        """Get standard trace style for descent phase."""
        return {
            'name': name,
            'line': {
                'color': Colors.DESCENT,
                'width': LineStyles.THICK
            },
            'mode': 'lines'
        }


# ============================================================================
# HOVER TEMPLATE FORMATTING
# ============================================================================

class HoverTemplates:
    """Standard hover template formats for consistency."""
    
    @staticmethod
    def standard(phase: str, x_label: str, x_unit: str, y_label: str, y_unit: str) -> str:
        """
        Create standard hover template.
        
        Args:
            phase: Phase name (Climb, Cruise, Descent)
            x_label: Label for x-axis data
            x_unit: Unit for x-axis
            y_label: Label for y-axis data
            y_unit: Unit for y-axis
        
        Returns:
            Formatted hover template string
        """
        return (
            f"<b>{phase}</b><br>"
            f"{x_label}: %{{x:.2f}} {x_unit}<br>"
            f"{y_label}: %{{y:.2f}} {y_unit}<br>"
            "<extra></extra>"
        )
    
    @staticmethod
    def fuel(phase: str, x_label: str, x_unit: str) -> str:
        """Hover template for fuel data."""
        return (
            f"<b>{phase}</b><br>"
            f"{x_label}: %{{x:.2f}} {x_unit}<br>"
            f"Fuel: %{{y:.1f}} kg<br>"
            "<extra></extra>"
        )
    
    @staticmethod
    def altitude(phase: str, x_label: str, x_unit: str) -> str:
        """Hover template for altitude data."""
        return (
            f"<b>{phase}</b><br>"
            f"{x_label}: %{{x:.2f}} {x_unit}<br>"
            f"Altitude: %{{y:.0f}} m<br>"
            "<extra></extra>"
        )


# ============================================================================
# EXPORT SETTINGS
# ============================================================================

class ExportConfig:
    """Configuration for exporting plots."""
    
    IMAGE_FORMAT = 'png'
    IMAGE_SCALE = 2  # For high-resolution exports
    IMAGE_HEIGHT = 800
    IMAGE_WIDTH = 1200
    
    @staticmethod
    def get_plotly_config() -> Dict[str, Any]:
        """Get standard Plotly configuration for exports."""
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
