"""
Climb Phase Visualization Module

This module provides comprehensive interactive visualization capabilities
for climb phase analysis using Plotly.
"""

# =========  1 - MODULE INITIALIZATION =================
# ========= IMPORTS AND BASIC SETUP ===========================================
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib import patheffects as pe
from typing import Optional, Callable, List, Dict, Any, Tuple
import os
import warnings
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.io as pio

# Suppress choreographer JSONError warnings (non-critical browser communication errors)
warnings.filterwarnings('ignore', category=UserWarning, module='choreographer')
try:
    import logging
    logging.getLogger('choreographer').setLevel(logging.ERROR)
except:
    pass

pio.renderers.default = "browser"

# Import aircraft configuration from centralized module
from aircraft_config import (
    isa_properties, a_from_altitude, INITIAL_MASS_KG,
    M_MMO, S_REF_M2, CL_MAX
)

# Import mission configuration parameters
from mission_config import (
    TARGET_ALT_CLIMB_M,
    MAX_SERVICE_CEILING_M
)

# Import climb module for data structures
from climb import ClimbingCore

# Import visualization configuration for consistent styling
from visualization_config import (
    Colors, Typography, Layout, LineStyles,
    get_standard_layout, get_standard_legend, get_axis_config,
    ExportConfig, get_or_create_run_directory
)

# =========  2 - PLOTTING CONFIGURATION ========================
class PlottingConfig:
    """Configuration constants for visualization and graphical representation."""
    
    # Specific excess power contour levels for visualization
    PS_LEVELS = np.array(
        [-30,-25,-20,-15,-12,-10,-8,-6,-4,-2,-1,-0.5,
          0.5,1,2,3,4,5,6,8,10,12,15,20,23,25,30,33,35,40,45,50],
        dtype=float
    )
    
    # User interface visualization limits
    M_XMAX_UI = 1.25  # Maximum Mach number for SEP x-axis visualization

# ========= GRID CONFIGURATION ========================
class GridConfig:
    """Configuration constants for grids, axes, and UI layout.
    
    These values control plotting and visualization aspects of the climb analysis.
    The target altitude is imported from mission_config.py and used for:
    - Calculating altitude fractions for penalty calculations
    - Determining step sizes for constant Mach strategies
    - Grid generation for dynamic programming optimization
    
    Plotting-specific parameters (ALT_STEP_M, Y_AXIS_TOP_M) are defined here
    as they are specific to visualization, not mission parameters.
    """
    
    # Target altitude (imported from mission_config.py - this is a mission parameter)
    TARGET_ALT_M = TARGET_ALT_CLIMB_M  # Target climb altitude [m] (from mission_config.py)
    
    # Plotting and visualization settings (defined here - plotting-specific)
    Y_AXIS_TOP_M = 14000.0  # Maximum altitude for plots [m]
    ALT_STEP_M = 200.0  # Altitude step size for plotting [m]
    
    # Grid resolution settings
    MACH_COLS = 81
    N_PLOT_STEPS = 50  # uniform # of points per trajectory

# ========= GRID AND PLOTTING UTILITIES ========================
class GridAndPlotting:
    """Handles computational grid generation and data preparation for visualization."""
    
    @staticmethod
    def compute_sep_grid_maxlever(aero, engine, ref_mass_kg: float,
                                  mach_grid=None,
                                  H_grid=None):
        """Compute specific excess power Ps = ((T-D)V)/W at maximum lever for visualization backgrounds."""
        from aircraft_config import SystemConfiguration, a_from_altitude
        
        if mach_grid is None: mach_grid = aero.mach_grid
        if H_grid is None: H_grid = aero.alt_grid_m
        Ps = np.full((len(H_grid), len(mach_grid)), np.nan)
        W = ref_mass_kg * aero.G_C
        for k, h in enumerate(H_grid):
            a = a_from_altitude(float(h))
            for i, M in enumerate(mach_grid):
                V = max(a*float(M), 0.1)
                T_per = engine.thrust_with_lever(1.0, M, h)  # max lever
                if T_per is None:
                    continue
                T_tot = T_per * SystemConfiguration.N_ENGINES
                D = aero.get_drag(M, h, ref_mass_kg)
                Ps[k, i] = ((T_tot - D) * V) / W
        return mach_grid, H_grid, Ps

# =========  3 - PLOTTING FUNCTIONS ========================
# =========================================================================
# STRATEGY PLOTTING FUNCTIONS MOVED TO climb_strategies_plotting.py
# =========================================================================
# The following functions have been moved to climb_strategies_plotting.py
# to support independent strategy comparison analysis:
#   - plot_strategies_interactive()
#   - create_strategy_comparison_plots()
#   - create_browser_comparison_table()
# 
# Import them from climb_strategies_plotting if ENABLE_STRATEGY_COMPARISON is True
# =========================================================================


# =========================================================================
# 3D VISUALIZATION FUNCTIONS
# =========================================================================

def plot_3d_cost_space(mach_grid: np.ndarray, altitude_sched: np.ndarray, 
                      lever_grid: np.ndarray, J_grid_3d: np.ndarray, 
                      min_path: Optional[Dict] = None, title: Optional[str] = None):
    """
    Visualize fuel cost density (J) in 3D state space using interactive Plotly.
    
    Creates an interactive 3D visualization of the cost landscape with flight envelope
    limits and optimal path overlay. Includes camera presets, data filtering, and export.
    
    Args:
        mach_grid: Mach number grid
        altitude_sched: Altitude schedule
        lever_grid: Lever position grid
        J_grid_3d: 3D array of J values (fuel cost density)
        min_path: Optional dict with keys 'mach', 'alt', 'lever' for optimal path
        title: Optional custom title
        
    Returns:
        Plotly figure object
    """
    # Engine envelope limits (from engine envelope analysis)
    MAX_ENGINE_MACH = 0.9392  # Maximum operational Mach from engine envelope analysis
    MIN_ENGINE_MACH = 0.200  # Minimum operational Mach from engine envelope test
    
    # Prepare meshgrid for scatter
    M, H, L = np.meshgrid(mach_grid, altitude_sched, lever_grid, indexing='ij')
    # Flatten arrays for plotting - NEW AXIS ASSIGNMENT
    x = L.flatten()  # Lever (X-axis)
    y = M.flatten()  # Mach (Y-axis) 
    z = H.flatten()  # Altitude (Z-axis)
    J_flat = J_grid_3d.flatten()
    # Only plot finite J values
    mask = np.isfinite(J_flat)
    x, y, z, J_flat = x[mask], y[mask], z[mask], J_flat[mask]
    
    # Enhanced hover text with more detailed information
    hover_text = []
    for lx, my, hz, jv in zip(x, y, z, J_flat):
        # Calculate additional metrics for hover
        _, _, rho = isa_properties(float(hz))
        a = a_from_altitude(float(hz))
        V = my * a
        hover_text.append(
            f"<b>Point Details</b><br>"
            f"Lever: {lx:.3f}<br>"
            f"Mach: {my:.3f}<br>"
            f"Altitude: {hz:.1f} m<br>"
            f"J: {jv:.4g} kg/m<br>"
            f"Velocity: {V:.1f} m/s<br>"
            f"Air Density: {rho:.3f} kg/m³"
        )

    fig = go.Figure()
    
    # Add flight envelope limits
    # 1. Maximum Engine Mach limit - vertical plane at MAX_ENGINE_MACH
    # Create a vertical plane at MAX_ENGINE_MACH across all altitudes and lever positions
    lever_range = np.linspace(0, 1, 10)
    alt_range = np.linspace(altitude_sched[0], altitude_sched[-1], 10)
    L_mmo, H_mmo = np.meshgrid(lever_range, alt_range)
    M_mmo = np.full_like(L_mmo, MAX_ENGINE_MACH)
    
    fig.add_trace(go.Surface(
        x=L_mmo,
        y=M_mmo,
        z=H_mmo,
        colorscale=[[0, Colors.ENVELOPE_LIMIT], [1, Colors.ENVELOPE_LIMIT]],
        opacity=0.45,
        showscale=False,
        name=f'Max Engine Mach Limit (M={MAX_ENGINE_MACH:.3f})'
    ))
    
    # 2. CLmax (stall) limit - compute stall curve
    def _compute_mstall_curve():
        # Get gravity constant from Atmosphere class
        from atmosphere import Atmosphere
        W = INITIAL_MASS_KG * Atmosphere.G_C
        # Use CL_MAX from aircraft_config (fixed value)
        cl_max_value = CL_MAX
        out = np.full_like(altitude_sched, np.nan, float)
        for k, h in enumerate(altitude_sched):
            _, _, rho = isa_properties(float(h))
            a = a_from_altitude(float(h))
            q_req = W / (S_REF_M2 * cl_max_value)
            if rho > 0:
                V = np.sqrt(2*q_req/max(rho,1e-12))
                out[k] = V / max(a,1e-12)
        return out
    
    M_stall = _compute_mstall_curve()
    if np.isfinite(M_stall).any():
        # Create stall surface - vertical plane at stall Mach
        lever_range = np.linspace(0, 1, 10)
        alt_range = np.linspace(altitude_sched[0], altitude_sched[-1], 10)
        L_stall, H_stall = np.meshgrid(lever_range, alt_range)
        M_stall_surface = np.full_like(L_stall, np.nan)
        
        # Fill in stall Mach values where they exist
        for i, h in enumerate(alt_range):
            h_idx = np.argmin(np.abs(altitude_sched - h))
            if h_idx < len(M_stall) and np.isfinite(M_stall[h_idx]):
                M_stall_surface[i, :] = M_stall[h_idx]
        
        # Only plot where we have valid stall data
        valid_mask = np.isfinite(M_stall_surface)
        if np.any(valid_mask):
            fig.add_trace(go.Surface(
                x=L_stall,
                y=M_stall_surface,
                z=H_stall,
                colorscale=[[0, Colors.ENVELOPE_LIMIT], [1, Colors.ENVELOPE_LIMIT]],
                opacity=0.45,
                showscale=False,
                name='Flight Envelope Limit'
            ))
    
    # 3. Operating envelope boundaries (between stall and Max Engine Mach) - as lines
    if np.isfinite(M_stall).any():
        cond = np.isfinite(M_stall) & (M_stall < MAX_ENGINE_MACH)
        if np.any(cond):
            # Create operating envelope boundary lines (separate traces to avoid connecting lines)
            for lever_val in [0.0, 0.5, 1.0]:  # Show at different lever positions
                # Stall edge line
                fig.add_trace(go.Scatter3d(
                    x=[lever_val] * int(np.sum(cond)),
                    y=M_stall[cond],
                    z=altitude_sched[cond],
                    mode='lines',
                    line=dict(color=Colors.ENVELOPE_LIMIT, width=LineStyles.THICK),
                    name='Operating Envelope' if lever_val == 0.0 else None,
                    showlegend=(lever_val == 0.0)
                ))
                # Max Engine Mach edge line
                fig.add_trace(go.Scatter3d(
                    x=[lever_val] * int(np.sum(cond)),
                    y=(MAX_ENGINE_MACH * np.ones_like(M_stall[cond])),
                    z=altitude_sched[cond],
                    mode='lines',
                    line=dict(color=Colors.ENVELOPE_LIMIT, width=LineStyles.THICK),
                    showlegend=False
                ))
    
    # 4. Maximum Service Ceiling limit - horizontal plane
    # Always show if within reasonable range (even if slightly above altitude_sched for visibility)
    if MAX_SERVICE_CEILING_M <= (altitude_sched[-1] if len(altitude_sched) > 0 else 15000) + 1000:
        lever_range_ceiling = np.linspace(0, 1, 10)
        mach_range_ceiling = np.linspace(mach_grid[0] if len(mach_grid) > 0 else 0.1, MAX_ENGINE_MACH, 10)
        L_ceiling, M_ceiling = np.meshgrid(lever_range_ceiling, mach_range_ceiling)
        H_ceiling = np.full_like(L_ceiling, MAX_SERVICE_CEILING_M)
        
        fig.add_trace(go.Surface(
            x=L_ceiling,
            y=M_ceiling,
            z=H_ceiling,
            colorscale=[[0, Colors.ENVELOPE_LIMIT], [1, Colors.ENVELOPE_LIMIT]],
            opacity=0.45,
            showscale=False,
            name=f'Max Service Ceiling ({MAX_SERVICE_CEILING_M/1000:.2f} km)'
        ))
    
    # 5. Minimum Engine Mach limit - vertical plane
    if MIN_ENGINE_MACH >= (mach_grid[0] if len(mach_grid) > 0 else 0.0):
        lever_range_min = np.linspace(0, 1, 10)
        alt_range_min = np.linspace(altitude_sched[0], altitude_sched[-1], 10)
        L_min, H_min = np.meshgrid(lever_range_min, alt_range_min)
        M_min = np.full_like(L_min, MIN_ENGINE_MACH)
        
        fig.add_trace(go.Surface(
            x=L_min,
            y=M_min,
            z=H_min,
            colorscale=[[0, Colors.ENVELOPE_LIMIT], [1, Colors.ENVELOPE_LIMIT]],
            opacity=0.45,
            showscale=False,
            name=f'Min Engine Mach ({MIN_ENGINE_MACH:.3f})'
        ))
    
    # Add J values scatter plot (dark purple)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=Colors.J_VALUES_CLIMB,
            opacity=0.7,
            line=dict(width=0)
        ),
        name='J values',
        text=hover_text,
        hovertemplate='%{text}<extra></extra>'
    ))

    # Overlay minimum-fuel path if provided
    if min_path is not None:
        # min_path should be a dict with keys: 'mach', 'alt', 'lever'
        fig.add_trace(go.Scatter3d(
            x=min_path['lever'],  # Lever on X-axis
            y=min_path['mach'],   # Mach on Y-axis
            z=min_path['alt'],    # Altitude on Z-axis
            mode='lines+markers',
            line=dict(color=Colors.OPTIMAL_PATH, width=6),
            marker=dict(size=6, color=Colors.OPTIMAL_PATH),
            name='Optimal Path',
            hovertemplate='<b>Optimal Path</b><br>Lever: %{x:.3f}<br>Mach: %{y:.3f}<br>Alt: %{z:.1f} m<extra></extra>'
        ))

    # Enhanced layout with user-friendly features
    fig.update_layout(
        scene=dict(
            xaxis_title='Lever',
            yaxis_title='Mach',
            zaxis_title='Altitude (m)',
            # Camera presets
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),  # Default isometric view
                center=dict(x=0, y=0, z=0)
            ),
            # Enhanced axis formatting using standard config
            xaxis=dict(
                title=dict(font=dict(size=Typography.AXIS_LABEL_SIZE)),
                tickfont=dict(size=Typography.AXIS_TICK_SIZE),
                gridcolor='rgba(128,128,128,0.3)'
            ),
            yaxis=dict(
                title=dict(font=dict(size=Typography.AXIS_LABEL_SIZE)),
                tickfont=dict(size=Typography.AXIS_TICK_SIZE),
                gridcolor='rgba(128,128,128,0.3)'
            ),
            zaxis=dict(
                title=dict(font=dict(size=Typography.AXIS_LABEL_SIZE)),
                tickfont=dict(size=Typography.AXIS_TICK_SIZE),
                gridcolor='rgba(128,128,128,0.3)'
            )
        ),
        title=dict(
            text=title if title is not None else '3D Visualization of J (Fuel/Energy) - Enhanced View',
            font=dict(size=Typography.MAIN_TITLE_SIZE, family=Typography.FONT_FAMILY),
            x=0.5
        ),
        # Add control buttons
        updatemenus=[
            # Camera preset buttons
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"scene.camera.eye": {"x": 0, "y": 0, "z": 2.5}}],
                        label="Front View",
                        method="relayout"
                    ),
                    dict(
                        args=[{"scene.camera.eye": {"x": 2.5, "y": 0, "z": 0}}],
                        label="Side View",
                        method="relayout"
                    ),
                    dict(
                        args=[{"scene.camera.eye": {"x": 0, "y": 2.5, "z": 0}}],
                        label="Top View",
                        method="relayout"
                    ),
                    dict(
                        args=[{"scene.camera.eye": {"x": 1.5, "y": 1.5, "z": 1.5}}],
                        label="Isometric",
                        method="relayout"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.02,
                yanchor="top"
            ),
            # Data filtering buttons
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"visible": [True] * len(fig.data)}],
                        label="Show All",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [True if trace.name and "J values" in trace.name else False for trace in fig.data]}],
                        label="J Values Only",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [True if trace.name and "Optimal Path" in trace.name else False for trace in fig.data]}],
                        label="Optimal Path Only",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [True if trace.name and ("MMO Limit" in trace.name or "Flight Envelope Limit" in trace.name or "Operating Envelope" in trace.name) else False for trace in fig.data]}],
                        label="Envelope Only",
                        method="restyle"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.25,
                xanchor="left",
                y=1.02,
                yanchor="top"
            )
        ],
        # Enhanced legend
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=Typography.LEGEND_SIZE)
        ),
        # Add margin for controls using standard config
        margin=dict(l=0, r=Layout.MARGIN_RIGHT, t=Layout.MARGIN_TOP, b=Layout.MARGIN_BOTTOM)
    )

    # Set template
    fig.update_layout(template="plotly_white")

    # Write to HTML file with enhanced features using standard export config
    export_config = ExportConfig.get_plotly_config()
    export_config['toImageButtonOptions']['filename'] = '3d_plot_export'
    export_config['modeBarButtonsToAdd'] = [
        "drawline", "drawopenpath", "drawclosedpath",
        "drawcircle", "drawrect", "eraseshape"
    ]
    
    # Save to Climb folder (both HTML for interaction and PNG for static)
    climb_dir = get_or_create_run_directory(phase="Climb")
    output_path_html = os.path.join(climb_dir, 'climb_J_3d_plot.html')
    output_path_png = os.path.join(climb_dir, 'climb_J_3d_plot.png')
    
    pio.write_html(
        fig, 
        file=output_path_html, 
        auto_open=True,
        config=export_config
    )
    
    # Also save as PNG
    try:
        fig.write_image(output_path_png, width=1600, height=1200, scale=2)
        print(f"[EXPORT] 3D J plot saved to: {output_path_html} (interactive) and {output_path_png} (PNG)")
    except Exception as e:
        print(f"[EXPORT] 3D J plot saved to: {output_path_html} (HTML only)")
        print(f"[WARNING] Could not save PNG version: {e}")
    
    return fig


# =========================================================================
# 2D PERFORMANCE ANALYSIS FUNCTIONS
# =========================================================================

def plot_performance_2d(climb_result, climb_info: Optional[Dict[str, Any]] = None):
    """
    Create detailed performance analysis in 2D multi-panel layout.
    
    Visualizes key performance metrics over time in a 3×2 subplot grid with
    consistent styling and individual subplot export capability.
    
    Args:
        climb_result: Results from climb DP optimization (MinFuelSchedule)
        climb_info: Optional dictionary with climb optimization info
        
    Returns:
        Plotly figure object
    """
    
    # Extract climb data
    climb_time_s = np.cumsum(np.nan_to_num(climb_result.dt_s, nan=0.0, posinf=0.0, neginf=0.0))
    climb_time_min = climb_time_s / 60.0  # Convert to minutes
    climb_alt_m = np.asarray(climb_result.alt_m, float)
    climb_mach = np.asarray(climb_result.mach, float)
    climb_lever = np.asarray(climb_result.lever, float)
    climb_thrust_N = np.asarray(climb_result.T_total_N, float)
    climb_drag_N = np.asarray(climb_result.D_N, float)
    climb_fuel_kg = np.asarray(climb_result.cumFuel_kg, float)
    climb_mass_kg = np.asarray(climb_result.mass_kg, float)  # Use actual dynamic mass from DP optimization
    
    # Calculate fuel flow rate (kg/h) - approximate from fuel consumption
    fuel_flow_kgh = []
    for i in range(len(climb_result.dt_s)):
        if i == 0:
            fuel_flow_kgh.append(0.0)
        else:
            dt_hours = climb_result.dt_s[i] / 3600.0
            if dt_hours > 0:
                fuel_consumed = climb_fuel_kg[i] - climb_fuel_kg[i-1]
                fuel_flow_kgh.append(fuel_consumed / dt_hours)
            else:
                fuel_flow_kgh.append(0.0)
    fuel_flow_kgh = np.array(fuel_flow_kgh)
    
    # Calculate true airspeed
    tas_ms = []
    for i in range(len(climb_mach)):
        a = a_from_altitude(float(climb_alt_m[i]))
        tas_ms.append(climb_mach[i] * a)
    tas_ms = np.array(tas_ms)
    
    # Convert forces to kN
    thrust_kn = climb_thrust_N / 1000
    drag_kn = climb_drag_N / 1000
    
    # Create climb performance plot (3x2 grid) with consistent styling
    fig = sp.make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '<b>Fuel Flow Rate</b>', '<b>Thrust vs Drag</b>', 
            '<b>Mass Evolution</b>', '<b>Lever Position</b>',
            '<b>True Airspeed</b>', '<b>Cumulative Fuel Consumption</b>'
        ),
        vertical_spacing=0.15,  # Same as cruise and descent
        horizontal_spacing=0.15
    )
    
    # 1. Fuel Flow Rate over time (aligned with cruise and descent)
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=fuel_flow_kgh,
            mode='lines',
            name='Fuel Flow (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(65, 105, 225, 0.15)',
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Fuel Flow: %{y:.1f} kg/h<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Thrust vs Drag over time (aligned with cruise and descent)
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=thrust_kn,
            mode='lines',
            name='Thrust (Climb)',
            line=dict(color='darkblue', width=LineStyles.THICK),
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Thrust: %{y:.1f} kN<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=drag_kn,
            mode='lines',
            name='Drag (Climb)',
            line=dict(color='lightblue', width=LineStyles.THICK, dash=LineStyles.DASH),
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Drag: %{y:.1f} kN<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Mass Evolution over time (aligned with cruise and descent)
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_mass_kg,
            mode='lines',
            name='Mass (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(65, 105, 225, 0.15)',
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Mass: %{y:.0f} kg<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Engine Lever Position over time (aligned with cruise and descent)
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_lever * 100,  # Convert to percentage
            mode='lines',
            name='Lever Position (Climb)',
            line=dict(color='steelblue', width=LineStyles.THICK),
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Lever: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Add reference lines for lever position (same as cruise and descent)
    fig.add_hline(y=85, line_dash=LineStyles.DASH, line_color=Colors.WARNING, 
                  annotation_text="MCT Limit", annotation_position="top right",
                  row=2, col=2)
    fig.add_hline(y=100, line_dash=LineStyles.DOT, line_color=Colors.CRITICAL,
                  annotation_text="Max Thrust", annotation_position="bottom right",
                  row=2, col=2)
    
    # 5. True Airspeed over time (aligned with cruise and descent)
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=tas_ms,
            mode='lines',
            name='True Airspeed (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK),
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>TAS: %{y:.1f} m/s<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 6. Cumulative Fuel Consumption over time (aligned with cruise and descent)
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_fuel_kg,
            mode='lines',
            name='Fuel Consumed (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(65, 105, 225, 0.15)',
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Fuel: %{y:.1f} kg<extra></extra>'
        ),
        row=3, col=2
    )
    
    # Calculate summary statistics
    total_time_min = climb_time_min[-1] if len(climb_time_min) > 0 else 0.0
    total_fuel = climb_fuel_kg[-1] if len(climb_fuel_kg) > 0 else 0.0
    avg_fuel_flow = np.mean(fuel_flow_kgh[fuel_flow_kgh > 0]) if np.any(fuel_flow_kgh > 0) else 0.0
    altitude_gain = climb_alt_m[-1] - climb_alt_m[0] if len(climb_alt_m) > 0 else 0.0
    
    # Update layout with standard configuration (aligned with cruise and descent)
    subtitle = (
        f"Altitude Gain: {altitude_gain:.0f} m | Time: {total_time_min:.1f} min | "
        f"Fuel: {total_fuel:.1f} kg | Avg Fuel Flow: {avg_fuel_flow:.0f} kg/h"
    )
    
    layout_config = get_standard_layout(
        "CLIMB PERFORMANCE ANALYSIS (2D)",
        subtitle,
        height=Layout.STANDARD_HEIGHT,
        width=Layout.STANDARD_WIDTH
    )
    
    # Add extra margin for title and legend spacing
    layout_config['margin'] = dict(l=80, r=200, t=120, b=80)
    
    fig.update_layout(
        **layout_config,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=11)
        )
    )
    
    # Update axes labels with standard configuration (aligned with cruise and descent)
    fig.update_xaxes(**get_axis_config("Time (min)"), row=1, col=1)
    fig.update_yaxes(**get_axis_config("Fuel Flow (kg/h)"), row=1, col=1)
    
    fig.update_xaxes(**get_axis_config("Time (min)"), row=1, col=2)
    fig.update_yaxes(**get_axis_config("Force (kN)"), row=1, col=2)
    
    fig.update_xaxes(**get_axis_config("Time (min)"), row=2, col=1)
    # Set y-axis range to zoom in on mass changes for better visibility
    mass_min, mass_max = np.min(climb_mass_kg), np.max(climb_mass_kg)
    mass_margin = (mass_max - mass_min) * 0.2  # Add 20% margin
    fig.update_yaxes(**get_axis_config("Mass (kg)"), 
                     range=[mass_min - mass_margin, mass_max + mass_margin], 
                     row=2, col=1)
    
    fig.update_xaxes(**get_axis_config("Time (min)"), row=2, col=2)
    fig.update_yaxes(**get_axis_config("Lever Position (%)"), row=2, col=2)
    
    fig.update_xaxes(**get_axis_config("Time (min)"), row=3, col=1)
    fig.update_yaxes(**get_axis_config("True Airspeed (m/s)"), row=3, col=1)
    
    fig.update_xaxes(**get_axis_config("Time (min)"), row=3, col=2)
    fig.update_yaxes(**get_axis_config("Cumulative Fuel (kg)"), row=3, col=2)
    
    # Subplot selection menu removed - all subplots always visible
    
    # Add export configuration for high-quality image export
    config = ExportConfig.get_plotly_config()
    config['toImageButtonOptions']['filename'] = 'climb_performance_2d'
    
    # Save individual plots as separate HTML files in timestamped directory/Climb subfolder
    try:
        run_dir = get_or_create_run_directory(phase="Climb")
        save_prefix = "climb_performance"
        
        # 1. Fuel Flow Rate
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=climb_time_min, y=fuel_flow_kgh, mode='lines', name='Fuel Flow (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(65, 105, 225, 0.15)',
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Fuel Flow: %{y:.1f} kg/h<extra></extra>'))
        fig1.update_layout(**get_standard_layout("CLIMB PERFORMANCE - Fuel Flow Rate", subtitle, height=600, width=900))
        fig1.update_xaxes(**get_axis_config("Time (min)")); fig1.update_yaxes(**get_axis_config("Fuel Flow (kg/h)"))
        fig1.write_image(os.path.join(run_dir, f'{save_prefix}_fuel_flow.png'), width=1200, height=800, scale=2)
        
        # 2. Thrust vs Drag
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=climb_time_min, y=thrust_kn, mode='lines', name='Thrust (Climb)',
            line=dict(color='darkblue', width=LineStyles.THICK), hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Thrust: %{y:.1f} kN<extra></extra>'))
        fig2.add_trace(go.Scatter(x=climb_time_min, y=drag_kn, mode='lines', name='Drag (Climb)',
            line=dict(color='lightblue', width=LineStyles.THICK, dash=LineStyles.DASH), hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Drag: %{y:.1f} kN<extra></extra>'))
        fig2.update_layout(**get_standard_layout("CLIMB PERFORMANCE - Thrust vs Drag", subtitle, height=600, width=900))
        fig2.update_xaxes(**get_axis_config("Time (min)")); fig2.update_yaxes(**get_axis_config("Force (kN)"))
        fig2.write_image(os.path.join(run_dir, f'{save_prefix}_thrust_drag.png'), width=1200, height=800, scale=2)
        
        # 3. Mass Evolution
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=climb_time_min, y=climb_mass_kg, mode='lines', name='Mass (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(65, 105, 225, 0.15)',
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Mass: %{y:.0f} kg<extra></extra>'))
        fig3.update_layout(**get_standard_layout("CLIMB PERFORMANCE - Mass Evolution", subtitle, height=600, width=900))
        # Set y-axis range to zoom in on mass changes for better visibility
        mass_min, mass_max = np.min(climb_mass_kg), np.max(climb_mass_kg)
        mass_margin = (mass_max - mass_min) * 0.2  # Add 20% margin
        fig3.update_xaxes(**get_axis_config("Time (min)")); 
        fig3.update_yaxes(**get_axis_config("Mass (kg)"), range=[mass_min - mass_margin, mass_max + mass_margin])
        fig3.write_image(os.path.join(run_dir, f'{save_prefix}_mass.png'), width=1200, height=800, scale=2)
        
        # 4. Lever Position
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=climb_time_min, y=climb_lever * 100, mode='lines', name='Lever Position (Climb)',
            line=dict(color='steelblue', width=LineStyles.THICK), hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Lever: %{y:.1f}%<extra></extra>'))
        fig4.add_hline(y=85, line_dash=LineStyles.DASH, line_color=Colors.WARNING, annotation_text="MCT Limit", annotation_position="top right")
        fig4.add_hline(y=100, line_dash=LineStyles.DOT, line_color=Colors.CRITICAL, annotation_text="Max Thrust", annotation_position="bottom right")
        fig4.update_layout(**get_standard_layout("CLIMB PERFORMANCE - Lever Position", subtitle, height=600, width=900))
        fig4.update_xaxes(**get_axis_config("Time (min)")); fig4.update_yaxes(**get_axis_config("Lever Position (%)"))
        fig4.write_image(os.path.join(run_dir, f'{save_prefix}_lever.png'), width=1200, height=800, scale=2)
        
        # 5. True Airspeed
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=climb_time_min, y=tas_ms, mode='lines', name='True Airspeed (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK), hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>TAS: %{y:.1f} m/s<extra></extra>'))
        fig5.update_layout(**get_standard_layout("CLIMB PERFORMANCE - True Airspeed", subtitle, height=600, width=900))
        fig5.update_xaxes(**get_axis_config("Time (min)")); fig5.update_yaxes(**get_axis_config("True Airspeed (m/s)"))
        fig5.write_image(os.path.join(run_dir, f'{save_prefix}_airspeed.png'), width=1200, height=800, scale=2)
        
        # 6. Cumulative Fuel
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=climb_time_min, y=climb_fuel_kg, mode='lines', name='Fuel Consumed (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(65, 105, 225, 0.15)',
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Fuel: %{y:.1f} kg<extra></extra>'))
        fig6.update_layout(**get_standard_layout("CLIMB PERFORMANCE - Cumulative Fuel Consumption", subtitle, height=600, width=900))
        fig6.update_xaxes(**get_axis_config("Time (min)")); fig6.update_yaxes(**get_axis_config("Cumulative Fuel (kg)"))
        fig6.write_image(os.path.join(run_dir, f'{save_prefix}_fuel.png'), width=1200, height=800, scale=2)
        
        print(f"[EXPORT] Individual climb plots saved as PNG to: {run_dir}")
    except Exception as e:
        print(f"[WARNING] Could not save individual climb plots: {e}")
    
    # Show the plot
    fig.show(config=config)
    
    return fig


# =========================================================================
# COMPUTATIONAL SUPPORT FOR VISUALIZATION
# =========================================================================

def compute_sep_grid_maxlever(aero, engine, ref_mass_kg: float,
                              mach_grid=None,
                              H_grid=None):
    """
    Compute Ps(M,h) at maximum lever for background visualization.
    Delegates to GridAndPlotting.compute_sep_grid_maxlever().
    
    This wrapper provides a module-level interface for computing specific excess
    power grids used in climb trajectory visualization backgrounds.
    
    Args:
        aero: PyAerodynamicsWrapper - aerodynamic model
        engine: EngineWrapper - propulsion model
        ref_mass_kg: float - reference mass for Ps computation
        mach_grid: Optional Mach number grid (defaults to aero.mach_grid)
        H_grid: Optional altitude grid (defaults to aero.alt_grid_m)
        
    Returns:
        tuple: (mach_grid, H_grid, Ps) where Ps is the specific excess power array
    """
    return GridAndPlotting.compute_sep_grid_maxlever(aero, engine, ref_mass_kg, mach_grid, H_grid)


def compute_full_envelope(aero, engine, mach_grid: np.ndarray, 
                         altitude_sched: np.ndarray, lever_grid: np.ndarray, mass_kg: float = None):
    """
    Compute 3D performance envelope J(M,h,δ) for visualization.
    Wrapper for ClimbingCore.compute_full_envelope().
    
    This function computes the fuel cost density J = ṁ/Ps across the entire
    feasible state space for 3D visualization and performance analysis.
    
    Args:
        aero: PyAerodynamicsWrapper - aerodynamic model
        engine: EngineWrapper - propulsion model
        mach_grid: np.ndarray - Mach number discretization
        altitude_sched: np.ndarray - altitude schedule
        lever_grid: np.ndarray - throttle lever discretization
        mass_kg: float - reference mass for envelope (default: INITIAL_MASS_KG)
        
    Returns:
        np.ndarray: J_envelope - 3D fuel cost density array
    """
    return ClimbingCore.compute_full_envelope(aero, engine, mach_grid, altitude_sched, lever_grid, mass_kg)