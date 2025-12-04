# ========================================================================
# DESCENT PHASE VISUALIZATION MODULE
# ========================================================================
"""
Performance and trajectory visualization for descent phase analysis.

Visualization capabilities:
    - 3D mission trajectory: Complete flight profile (δ,M,h) space
    - 2D performance dashboard: Time-series of ṁ, T/D, m, δ, V, Σm_fuel
    - 3D cost space: J(δ,M,h) with optimal path and envelope limits

Output: Interactive Plotly dashboards with PNG exports.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Callable, List, Dict, Any, Tuple
import os
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning, module='choreographer')
try:
    import logging
    logging.getLogger('choreographer').setLevel(logging.ERROR)
except:
    pass

# Plotly configuration
pio.renderers.default = "browser"

# Aircraft parameters: ISA properties, Mach limits, reference geometry
from aircraft_config import isa_properties, a_from_altitude, G_C, M_MMO, S_REF_M2, INITIAL_MASS_KG, CL_MAX

# Mission phase data structures
from descent import DescentResults, calculate_min_descent_mach
from cruise import CruiseResults
from climb import MinFuelSchedule

# Visualization styling configuration
from visualization_config import (
    Colors, Typography, Layout, LineStyles,
    get_standard_layout, get_axis_config,
    ExportConfig, get_or_create_run_directory
)

# ========================================================================
# SECTION 1: 3D MISSION TRAJECTORY VISUALIZATION
# ========================================================================

def plot_complete_mission_3d(*args, **kwargs):
    """Deprecated wrapper. Use mission_summary.plot_complete_mission_3d() instead."""
    from mission_summary import plot_complete_mission_3d as mission_plot
    warnings.warn("descent_plotting.plot_complete_mission_3d() is deprecated, use mission_summary.plot_complete_mission_3d() instead", DeprecationWarning, stacklevel=2)
    return mission_plot(*args, **kwargs)

# ========================================================================
# SECTION 2: 2D PERFORMANCE DASHBOARD
# ========================================================================

def plot_performance_2d(descent_result: DescentResults,
                       save_html: Optional[str] = None):
    """
    Generate 6-panel performance dashboard for descent phase.
    
    Dashboard structure (3×2 grid):
        Row 1: ṁ(t), [T(t), D(t)]
        Row 2: m(t), δ(t)
        Row 3: V(t), Σm_fuel(t)
    
    Output: Interactive HTML with individual panel PNG exports.
    
    Parameters:
        descent_result: DescentResults - descent trajectory data
        save_html: str - optional custom save path
        
    Returns:
        Plotly figure object
    """
    # ════════════════════════════════════════════════════════════════════
    # Dashboard Layout: 3×2 Grid
    # ════════════════════════════════════════════════════════════════════
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '<b>Fuel Flow Rate ṁ(t)</b>',
            '<b>Forces T(t), D(t)</b>',
            '<b>Mass Evolution m(t)</b>',
            '<b>Throttle δ(t)</b>',
            '<b>True Airspeed V(t)</b>',
            '<b>Cumulative Fuel Σm(t)</b>'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # ────────────────────────────────────────────────────────────────────
    # Data Extraction and Unit Conversion
    # ────────────────────────────────────────────────────────────────────
    time_min = descent_result.time_s / 60.0                         # t [min]
    descent_rate_mpm = descent_result.descent_rate_mps * 60.0       # Ps [m/min]
    fuel_flow_kgh = descent_result.fuel_flow_kgps * 3600            # ṁ [kg/h]
    thrust_kn = descent_result.thrust_total_N / 1000                 # T [kN]
    drag_kn = descent_result.drag_N / 1000                           # D [kN]
    mass_kg = descent_result.mass_kg                                 # m [kg]
    lever_pct = descent_result.lever * 100                           # δ [%]
    cum_fuel_kg = descent_result.cumFuel_kg                          # Σm_fuel [kg]
    
    # True airspeed calculation: V = M·a(h)
    tas_ms = []
    for i in range(len(descent_result.mach)):
        a = a_from_altitude(float(descent_result.alt_m[i]))
        tas_ms.append(descent_result.mach[i] * a)
    tas_ms = np.array(tas_ms)
    
    # ────────────────────────────────────────────────────────────────────
    # Panel 1: Fuel Flow Rate ṁ(t)
    # ────────────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=time_min,
            y=fuel_flow_kgh,
            mode='lines',
            name='Fuel Flow (Descent)',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(220, 20, 60, 0.15)',
            hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Fuel Flow: %{y:.1f} kg/h<extra></extra>'
        ),
        row=1, col=1
    )
    
    # ────────────────────────────────────────────────────────────────────
    # Panel 2: Forces T(t), D(t)
    # ────────────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=time_min,
            y=thrust_kn,
            mode='lines',
            name='Thrust (Descent)',
            line=dict(color='darkred', width=LineStyles.THICK),
            hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Thrust: %{y:.1f} kN<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=time_min,
            y=drag_kn,
            mode='lines',
            name='Drag (Descent)',
            line=dict(color='salmon', width=LineStyles.THICK, dash=LineStyles.DASH),
            hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Drag: %{y:.1f} kN<extra></extra>'
        ),
        row=1, col=2
    )
    
    # ────────────────────────────────────────────────────────────────────
    # Panel 3: Mass Evolution m(t)
    # ────────────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=time_min,
            y=mass_kg,
            mode='lines',
            name='Mass (Descent)',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(220, 20, 60, 0.15)',
            hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Mass: %{y:.0f} kg<extra></extra>'
        ),
        row=2, col=1
    )
    
    # ────────────────────────────────────────────────────────────────────
    # Panel 4: Throttle Position δ(t)
    # ────────────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=time_min,
            y=lever_pct,
            mode='lines',
            name='Lever Position (Descent)',
            line=dict(color='firebrick', width=LineStyles.THICK),
            hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Lever: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Reference lines: δ_MCT = 85%, δ_max = 100%
    fig.add_hline(y=85, line_dash=LineStyles.DASH, line_color=Colors.WARNING, 
                  annotation_text="MCT (85%)", annotation_position="top right",
                  row=2, col=2)
    fig.add_hline(y=100, line_dash=LineStyles.DOT, line_color=Colors.CRITICAL,
                  annotation_text="Max (100%)", annotation_position="bottom right",
                  row=2, col=2)
    
    # ────────────────────────────────────────────────────────────────────
    # Panel 5: True Airspeed V(t)
    # ────────────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=time_min,
            y=tas_ms,
            mode='lines',
            name='True Airspeed (Descent)',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK),
            hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>TAS: %{y:.1f} m/s<extra></extra>'
        ),
        row=3, col=1
    )
    
    # ────────────────────────────────────────────────────────────────────
    # Panel 6: Cumulative Fuel Σm_fuel(t)
    # ────────────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=time_min,
            y=cum_fuel_kg,
            mode='lines',
            name='Fuel Consumed (Descent)',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(220, 20, 60, 0.15)',
            hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Fuel: %{y:.2f} kg<extra></extra>'
        ),
        row=3, col=2
    )
    
    # ════════════════════════════════════════════════════════════════════
    # Summary Statistics and Layout
    # ════════════════════════════════════════════════════════════════════
    summary = descent_result.get_summary_dict()
    
    subtitle = (
        f"Δh: {summary['descent_altitude_change_m']:.0f} m | "
        f"t: {summary['descent_time_minutes']:.1f} min | "
        f"Δm_fuel: {summary['descent_fuel_kg']:.2f} kg | "
        f"<Ps>: {summary['avg_descent_rate_mpm']:.0f} m/min"
    )
    
    layout_config = get_standard_layout(
        "DESCENT PERFORMANCE ANALYSIS (2D)",
        subtitle,
        height=Layout.STANDARD_HEIGHT,
        width=Layout.STANDARD_WIDTH
    )
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
    
    # ════════════════════════════════════════════════════════════════════
    # Axis Configuration
    # ════════════════════════════════════════════════════════════════════
    fig.update_xaxes(**get_axis_config("t [min]"), row=1, col=1)
    fig.update_yaxes(**get_axis_config("ṁ [kg/h]"), row=1, col=1)
    
    fig.update_xaxes(**get_axis_config("t [min]"), row=1, col=2)
    fig.update_yaxes(**get_axis_config("Force [kN]"), row=1, col=2)
    
    fig.update_xaxes(**get_axis_config("t [min]"), row=2, col=1)
    # Zoom to mass variation range
    mass_min, mass_max = np.min(mass_kg), np.max(mass_kg)
    mass_margin = (mass_max - mass_min) * 0.2
    fig.update_yaxes(**get_axis_config("m [kg]"),
                     range=[mass_min - mass_margin, mass_max + mass_margin], 
                     row=2, col=1)
    
    fig.update_xaxes(**get_axis_config("t [min]"), row=2, col=2)
    fig.update_yaxes(**get_axis_config("δ [%]"), row=2, col=2)
    
    fig.update_xaxes(**get_axis_config("t [min]"), row=3, col=1)
    fig.update_yaxes(**get_axis_config("V [m/s]"), row=3, col=1)
    
    fig.update_xaxes(**get_axis_config("t [min]"), row=3, col=2)
    fig.update_yaxes(**get_axis_config("Σm_fuel [kg]"), row=3, col=2)
    
    # ════════════════════════════════════════════════════════════════════
    # Export Configuration
    # ════════════════════════════════════════════════════════════════════
    config = ExportConfig.get_plotly_config()
    config['toImageButtonOptions']['filename'] = 'descent_performance_2d'
    
    # ════════════════════════════════════════════════════════════════════
    # Individual Panel Export (PNG)
    # ════════════════════════════════════════════════════════════════════
    try:
        run_dir = get_or_create_run_directory(phase="Descent")
        save_prefix = "descent_performance"
        summary = descent_result.get_summary_dict()
        subtitle_text = (
            f"Δh: {summary['descent_altitude_change_m']:.0f} m | "
            f"t: {summary['descent_time_minutes']:.1f} min | "
            f"Δm_fuel: {summary['descent_fuel_kg']:.2f} kg | "
            f"<Ps>: {summary['avg_descent_rate_mpm']:.0f} m/min"
        )
        
        # Panel 1: Fuel flow ṁ(t)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=time_min, y=fuel_flow_kgh, mode='lines', name='Fuel Flow (Descent)',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(220, 20, 60, 0.15)',
            hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Fuel Flow: %{y:.1f} kg/h<extra></extra>'))
        fig1.update_layout(**get_standard_layout("DESCENT PERFORMANCE - Fuel Flow Rate", subtitle_text, height=600, width=900))
        fig1.update_xaxes(**get_axis_config("t [min]")); fig1.update_yaxes(**get_axis_config("ṁ [kg/h]"))
        fig1.write_image(os.path.join(run_dir, f'{save_prefix}_fuel_flow.png'), width=1200, height=800, scale=2)
        
        # Panel 2: Forces T(t), D(t)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=time_min, y=thrust_kn, mode='lines', name='Thrust (Descent)',
            line=dict(color='darkred', width=LineStyles.THICK), hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Thrust: %{y:.1f} kN<extra></extra>'))
        fig2.add_trace(go.Scatter(x=time_min, y=drag_kn, mode='lines', name='Drag (Descent)',
            line=dict(color='salmon', width=LineStyles.THICK, dash=LineStyles.DASH), hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Drag: %{y:.1f} kN<extra></extra>'))
        fig2.update_layout(**get_standard_layout("DESCENT PERFORMANCE - Forces", subtitle_text, height=600, width=900))
        fig2.update_xaxes(**get_axis_config("t [min]")); fig2.update_yaxes(**get_axis_config("Force [kN]"))
        fig2.write_image(os.path.join(run_dir, f'{save_prefix}_thrust_drag.png'), width=1200, height=800, scale=2)
        
        # Panel 3: Mass m(t)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=time_min, y=mass_kg, mode='lines', name='Mass (Descent)',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(220, 20, 60, 0.15)',
            hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Mass: %{y:.0f} kg<extra></extra>'))
        fig3.update_layout(**get_standard_layout("DESCENT PERFORMANCE - Mass Evolution", subtitle_text, height=600, width=900))
        mass_min, mass_max = np.min(mass_kg), np.max(mass_kg)
        mass_margin = (mass_max - mass_min) * 0.2
        fig3.update_xaxes(**get_axis_config("t [min]")); 
        fig3.update_yaxes(**get_axis_config("m [kg]"), range=[mass_min - mass_margin, mass_max + mass_margin])
        fig3.write_image(os.path.join(run_dir, f'{save_prefix}_mass.png'), width=1200, height=800, scale=2)
        
        # Panel 4: Throttle δ(t)
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=time_min, y=lever_pct, mode='lines', name='Lever Position (Descent)',
            line=dict(color='firebrick', width=LineStyles.THICK), hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Lever: %{y:.1f}%<extra></extra>'))
        fig4.add_hline(y=85, line_dash=LineStyles.DASH, line_color=Colors.WARNING, annotation_text="MCT Limit", annotation_position="top right")
        fig4.add_hline(y=100, line_dash=LineStyles.DOT, line_color=Colors.CRITICAL, annotation_text="Max Thrust", annotation_position="bottom right")
        fig4.update_layout(**get_standard_layout("DESCENT PERFORMANCE - Throttle", subtitle_text, height=600, width=900))
        fig4.update_xaxes(**get_axis_config("t [min]")); fig4.update_yaxes(**get_axis_config("δ [%]"))
        fig4.write_image(os.path.join(run_dir, f'{save_prefix}_lever.png'), width=1200, height=800, scale=2)
        
        # Panel 5: True airspeed V(t)
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=time_min, y=tas_ms, mode='lines', name='True Airspeed (Descent)',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK), hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>TAS: %{y:.1f} m/s<extra></extra>'))
        fig5.update_layout(**get_standard_layout("DESCENT PERFORMANCE - True Airspeed", subtitle_text, height=600, width=900))
        fig5.update_xaxes(**get_axis_config("t [min]")); fig5.update_yaxes(**get_axis_config("V [m/s]"))
        fig5.write_image(os.path.join(run_dir, f'{save_prefix}_airspeed.png'), width=1200, height=800, scale=2)
        
        # Panel 6: Cumulative fuel Σm_fuel(t)
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=time_min, y=cum_fuel_kg, mode='lines', name='Fuel Consumed (Descent)',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(220, 20, 60, 0.15)',
            hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Fuel: %{y:.2f} kg<extra></extra>'))
        fig6.update_layout(**get_standard_layout("DESCENT PERFORMANCE - Cumulative Fuel", subtitle_text, height=600, width=900))
        fig6.update_xaxes(**get_axis_config("t [min]")); fig6.update_yaxes(**get_axis_config("Σm_fuel [kg]"))
        fig6.write_image(os.path.join(run_dir, f'{save_prefix}_fuel.png'), width=1200, height=800, scale=2)
        
        print(f"[EXPORT] Individual panels saved to: {run_dir}")
    except Exception as e:
        print(f"[WARNING] Panel export failed: {e}")
    
    # Optional custom save
    if save_html:
        fig.write_html(save_html, config=config)
    
    # Display interactive dashboard
    fig.show(config=config)
    
    return fig

# ========================================================================
# SECTION 3: 3D COST SPACE VISUALIZATION
# ========================================================================

def plot_3d_cost_space(mach_grid: np.ndarray, altitude_sched: np.ndarray, 
                      lever_grid: np.ndarray, J_grid_3d: np.ndarray,
                      min_path: Optional[Dict] = None, 
                      title: Optional[str] = None,
                      mass_kg: Optional[float] = None):
    """
    Visualize fuel cost density J(δ,M,h) in 3D state space.
    
    Visualization: Cost landscape J = ṁ/|Ps| with optimal path X*(t) overlay.
    
    Components:
        - Scatter points: J values at feasible (δ,M,h) grid points
        - Optimal path: X*(t) from DP optimization (if provided)
        - Envelope limits: M_MMO, M_stall(h), h_ceiling surfaces
        - Interactive controls: Camera presets, data filtering
    
    Parameters:
        mach_grid: np.ndarray - M_i discretization
        altitude_sched: np.ndarray - h_k discretization (descending)
        lever_grid: np.ndarray - δ_j discretization
        J_grid_3d: np.ndarray (I×K×L) - fuel cost density [kg/m]
        min_path: dict - optimal path {'mach', 'alt', 'lever'} (optional)
        title: str - custom title (optional)
        mass_kg: m_0 [kg] - reference mass for M_stall calculation (optional)
        
    Returns:
        Plotly figure object
    """
    if mass_kg is None:
        mass_kg = INITIAL_MASS_KG
    
    # ────────────────────────────────────────────────────────────────────
    # Data Preparation
    # ────────────────────────────────────────────────────────────────────
    # Generate 3D meshgrid: (δ,M,h) space
    M, H, L = np.meshgrid(mach_grid, altitude_sched, lever_grid, indexing='ij')
    
    # Flatten for scatter plot: Axis assignment δ→x, M→y, h→z
    x = L.flatten()  # δ (X-axis)
    y = M.flatten()  # M (Y-axis) 
    z = H.flatten()  # h (Z-axis)
    J_flat = J_grid_3d.flatten()
    
    # Filter: plot only feasible points (finite J)
    mask = np.isfinite(J_flat)
    x, y, z, J_flat = x[mask], y[mask], z[mask], J_flat[mask]
    
    # Hover tooltip construction
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
    
    # ────────────────────────────────────────────────────────────────────
    # Flight Envelope Constraints
    # ────────────────────────────────────────────────────────────────────
    # Constraint 1: M ≤ M_MMO (vertical plane)
    lever_range = np.linspace(0, 1.0, 10)   # δ ∈ [0,1]
    alt_range = np.linspace(altitude_sched[-1], altitude_sched[0], 10)  # h: low → high
    L_mmo, H_mmo = np.meshgrid(lever_range, alt_range)
    M_mmo = np.full_like(L_mmo, M_MMO)  # M_MMO = 0.9392
    
    fig.add_trace(go.Surface(
        x=L_mmo,
        y=M_mmo,
        z=H_mmo,
        colorscale=[[0, Colors.ENVELOPE_LIMIT], [1, Colors.ENVELOPE_LIMIT]],
        opacity=0.45,
        showscale=False,
        name=f'Max Engine Mach Limit (M={M_MMO:.3f})'
    ))
    
    # Constraint 2: M ≥ M_stall(h,m) - Stall boundary
    def _compute_mstall_curve_descent():
        """Compute M_stall(h) from lift balance: L = mg at CL_max."""
        W = mass_kg * G_C
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
    
    M_stall = _compute_mstall_curve_descent()
    if np.isfinite(M_stall).any():
        # Stall surface: M = M_stall(h) plane
        lever_range = np.linspace(0, 1.0, 10)  # Full lever range
        alt_range = np.linspace(altitude_sched[-1], altitude_sched[0], 10)
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
                name='Flight Envelope Limit (Stall)'
            ))
    
    # Operating envelope: Region between M_stall(h) and M_MMO
    if np.isfinite(M_stall).any():
        cond = np.isfinite(M_stall) & (M_stall < M_MMO)  # M_MMO now set to 0.9392 from engine envelope analysis
        if np.any(cond):
            # Create operating envelope boundary lines at different lever positions
            for lever_val in [0.0, 0.5, 1.0]:  # Show at different lever positions (full range)
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
                    y=(M_MMO * np.ones_like(M_stall[cond])),
                    z=altitude_sched[cond],
                    mode='lines',
                    line=dict(color=Colors.ENVELOPE_LIMIT, width=LineStyles.THICK),
                    showlegend=False
                ))
    
    # Constraint 3: h ≤ h_max - Service ceiling limit (horizontal plane)
    MAX_SERVICE_CEILING_M = 13994.1  # h_max [m] at δ=1.0, M=0.900
    if MAX_SERVICE_CEILING_M <= (altitude_sched[0] if len(altitude_sched) > 0 else 15000) + 1000:
        lever_range_ceiling = np.linspace(0, 1.0, 10)
        mach_range_ceiling = np.linspace(mach_grid[0] if len(mach_grid) > 0 else 0.1, M_MMO, 10)
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
    
    # Constraint 4: M ≥ M_min (vertical plane)
    MIN_ENGINE_MACH = 0.200  # M_min = 0.200
    if MIN_ENGINE_MACH >= (mach_grid[0] if len(mach_grid) > 0 else 0.0):
        lever_range_min = np.linspace(0, 1.0, 10)
        alt_range_min = np.linspace(altitude_sched[-1], altitude_sched[0], 10)
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
    
    # Add J values scatter plot (orange/red for descent)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=Colors.J_VALUES_DESCENT,
            opacity=0.7,
            line=dict(width=0)
        ),
        name='J values (Descent)',
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
            line=dict(color=Colors.OPTIMAL_PATH_DESCENT, width=6),
            marker=dict(size=6, color=Colors.OPTIMAL_PATH_DESCENT),
            name='Optimal Descent Path',
            hovertemplate='<b>Optimal Path</b><br>Lever: %{x:.3f}<br>Mach: %{y:.3f}<br>Alt: %{z:.1f} m<extra></extra>'
        ))

    # Enhanced layout with user-friendly features (same as climb)
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
                gridcolor='rgba(128,128,128,0.3)',
                range=[0, 1.05]  # Full lever range (same as climb)
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
            text=title if title is not None else '3D Visualization of Descent J (Fuel/Energy)',
            font=dict(size=Typography.MAIN_TITLE_SIZE, family=Typography.FONT_FAMILY),
            x=0.5
        ),
        # Add control buttons (same as climb)
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
                        args=[{"visible": [True if trace.name and "Optimal" in trace.name else False for trace in fig.data]}],
                        label="Optimal Path Only",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [True if trace.name and ("MMO Limit" in trace.name or "Flight Envelope" in trace.name or "Operating Envelope" in trace.name) else False for trace in fig.data]}],
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
        # Add margin for controls
        margin=dict(l=0, r=Layout.MARGIN_RIGHT, t=Layout.MARGIN_TOP, b=Layout.MARGIN_BOTTOM)
    )

    # Set template
    fig.update_layout(template="plotly_white")

    # Write to HTML file with enhanced features using standard export config
    export_config = ExportConfig.get_plotly_config()
    export_config['toImageButtonOptions']['filename'] = 'descent_3d_plot_export'
    export_config['modeBarButtonsToAdd'] = [
        "drawline", "drawopenpath", "drawclosedpath",
        "drawcircle", "drawrect", "eraseshape"
    ]
    
    # Save to Descent folder (both HTML for interaction and PNG for static)
    descent_dir = get_or_create_run_directory(phase="Descent")
    output_path_html = os.path.join(descent_dir, 'descent_J_3d_plot.html')
    output_path_png = os.path.join(descent_dir, 'descent_J_3d_plot.png')
    
    pio.write_html(
        fig, 
        file=output_path_html, 
        auto_open=True,
        config=export_config
    )
    
    # Also save as PNG
    try:
        fig.write_image(output_path_png, width=1600, height=1200, scale=2)
        print(f"[EXPORT] Descent 3D J plot saved to: {output_path_html} (interactive) and {output_path_png} (PNG)")
    except Exception as e:
        print(f"[EXPORT] Descent 3D J plot saved to: {output_path_html} (HTML only)")
        print(f"[WARNING] Could not save PNG version: {e}")
    
    return fig

# ========================================================================
# SECTION 4: BACKWARD COMPATIBILITY
# ========================================================================

def plot_complete_mission_3d_interactive(*args, **kwargs):
    """Deprecated wrapper. Use plot_complete_mission_3d() instead."""
    import warnings
    warnings.warn("plot_complete_mission_3d_interactive() is deprecated, use plot_complete_mission_3d() instead", DeprecationWarning, stacklevel=2)
    return plot_complete_mission_3d(*args, **kwargs)

def plot_descent_trajectory_interactive(*args, **kwargs):
    """Deprecated wrapper. Use plot_performance_2d() instead."""
    import warnings
    warnings.warn("plot_descent_trajectory_interactive() is deprecated, use plot_performance_2d() instead", DeprecationWarning, stacklevel=2)
    return plot_performance_2d(*args, **kwargs)

def plot_descent_J_3d_plotly(*args, **kwargs):
    """Deprecated wrapper. Use plot_3d_cost_space() instead."""
    import warnings
    warnings.warn("plot_descent_J_3d_plotly() is deprecated, use plot_3d_cost_space() instead", DeprecationWarning, stacklevel=2)
    return plot_3d_cost_space(*args, **kwargs)
