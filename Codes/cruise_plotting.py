# ========================================================================
# CRUISE PHASE VISUALIZATION MODULE
# ========================================================================
"""
Performance visualization for cruise phase analysis.

Visualization panels (3×2 grid):
    - Fuel flow: ṁ(t) [kg/h]
    - Forces: T(t), D(t) [kN]
    - Mass evolution: m(t) [kg]
    - Throttle: δ(t) [%]
    - Airspeed: V(t) [m/s]
    - Cumulative fuel: Σm_fuel(t) [kg]

Output: Interactive Plotly dashboards with individual panel exports.
"""

from __future__ import annotations
import numpy as np
import os
import warnings
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.io as pio
from typing import Optional, Callable, List, Dict, Any, Tuple, TYPE_CHECKING

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning, module='choreographer')
try:
    import logging
    logging.getLogger('choreographer').setLevel(logging.ERROR)
except:
    pass

# Plotly configuration
pio.renderers.default = "browser"

if TYPE_CHECKING:
    from cruise import CruiseResults

# Visualization styling configuration
from visualization_config import (
    Colors, Typography, Layout, LineStyles,
    get_standard_layout, get_axis_config,
    ExportConfig, get_or_create_run_directory
)

# ========================================================================
# SECTION 1: PERFORMANCE DASHBOARD
# ========================================================================

def plot_performance_2d(cruise_results: 'CruiseResults'):
    """
    Generate 6-panel performance dashboard for cruise phase.
    
    Dashboard structure (3×2 grid):
        Row 1: ṁ(t), [T(t), D(t)]
        Row 2: m(t), δ(t)
        Row 3: V(t), Σm_fuel(t)
    
    Output: Interactive HTML with individual panel PNG exports.
    
    Parameters:
        cruise_results: CruiseResults - cruise trajectory data
        
    Returns:
        Plotly figure object
    """
    
    # ────────────────────────────────────────────────────────────────────
    # Data Extraction and Unit Conversion
    # ────────────────────────────────────────────────────────────────────
    cruise_time_min = cruise_results.time_s / 60.0                  # t [min]
    cruise_distance_km = cruise_results.distance_km                 # s [km]
    cruise_fuel_flow_kgh = cruise_results.fuel_flow_kgps * 3600    # ṁ [kg/h]
    cruise_thrust_kn = cruise_results.thrust_total_N / 1000         # T [kN]
    cruise_drag_kn = cruise_results.drag_N / 1000                   # D [kN]
    cruise_mass_kg = cruise_results.mass_kg                         # m [kg]
    cruise_lever = cruise_results.lever_position * 100              # δ [%]
    cruise_fuel_consumed = cruise_results.fuel_consumed_kg          # Σm_fuel [kg]
    cruise_tas_ms = cruise_results.true_airspeed_mps                # V [m/s]
    
    # ════════════════════════════════════════════════════════════════════
    # Dashboard Layout: 3×2 Grid
    # ════════════════════════════════════════════════════════════════════
    fig = sp.make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '<b>Fuel Flow Rate ṁ(t)</b>', '<b>Forces T(t), D(t)</b>', 
            '<b>Mass Evolution m(t)</b>', '<b>Throttle δ(t)</b>',
            '<b>True Airspeed V(t)</b>', '<b>Cumulative Fuel Σm(t)</b>'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # ────────────────────────────────────────────────────────────────────
    # Panel 1: Fuel Flow Rate ṁ(t)
    # ────────────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=cruise_time_min,
            y=cruise_fuel_flow_kgh,
            mode='lines',
            name='Fuel Flow (Cruise)',
            line=dict(color=Colors.CRUISE, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(0, 128, 0, 0.15)',
            hovertemplate='<b>Cruise</b><br>Time: %{x:.1f} min<br>Fuel Flow: %{y:.1f} kg/h<extra></extra>'
        ),
        row=1, col=1
    )
    
    # ────────────────────────────────────────────────────────────────────
    # Panel 2: Forces T(t), D(t)
    # ────────────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=cruise_time_min,
            y=cruise_thrust_kn,
            mode='lines',
            name='Thrust (Cruise)',
            line=dict(color='darkgreen', width=LineStyles.THICK),
            hovertemplate='<b>Cruise</b><br>Time: %{x:.1f} min<br>Thrust: %{y:.1f} kN<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=cruise_time_min,
            y=cruise_drag_kn,
            mode='lines',
            name='Drag (Cruise)',
            line=dict(color='lightcoral', width=LineStyles.THICK, dash=LineStyles.DASH),
            hovertemplate='<b>Cruise</b><br>Time: %{x:.1f} min<br>Drag: %{y:.1f} kN<extra></extra>'
        ),
        row=1, col=2
    )
    
    # ────────────────────────────────────────────────────────────────────
    # Panel 3: Mass Evolution m(t)
    # ────────────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=cruise_time_min,
            y=cruise_mass_kg,  # Renamed for physics accuracy
            mode='lines',
            name='Mass (Cruise)',
            line=dict(color=Colors.CRUISE, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(0, 128, 0, 0.15)',
            hovertemplate='<b>Cruise</b><br>Time: %{x:.1f} min<br>Mass: %{y:.0f} kg<extra></extra>'
        ),
        row=2, col=1
    )
    
    # ────────────────────────────────────────────────────────────────────
    # Panel 4: Throttle Position δ(t)
    # ────────────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=cruise_time_min,
            y=cruise_lever,
            mode='lines',
            name='Lever Position (Cruise)',
            line=dict(color='olive', width=LineStyles.THICK),
            hovertemplate='<b>Cruise</b><br>Time: %{x:.1f} min<br>Lever: %{y:.1f}%<extra></extra>'
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
            x=cruise_time_min,
            y=cruise_tas_ms,
            mode='lines',
            name='True Airspeed (Cruise)',
            line=dict(color=Colors.CRUISE, width=LineStyles.THICK),
            hovertemplate='<b>Cruise</b><br>Time: %{x:.1f} min<br>TAS: %{y:.1f} m/s<extra></extra>'
        ),
        row=3, col=1
    )
    
    # ────────────────────────────────────────────────────────────────────
    # Panel 6: Cumulative Fuel Σm_fuel(t)
    # ────────────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=cruise_time_min,
            y=cruise_fuel_consumed,
            mode='lines',
            name='Fuel Consumed (Cruise)',
            line=dict(color=Colors.CRUISE, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(0, 128, 0, 0.15)',
            hovertemplate='<b>Cruise</b><br>Time: %{x:.1f} min<br>Fuel: %{y:.1f} kg<extra></extra>'
        ),
        row=3, col=2
    )
    
    # ════════════════════════════════════════════════════════════════════
    # Summary Statistics
    # ════════════════════════════════════════════════════════════════════
    total_time_min = cruise_time_min[-1] if len(cruise_time_min) > 0 else 0.0     # t_total [min]
    total_distance = cruise_distance_km[-1] if len(cruise_distance_km) > 0 else 0.0  # s_total [km]
    total_fuel = cruise_fuel_consumed[-1] if len(cruise_fuel_consumed) > 0 else 0.0  # Σm_fuel [kg]
    avg_fuel_flow = np.mean(cruise_fuel_flow_kgh)                                    # <ṁ> [kg/h]
    
    subtitle = (
        f"s: {total_distance:.0f} km | t: {total_time_min:.1f} min | "
        f"Δm_fuel: {total_fuel:.1f} kg | <ṁ>: {avg_fuel_flow:.0f} kg/h"
    )
    
    # ════════════════════════════════════════════════════════════════════
    # Layout Configuration
    # ════════════════════════════════════════════════════════════════════
    layout_config = get_standard_layout(
        "CRUISE PERFORMANCE ANALYSIS (2D)",
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
    # All panels use time as independent variable
    fig.update_xaxes(**get_axis_config("t [min]"), row=1, col=1)
    fig.update_yaxes(**get_axis_config("ṁ [kg/h]"), row=1, col=1)
    
    fig.update_xaxes(**get_axis_config("t [min]"), row=1, col=2)
    fig.update_yaxes(**get_axis_config("Force [kN]"), row=1, col=2)
    
    fig.update_xaxes(**get_axis_config("t [min]"), row=2, col=1)
    # Zoom to mass variation range for visibility
    mass_min, mass_max = np.min(cruise_mass_kg), np.max(cruise_mass_kg)
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
    config['toImageButtonOptions']['filename'] = 'cruise_performance_2d'
    
    # ════════════════════════════════════════════════════════════════════
    # Individual Panel Export (PNG)
    # ════════════════════════════════════════════════════════════════════
    try:
        run_dir = get_or_create_run_directory(phase="Cruise")
        save_prefix = "cruise_performance"
        
        # Panel 1: Fuel flow ṁ(t)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=cruise_time_min, y=cruise_fuel_flow_kgh, mode='lines', name='Fuel Flow (Cruise)',
            line=dict(color=Colors.CRUISE, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(0, 128, 0, 0.15)',
            hovertemplate='<b>Cruise</b><br>Time: %{x:.1f} min<br>Fuel Flow: %{y:.1f} kg/h<extra></extra>'))
        fig1.update_layout(**get_standard_layout("CRUISE PERFORMANCE - Fuel Flow Rate", subtitle, height=600, width=900))
        fig1.update_xaxes(**get_axis_config("t [min]")); fig1.update_yaxes(**get_axis_config("ṁ [kg/h]"))
        fig1.write_image(os.path.join(run_dir, f'{save_prefix}_fuel_flow.png'), width=1200, height=800, scale=2)
        
        # Panel 2: Forces T(t), D(t)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=cruise_time_min, y=cruise_thrust_kn, mode='lines', name='Thrust (Cruise)',
            line=dict(color='darkgreen', width=LineStyles.THICK), hovertemplate='<b>Cruise</b><br>Time: %{x:.1f} min<br>Thrust: %{y:.1f} kN<extra></extra>'))
        fig2.add_trace(go.Scatter(x=cruise_time_min, y=cruise_drag_kn, mode='lines', name='Drag (Cruise)',
            line=dict(color='lightcoral', width=LineStyles.THICK, dash=LineStyles.DASH), hovertemplate='<b>Cruise</b><br>Time: %{x:.1f} min<br>Drag: %{y:.1f} kN<extra></extra>'))
        fig2.update_layout(**get_standard_layout("CRUISE PERFORMANCE - Thrust vs Drag", subtitle, height=600, width=900))
        fig2.update_xaxes(**get_axis_config("t [min]")); fig2.update_yaxes(**get_axis_config("Force [kN]"))
        fig2.write_image(os.path.join(run_dir, f'{save_prefix}_thrust_drag.png'), width=1200, height=800, scale=2)
        
        # Panel 3: Mass m(t)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=cruise_time_min, y=cruise_mass_kg, mode='lines', name='Mass (Cruise)',  # Renamed for physics accuracy
            line=dict(color=Colors.CRUISE, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(0, 128, 0, 0.15)',
            hovertemplate='<b>Cruise</b><br>Time: %{x:.1f} min<br>Mass: %{y:.0f} kg<extra></extra>'))
        fig3.update_layout(**get_standard_layout("CRUISE PERFORMANCE - Mass Evolution", subtitle, height=600, width=900))
        mass_min, mass_max = np.min(cruise_mass_kg), np.max(cruise_mass_kg)
        mass_margin = (mass_max - mass_min) * 0.2
        fig3.update_xaxes(**get_axis_config("t [min]")); 
        fig3.update_yaxes(**get_axis_config("m [kg]"), range=[mass_min - mass_margin, mass_max + mass_margin])
        fig3.write_image(os.path.join(run_dir, f'{save_prefix}_mass.png'), width=1200, height=800, scale=2)
        
        # Panel 4: Throttle δ(t)
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=cruise_time_min, y=cruise_lever, mode='lines', name='Lever Position (Cruise)',
            line=dict(color='olive', width=LineStyles.THICK), hovertemplate='<b>Cruise</b><br>Time: %{x:.1f} min<br>Lever: %{y:.1f}%<extra></extra>'))
        fig4.add_hline(y=85, line_dash=LineStyles.DASH, line_color=Colors.WARNING, annotation_text="MCT Limit", annotation_position="top right")
        fig4.add_hline(y=100, line_dash=LineStyles.DOT, line_color=Colors.CRITICAL, annotation_text="Max Thrust", annotation_position="bottom right")
        fig4.update_layout(**get_standard_layout("CRUISE PERFORMANCE - Throttle Position", subtitle, height=600, width=900))
        fig4.update_xaxes(**get_axis_config("t [min]")); fig4.update_yaxes(**get_axis_config("δ [%]"))
        fig4.write_image(os.path.join(run_dir, f'{save_prefix}_lever.png'), width=1200, height=800, scale=2)
        
        # Panel 5: True airspeed V(t)
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=cruise_time_min, y=cruise_tas_ms, mode='lines', name='True Airspeed (Cruise)',
            line=dict(color=Colors.CRUISE, width=LineStyles.THICK), hovertemplate='<b>Cruise</b><br>Time: %{x:.1f} min<br>TAS: %{y:.1f} m/s<extra></extra>'))
        fig5.update_layout(**get_standard_layout("CRUISE PERFORMANCE - True Airspeed", subtitle, height=600, width=900))
        fig5.update_xaxes(**get_axis_config("t [min]")); fig5.update_yaxes(**get_axis_config("V [m/s]"))
        fig5.write_image(os.path.join(run_dir, f'{save_prefix}_airspeed.png'), width=1200, height=800, scale=2)
        
        # Panel 6: Cumulative fuel Σm_fuel(t)
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=cruise_time_min, y=cruise_fuel_consumed, mode='lines', name='Fuel Consumed (Cruise)',
            line=dict(color=Colors.CRUISE, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(0, 128, 0, 0.15)',
            hovertemplate='<b>Cruise</b><br>Time: %{x:.1f} min<br>Fuel: %{y:.1f} kg<extra></extra>'))
        fig6.update_layout(**get_standard_layout("CRUISE PERFORMANCE - Cumulative Fuel", subtitle, height=600, width=900))
        fig6.update_xaxes(**get_axis_config("t [min]")); fig6.update_yaxes(**get_axis_config("Σm_fuel [kg]"))
        fig6.write_image(os.path.join(run_dir, f'{save_prefix}_fuel.png'), width=1200, height=800, scale=2)
        
        print(f"[EXPORT] Individual panels saved to: {run_dir}")
    except Exception as e:
        print(f"[WARNING] Panel export failed: {e}")
    
    # Display interactive dashboard
    fig.show(config=config)
    
    return fig

def plot_cruise_performance_detailed(*args, **kwargs):
    """Deprecated wrapper. Use plot_performance_2d() instead."""
    import warnings
    warnings.warn("plot_cruise_performance_detailed() is deprecated, use plot_performance_2d() instead", DeprecationWarning, stacklevel=2)
    return plot_performance_2d(*args, **kwargs)
