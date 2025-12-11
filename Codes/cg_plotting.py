# ========================================================================
# CENTER OF GRAVITY VISUALIZATION MODULE
# ========================================================================
"""
Total aircraft CG movement analysis and visualization during mission.

Visualization capabilities:
    - Aircraft x_CG evolution: x_CG,total(Δm_fuel), x_CG,total(m_total)
    - Tank fuel levels: m_i(t) for tanks i=0..4
    - Aircraft CG range: [x_CG,min, x_CG,max]
    - Fuel tank CG evolution: x_CG,fuel(Δm_fuel) showing fuel distribution shift
    - Current tank state: m_i,current

Output: Interactive Plotly dashboard with scientific styling.
"""

from __future__ import annotations
import numpy as np
import os
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Optional

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning, module='choreographer')
try:
    import logging
    logging.getLogger('choreographer').setLevel(logging.ERROR)
except:
    pass

# Plotly configuration
pio.renderers.default = "browser"

# CG system interface: fuel distribution and history
from cg_x_calculation import (
    FuelSystem, get_fuel_tank_status, TANK_NAMES, TANK_CG_POSITIONS
)
from aircraft_config import ZERO_FUEL_CG_X

# Visualization styling configuration
from visualization_config import (
    Colors, Typography, Layout, LineStyles,
    get_or_create_run_directory, ExportConfig, get_standard_layout, get_axis_config
)
# ========================================================================
# SECTION 1: PLOTTING UTILITIES
# ========================================================================

class CGPlotter:
    """
    CG visualization helper for dashboard generation.
    
    Methods:
        - _get_grouped_tank_fuel: Aggregate tank groups for plotting
        - Interfaces with FuelSystem for history data access
    
    Used internally by plot_cg_analysis() for subplot construction.
    """
    
    def __init__(self, tank_system: Optional[FuelSystem] = None):
        """
        Initialize plotter with fuel system reference.
        
        Parameters:
            tank_system: FuelSystem - fuel system instance (optional, uses global)
        """
        if tank_system is None:
            from cg_x_calculation import _get_fuel_system
            tank_system = _get_fuel_system()
        
        self.tank_system = tank_system
    
    def _get_grouped_tank_fuel(self) -> dict:
        """
        Aggregate tank fuel levels by spatial grouping.
        
        Tank groups:
            Outer: {1,3} - m_outer(t) = m_1(t) + m_3(t)
            Inner: {0,2} - m_inner(t) = m_0(t) + m_2(t)
            Center: {4} - m_center(t) = m_4(t)
        
        Returns:
            {'outer': array, 'inner': array, 'center': array} [kg]
        """
        # Tank grouping by spatial location
        outer_tanks = [1, 3]  # Tanks 1,3: Outer Left, Outer Right
        inner_tanks = [0, 2]  # Tanks 0,2: Inner Left, Inner Right
        center_tank = [4]     # Tank 4: Center Wing
        
        def sum_tank_fuel(tank_ids):
            """Sum time series: m_group(t) = Σ m_i(t) for i ∈ group."""
            if len(tank_ids) == 0:
                return None
            result = None
            for tank_id in tank_ids:
                if len(self.tank_system.tank_fuel_history[tank_id]) > 0:
                    tank_fuel = np.array(self.tank_system.tank_fuel_history[tank_id])
                    if result is None:
                        result = tank_fuel.copy()
                    else:
                        min_len = min(len(result), len(tank_fuel))
                        result[:min_len] += tank_fuel[:min_len]
            
            # Diagnostic: check array length consistency
            if result is not None and len(tank_ids) > 1:
                lengths = [len(self.tank_system.tank_fuel_history[tid]) for tid in tank_ids 
                          if len(self.tank_system.tank_fuel_history[tid]) > 0]
                if len(set(lengths)) > 1:
                    print(f"[CG_PLOTTER] WARNING: Array length mismatch: {lengths}")
            
            return result
        
        return {
            'outer': sum_tank_fuel(outer_tanks),
            'inner': sum_tank_fuel(inner_tanks),
            'center': sum_tank_fuel(center_tank)
        }
# ========================================================================
# SECTION 2: DASHBOARD GENERATION
# ========================================================================

def plot_cg_analysis(save_plots: bool = True, show_plots: bool = True):
    """
    Generate comprehensive total aircraft CG analysis dashboard.
    
    Dashboard structure (3×2 grid):
        Row 1: Aircraft x_CG(Δm_fuel), Aircraft x_CG(m_total)
        Row 2: m_i(Δm_fuel) by tank group, Aircraft CG range [x_min, x_max]
        Row 3: Current tank levels (bar chart), Fuel tank CG evolution
    
    Note: CG values shown are total aircraft CG (empty aircraft + fuel), not just fuel tanks.
    
    Output: Interactive HTML with Plotly (optional PNG export).
    
    Parameters:
        save_plots: Boolean - save to disk
        show_plots: Boolean - display in browser
    """
    print("\n[CG_PLOTTER] Generating CG analysis dashboard...")
    
    plotter = CGPlotter()
    
    if plotter.tank_system is None or len(plotter.tank_system.cg_history) == 0:
        print("[CG_PLOTTER] No history data. Execute mission simulation first.")
        return None
    
    # Extract time-series data
    fuel_consumed = np.array(plotter.tank_system.fuel_consumed_history)  # Δm_fuel(t) [kg]
    cg_x = np.array(plotter.tank_system.cg_history)                      # x_CG,total(t) [m] - total aircraft CG
    weight_history = plotter.tank_system.weight_history                  # m_total(t) [kg]
    fuel_remaining = np.array(plotter.tank_system.fuel_remaining_history)  # m_fuel(t) [kg]
    
    # Compute fuel-only CG evolution (CG of fuel tanks, excluding empty aircraft)
    fuel_tank_cg_history = []
    for i in range(len(fuel_remaining)):
        if fuel_remaining[i] > 0:
            # Compute weighted average of fuel tank CG positions
            weighted_sum = 0.0
            total_fuel_mass = 0.0
            for tank_id in range(5):
                if i < len(plotter.tank_system.tank_fuel_history[tank_id]):
                    tank_fuel = plotter.tank_system.tank_fuel_history[tank_id][i]
                    if tank_fuel > 0:
                        weighted_sum += tank_fuel * TANK_CG_POSITIONS[tank_id]
                        total_fuel_mass += tank_fuel
            
            if total_fuel_mass > 0:
                fuel_tank_cg_history.append(weighted_sum / total_fuel_mass)
            else:
                fuel_tank_cg_history.append(ZERO_FUEL_CG_X)
        else:
            fuel_tank_cg_history.append(ZERO_FUEL_CG_X)
    
    fuel_tank_cg = np.array(fuel_tank_cg_history)  # x_CG,fuel(t) [m] - fuel tanks CG only
    
    # CG statistics
    total_fuel = fuel_consumed[-1] if len(fuel_consumed) > 0 else 0.0
    cg_min = np.min(cg_x) if len(cg_x) > 0 else ZERO_FUEL_CG_X            # x_CG,min [m]
    cg_max = np.max(cg_x) if len(cg_x) > 0 else ZERO_FUEL_CG_X            # x_CG,max [m]
    cg_range = cg_max - cg_min                                             # Δx_CG [m]
    cg_initial = cg_x[0] if len(cg_x) > 0 else ZERO_FUEL_CG_X             # x_CG,0 [m]
    cg_final = cg_x[-1] if len(cg_x) > 0 else ZERO_FUEL_CG_X              # x_CG,f [m]
    cg_shift = cg_final - cg_initial                                       # Δx_CG,travel [m]
    
    # Calculate focused y-axis range for CG plots (with padding for better visibility)
    # Add 10% padding on each side to ensure data is clearly visible
    cg_padding = max(cg_range * 0.1, 0.001)  # At least 1mm padding, or 10% of range
    cg_y_min = cg_min - cg_padding
    cg_y_max = cg_max + cg_padding
    
    # Mass statistics
    weight_initial = weight_history[0] if len(weight_history) > 0 else 0.0    # m_0 [kg]
    weight_final = weight_history[-1] if len(weight_history) > 0 else 0.0     # m_f [kg]
    weight_loss = weight_initial - weight_final if len(weight_history) > 0 else 0.0  # Δm [kg]
    
    # ════════════════════════════════════════════════════════════════════
    # Dashboard Layout: 3×2 Grid
    # ════════════════════════════════════════════════════════════════════
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '<b>Aircraft x_CG vs Fuel Consumed</b>',
            '<b>Aircraft x_CG vs Aircraft Mass</b>',
            '<b>Tank Fuel Evolution (Grouped)</b>',
            '<b>Aircraft CG Range Analysis</b>',
            '<b>Current Tank Fuel Levels</b>',
            '<b>Fuel Tank CG Evolution</b>'
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.12,
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "scatter"}]
        ]
    )
    
    # ────────────────────────────────────────────────────────────────────
    # Plot 1: x_CG,total(Δm_fuel) - Total aircraft CG evolution vs fuel consumed
    # ────────────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=fuel_consumed,
            y=cg_x,
            mode='lines',
            name='Aircraft CG_X',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(65, 105, 225, 0.15)',
            showlegend=True,
            hovertemplate='<b>Aircraft CG</b><br>Fuel Consumed: %{x:.1f} kg<br>Aircraft CG_X: %{y:.3f} m<extra></extra>'
        ),
        row=1, col=1
    )
    fig.add_hline(
        y=ZERO_FUEL_CG_X,
        line_dash="dash",
        line_color=Colors.WARNING,
        line_width=LineStyles.MEDIUM,
        annotation_text=f'Zero Fuel CG ({ZERO_FUEL_CG_X} m)',
        annotation_position="right",
        annotation_font_size=10,
        row=1, col=1
    )
    
    # ────────────────────────────────────────────────────────────────────
    # Plot 2: x_CG,total(m_total) - Total aircraft CG evolution vs total mass
    # ────────────────────────────────────────────────────────────────────
    if len(weight_history) > 0:
        min_len = min(len(cg_x), len(weight_history))
        weight = np.array(weight_history[:min_len])
        cg_x_aligned = cg_x[:min_len]
        
        fig.add_trace(
            go.Scatter(
                x=weight,
                y=cg_x_aligned,
                mode='lines',
                name='Aircraft CG_X',
                line=dict(color=Colors.CRUISE, width=LineStyles.THICK),
                fill='tozeroy',
                fillcolor='rgba(0, 128, 0, 0.15)',
                showlegend=False,
                hovertemplate='<b>Aircraft CG</b><br>Mass: %{x:.1f} kg<br>Aircraft CG_X: %{y:.3f} m<extra></extra>'
            ),
            row=1, col=2
        )
        fig.add_hline(
            y=ZERO_FUEL_CG_X,
            line_dash="dash",
            line_color=Colors.WARNING,
            line_width=LineStyles.MEDIUM,
            annotation_text=f'Zero Fuel CG ({ZERO_FUEL_CG_X} m)',
            annotation_position="right",
            annotation_font_size=10,
            row=1, col=2
        )
    
    # ────────────────────────────────────────────────────────────────────
    # Plot 3: m_group(Δm_fuel) - Tank group fuel evolution
    # ────────────────────────────────────────────────────────────────────
    grouped_fuel = plotter._get_grouped_tank_fuel()
    
    # Color mapping by tank spatial group
    category_colors = {
        'outer': Colors.CRUISE,    # Green: Outer tanks {1,3}
        'inner': Colors.CLIMB,     # Blue: Inner tanks {0,2}
        'center': Colors.DESCENT   # Red: Center tank {4}
    }
    
    # Outer tanks time series
    if grouped_fuel['outer'] is not None:
        min_len = min(len(fuel_consumed), len(grouped_fuel['outer']))
        outer_cg_avg = np.mean([TANK_CG_POSITIONS[1], TANK_CG_POSITIONS[3]])
        fig.add_trace(
            go.Scatter(
                x=fuel_consumed[:min_len],
                y=grouped_fuel['outer'][:min_len],
                mode='lines',
                name='Outer Tanks',
                line=dict(color=category_colors['outer'], width=LineStyles.THICK),
                showlegend=True,
                hovertemplate=f'<b>Outer Tanks</b><br>Fuel Consumed: %{{x:.1f}} kg<br>Total Fuel: %{{y:.1f}} kg<br>Avg CG Position: {outer_cg_avg:.2f} m<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Inner tanks time series
    if grouped_fuel['inner'] is not None:
        min_len = min(len(fuel_consumed), len(grouped_fuel['inner']))
        inner_cg_avg = np.mean([TANK_CG_POSITIONS[0], TANK_CG_POSITIONS[2]])
        fig.add_trace(
            go.Scatter(
                x=fuel_consumed[:min_len],
                y=grouped_fuel['inner'][:min_len],
                mode='lines',
                name='Inner Tanks',
                line=dict(color=category_colors['inner'], width=LineStyles.THICK),
                showlegend=True,
                hovertemplate=f'<b>Inner Tanks</b><br>Fuel Consumed: %{{x:.1f}} kg<br>Total Fuel: %{{y:.1f}} kg<br>Avg CG Position: {inner_cg_avg:.2f} m<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Center tank time series
    if grouped_fuel['center'] is not None:
        min_len = min(len(fuel_consumed), len(grouped_fuel['center']))
        center_cg = TANK_CG_POSITIONS[4]
        fig.add_trace(
            go.Scatter(
                x=fuel_consumed[:min_len],
                y=grouped_fuel['center'][:min_len],
                mode='lines',
                name='Center Tank',
                line=dict(color=category_colors['center'], width=LineStyles.THICK),
                showlegend=True,
                hovertemplate=f'<b>Center Tank</b><br>Fuel Consumed: %{{x:.1f}} kg<br>Tank Fuel: %{{y:.1f}} kg<br>CG Position: {center_cg:.2f} m<extra></extra>'
            ),
            row=2, col=1
        )
    
    # ────────────────────────────────────────────────────────────────────
    # Plot 4: Aircraft CG Range [x_min, x_max] - Total aircraft CG envelope
    # ────────────────────────────────────────────────────────────────────
    if len(cg_x) > 0:
        fig.add_trace(
            go.Scatter(
                x=[cg_min, cg_max],
                y=[0, 0],
                mode='markers+lines',
                name='Aircraft CG Range',
                line=dict(color=Colors.CLIMB, width=LineStyles.THICK),
                marker=dict(size=12, color=Colors.CLIMB),
                showlegend=False,
                hovertemplate='Aircraft CG_X: %{x:.3f} m<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Reference lines: Zero Fuel CG, x_CG,min, x_CG,max
        fig.add_vline(
            x=ZERO_FUEL_CG_X,
            line_dash="dash",
            line_color=Colors.WARNING,
            line_width=LineStyles.MEDIUM,
            annotation_text=f'Zero Fuel ({ZERO_FUEL_CG_X} m)',
            annotation_position="top",
            annotation_font_size=9,
            row=2, col=2
        )
        fig.add_vline(
            x=cg_min,
            line_dash="dot",
            line_color=Colors.CRUISE,
            line_width=LineStyles.THIN,
            annotation_text=f'Min ({cg_min:.3f} m)',
            annotation_position="bottom",
            annotation_font_size=9,
            row=2, col=2
        )
        fig.add_vline(
            x=cg_max,
            line_dash="dot",
            line_color=Colors.CRITICAL,
            line_width=LineStyles.THIN,
            annotation_text=f'Max ({cg_max:.3f} m)',
            annotation_position="bottom",
            annotation_font_size=9,
            row=2, col=2
        )
        
        # Range annotation: Δx_CG = x_max - x_min
        fig.add_annotation(
            x=(cg_min + cg_max) / 2,
            y=0.1,
            text=f'Aircraft CG Travel: {cg_range:.3f} m',
            showarrow=False,
            font=dict(size=12, color=Colors.CLIMB),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor=Colors.CLIMB,
            borderwidth=1,
            row=2, col=2
        )
    
    # ────────────────────────────────────────────────────────────────────
    # Plot 5: Current Tank Levels m_i,current - Bar chart
    # ────────────────────────────────────────────────────────────────────
    status = get_fuel_tank_status()
    if status.get('initialized', False):
        # Tank ordering for visual display: Inner L, Outer L, Center, Inner R, Outer R
        tank_fuel_dict = status.get('tank_fuel_kg', {})
        if not tank_fuel_dict:
            tank_names = []
            tank_fuel = []
            bar_colors = []
        else:
            # Spatial ordering: tanks {0,1,4,2,3}
            tank_order_indices = [0, 1, 4, 2, 3]
            tank_order_names = [TANK_NAMES[i] for i in tank_order_indices if i in TANK_NAMES]
            tank_names = [name for name in tank_order_names if name in tank_fuel_dict]
            tank_fuel = [tank_fuel_dict[name] for name in tank_names]
            
            # Color mapping by tank
            color_map = {
                TANK_NAMES[0]: Colors.CLIMB,      # Inner Left
                TANK_NAMES[1]: Colors.CRUISE,     # Outer Left
                TANK_NAMES[4]: Colors.DESCENT,    # Center Wing
                TANK_NAMES[2]: Colors.WARNING,    # Inner Right
                TANK_NAMES[3]: Colors.THRUST      # Outer Right
            }
            bar_colors = [color_map.get(name, Colors.CLIMB) for name in tank_names]
        
        if tank_names and tank_fuel:
            fig.add_trace(
                go.Bar(
                    x=tank_names,
                    y=tank_fuel,
                    name='Tank Fuel',
                    marker_color=bar_colors,
                    marker_opacity=0.7,
                    showlegend=False,
                    hovertemplate='<b>%{x}</b><br>Fuel: %{y:.1f} kg<extra></extra>',
                    text=[f'{f:.1f} kg' for f in tank_fuel],
                    textposition='outside',
                    textfont=dict(size=10)
                ),
                row=3, col=1
            )
    
    # ────────────────────────────────────────────────────────────────────
    # Plot 6: x_CG,fuel(Δm_fuel) - Fuel tank CG evolution (fuel only, no empty aircraft)
    # ────────────────────────────────────────────────────────────────────
    if len(fuel_tank_cg) > 0:
        min_len = min(len(fuel_consumed), len(fuel_tank_cg))
        fig.add_trace(
            go.Scatter(
                x=fuel_consumed[:min_len],
                y=fuel_tank_cg[:min_len],
                mode='lines',
                name='Fuel Tank CG',
                line=dict(color=Colors.WARNING, width=LineStyles.THICK),
                showlegend=True,
                hovertemplate='<b>Fuel Tank CG</b><br>Fuel Consumed: %{x:.1f} kg<br>Fuel CG_X: %{y:.3f} m<extra></extra>'
            ),
            row=3, col=2
        )
        
        # Reference lines: Tank CG positions x_i for i=0..4
        for tank_id, cg_pos in TANK_CG_POSITIONS.items():
            tank_name = TANK_NAMES.get(tank_id, f'Tank {tank_id}')
            fig.add_hline(
                y=cg_pos,
                line_dash="dot",
                line_color=Colors.GRID,
                line_width=LineStyles.THIN,
                opacity=0.5,
                annotation_text=f'{tank_name}: {cg_pos:.2f}m',
                annotation_position="right",
                annotation_font_size=8,
                row=3, col=2
            )
    
    # ════════════════════════════════════════════════════════════════════
    # Axis Configuration
    # ════════════════════════════════════════════════════════════════════
    fig.update_xaxes(**get_axis_config("Fuel Consumed (kg)"), row=1, col=1)
    # Custom y-axis range for CG plot to focus on actual data variation
    yaxis_config_1 = get_axis_config("Aircraft x_CG (m)")
    yaxis_config_1['range'] = [cg_y_min, cg_y_max]
    fig.update_yaxes(**yaxis_config_1, row=1, col=1)
    # Reverse x-axis for Aircraft Mass plot (mass decreases from left to right)
    xaxis_config_mass = get_axis_config("Aircraft Mass (kg)")
    xaxis_config_mass['autorange'] = 'reversed'
    fig.update_xaxes(**xaxis_config_mass, row=1, col=2)
    # Custom y-axis range for CG plot to focus on actual data variation
    yaxis_config_2 = get_axis_config("Aircraft x_CG (m)")
    yaxis_config_2['range'] = [cg_y_min, cg_y_max]
    fig.update_yaxes(**yaxis_config_2, row=1, col=2)
    fig.update_xaxes(**get_axis_config("Fuel Consumed (kg)"), row=2, col=1)
    fig.update_yaxes(**get_axis_config("Tank Fuel (kg)"), row=2, col=1)
    fig.update_xaxes(**get_axis_config("x_CG (m)"), row=2, col=2)
    fig.update_yaxes(title_text="", showticklabels=False, row=2, col=2)
    fig.update_xaxes(**get_axis_config("Tank"), row=3, col=1)
    fig.update_yaxes(**get_axis_config("Fuel (kg)"), row=3, col=1)
    fig.update_xaxes(**get_axis_config("Fuel Consumed (kg)"), row=3, col=2)
    fig.update_yaxes(**get_axis_config("Fuel Tank x_CG (m)"), row=3, col=2)
    
    # ════════════════════════════════════════════════════════════════════
    # Dashboard Subtitle: Summary Statistics
    # ════════════════════════════════════════════════════════════════════
    subtitle = (
        f"Scenario: {plotter.tank_system.scenario} | "
        f"Δm_fuel,total: {total_fuel:.1f} kg | "
        f"Δx_CG: {cg_range:.3f} m | "
        f"Δx_shift: {cg_shift:+.3f} m | "
        f"x_CG,0: {cg_initial:.3f} m | "
        f"x_CG,f: {cg_final:.3f} m"
    )
    if len(weight_history) > 0:
        subtitle += f" | Δm_total: {weight_loss:.1f} kg"
    
    # ════════════════════════════════════════════════════════════════════
    # Layout Configuration
    # ════════════════════════════════════════════════════════════════════
    layout_config = get_standard_layout(
        "CG MOVEMENT ANALYSIS - COMPREHENSIVE DASHBOARD",
        subtitle,
        height=1600,
        width=Layout.WIDE_WIDTH
    )
    layout_config['margin'] = dict(l=80, r=200, t=120, b=80)
    
    fig.update_layout(
        **layout_config,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=Typography.LEGEND_SIZE - 1)
        )
    )
    
    # ════════════════════════════════════════════════════════════════════
    # Export and Display
    # ════════════════════════════════════════════════════════════════════
    if save_plots:
        cg_dir = get_or_create_run_directory(phase="CG")
        output_path_html = os.path.join(cg_dir, 'cg_analysis_dashboard.html')
        output_path_png = os.path.join(cg_dir, 'cg_analysis_dashboard.png')
        
        export_config = ExportConfig.get_plotly_config()
        export_config['toImageButtonOptions']['filename'] = 'cg_analysis_dashboard'
        
        # Export HTML (interactive)
        pio.write_html(fig, file=output_path_html, auto_open=show_plots, config=export_config)
        
        # Export PNG (static)
        try:
            fig.write_image(output_path_png, width=1800, height=1600, scale=2)
            print(f"[CG_PLOTTER] Saved: {output_path_html} (HTML) and {output_path_png} (PNG)")
        except Exception as e:
            print(f"[CG_PLOTTER] Saved: {output_path_html} (HTML only)")
            print(f"[WARNING] PNG export failed: {e}")
        
        # ════════════════════════════════════════════════════════════════
        # Save individual plots as separate PNG files
        # ════════════════════════════════════════════════════════════════
        try:
            save_prefix = "cg_analysis"
            
            # 1. Aircraft x_CG vs Fuel Consumed
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=fuel_consumed, y=cg_x, mode='lines', name='Aircraft CG_X',
                line=dict(color=Colors.CLIMB, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(65, 105, 225, 0.15)',
                hovertemplate='<b>Aircraft CG</b><br>Fuel Consumed: %{x:.1f} kg<br>Aircraft CG_X: %{y:.3f} m<extra></extra>'))
            fig1.add_hline(y=ZERO_FUEL_CG_X, line_dash="dash", line_color=Colors.WARNING, line_width=LineStyles.MEDIUM,
                annotation_text=f'Zero Fuel CG ({ZERO_FUEL_CG_X} m)', annotation_position="right", annotation_font_size=10)
            fig1.update_layout(**get_standard_layout("CG ANALYSIS - Aircraft x_CG vs Fuel Consumed", subtitle, height=600, width=900))
            fig1.update_xaxes(**get_axis_config("Fuel Consumed (kg)"))
            # Custom y-axis range for CG plot to focus on actual data variation
            yaxis_config_fig1 = get_axis_config("Aircraft x_CG (m)")
            yaxis_config_fig1['range'] = [cg_y_min, cg_y_max]
            fig1.update_yaxes(**yaxis_config_fig1)
            fig1.write_image(os.path.join(cg_dir, f'{save_prefix}_cg_vs_fuel.png'), width=1200, height=800, scale=2)
            
            # 2. Aircraft x_CG vs Aircraft Mass
            if len(weight_history) > 0:
                min_len = min(len(cg_x), len(weight_history))
                weight = np.array(weight_history[:min_len])
                cg_x_aligned = cg_x[:min_len]
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=weight, y=cg_x_aligned, mode='lines', name='Aircraft CG_X',
                    line=dict(color=Colors.CRUISE, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(0, 128, 0, 0.15)',
                    hovertemplate='<b>Aircraft CG</b><br>Mass: %{x:.1f} kg<br>Aircraft CG_X: %{y:.3f} m<extra></extra>'))
                fig2.add_hline(y=ZERO_FUEL_CG_X, line_dash="dash", line_color=Colors.WARNING, line_width=LineStyles.MEDIUM,
                    annotation_text=f'Zero Fuel CG ({ZERO_FUEL_CG_X} m)', annotation_position="right", annotation_font_size=10)
                fig2.update_layout(**get_standard_layout("CG ANALYSIS - Aircraft x_CG vs Aircraft Mass", subtitle, height=600, width=900))
                # Reverse x-axis for Aircraft Mass plot (mass decreases from left to right)
                xaxis_config_mass_fig2 = get_axis_config("Aircraft Mass (kg)")
                xaxis_config_mass_fig2['autorange'] = 'reversed'
                fig2.update_xaxes(**xaxis_config_mass_fig2)
                # Custom y-axis range for CG plot to focus on actual data variation
                yaxis_config_fig2 = get_axis_config("Aircraft x_CG (m)")
                yaxis_config_fig2['range'] = [cg_y_min, cg_y_max]
                fig2.update_yaxes(**yaxis_config_fig2)
                fig2.write_image(os.path.join(cg_dir, f'{save_prefix}_cg_vs_mass.png'), width=1200, height=800, scale=2)
            
            # 3. Tank Fuel Evolution (Grouped)
            fig3 = go.Figure()
            if grouped_fuel['outer'] is not None:
                min_len = min(len(fuel_consumed), len(grouped_fuel['outer']))
                outer_cg_avg = np.mean([TANK_CG_POSITIONS[1], TANK_CG_POSITIONS[3]])
                fig3.add_trace(go.Scatter(
                    x=fuel_consumed[:min_len], y=grouped_fuel['outer'][:min_len], mode='lines', name='Outer Tanks',
                    line=dict(color=category_colors['outer'], width=LineStyles.THICK),
                    hovertemplate=f'<b>Outer Tanks</b><br>Fuel Consumed: %{{x:.1f}} kg<br>Total Fuel: %{{y:.1f}} kg<br>Avg CG Position: {outer_cg_avg:.2f} m<extra></extra>'))
            if grouped_fuel['inner'] is not None:
                min_len = min(len(fuel_consumed), len(grouped_fuel['inner']))
                inner_cg_avg = np.mean([TANK_CG_POSITIONS[0], TANK_CG_POSITIONS[2]])
                fig3.add_trace(go.Scatter(
                    x=fuel_consumed[:min_len], y=grouped_fuel['inner'][:min_len], mode='lines', name='Inner Tanks',
                    line=dict(color=category_colors['inner'], width=LineStyles.THICK),
                    hovertemplate=f'<b>Inner Tanks</b><br>Fuel Consumed: %{{x:.1f}} kg<br>Total Fuel: %{{y:.1f}} kg<br>Avg CG Position: {inner_cg_avg:.2f} m<extra></extra>'))
            if grouped_fuel['center'] is not None:
                min_len = min(len(fuel_consumed), len(grouped_fuel['center']))
                center_cg = TANK_CG_POSITIONS[4]
                fig3.add_trace(go.Scatter(
                    x=fuel_consumed[:min_len], y=grouped_fuel['center'][:min_len], mode='lines', name='Center Tank',
                    line=dict(color=category_colors['center'], width=LineStyles.THICK),
                    hovertemplate=f'<b>Center Tank</b><br>Fuel Consumed: %{{x:.1f}} kg<br>Tank Fuel: %{{y:.1f}} kg<br>CG Position: {center_cg:.2f} m<extra></extra>'))
            fig3.update_layout(**get_standard_layout("CG ANALYSIS - Tank Fuel Evolution (Grouped)", subtitle, height=600, width=900))
            fig3.update_xaxes(**get_axis_config("Fuel Consumed (kg)")); fig3.update_yaxes(**get_axis_config("Tank Fuel (kg)"))
            fig3.write_image(os.path.join(cg_dir, f'{save_prefix}_tank_evolution.png'), width=1200, height=800, scale=2)
            
            # 4. Aircraft CG Range Analysis
            if len(cg_x) > 0:
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    x=[cg_min, cg_max], y=[0, 0], mode='markers+lines', name='Aircraft CG Range',
                    line=dict(color=Colors.CLIMB, width=LineStyles.THICK), marker=dict(size=12, color=Colors.CLIMB),
                    hovertemplate='Aircraft CG_X: %{x:.3f} m<extra></extra>'))
                fig4.add_vline(x=ZERO_FUEL_CG_X, line_dash="dash", line_color=Colors.WARNING, line_width=LineStyles.MEDIUM,
                    annotation_text=f'Zero Fuel ({ZERO_FUEL_CG_X} m)', annotation_position="top", annotation_font_size=9)
                fig4.add_vline(x=cg_min, line_dash="dot", line_color=Colors.CRUISE, line_width=LineStyles.THIN,
                    annotation_text=f'Min ({cg_min:.3f} m)', annotation_position="bottom", annotation_font_size=9)
                fig4.add_vline(x=cg_max, line_dash="dot", line_color=Colors.CRITICAL, line_width=LineStyles.THIN,
                    annotation_text=f'Max ({cg_max:.3f} m)', annotation_position="bottom", annotation_font_size=9)
                fig4.add_annotation(x=(cg_min + cg_max) / 2, y=0.1, text=f'Aircraft CG Travel: {cg_range:.3f} m',
                    showarrow=False, font=dict(size=12, color=Colors.CLIMB), bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor=Colors.CLIMB, borderwidth=1)
                fig4.update_layout(**get_standard_layout("CG ANALYSIS - Aircraft CG Range", subtitle, height=600, width=900))
                fig4.update_xaxes(**get_axis_config("x_CG (m)")); fig4.update_yaxes(title_text="", showticklabels=False)
                fig4.write_image(os.path.join(cg_dir, f'{save_prefix}_cg_range.png'), width=1200, height=800, scale=2)
            
            # 5. Current Tank Fuel Levels
            status = get_fuel_tank_status()
            if status.get('initialized', False):
                tank_fuel_dict = status.get('tank_fuel_kg', {})
                if tank_fuel_dict:
                    tank_order_indices = [0, 1, 4, 2, 3]
                    tank_order_names = [TANK_NAMES[i] for i in tank_order_indices if i in TANK_NAMES]
                    tank_names = [name for name in tank_order_names if name in tank_fuel_dict]
                    tank_fuel = [tank_fuel_dict[name] for name in tank_names]
                    
                    color_map = {
                        TANK_NAMES[0]: Colors.CLIMB, TANK_NAMES[1]: Colors.CRUISE,
                        TANK_NAMES[4]: Colors.DESCENT, TANK_NAMES[2]: Colors.WARNING,
                        TANK_NAMES[3]: Colors.THRUST
                    }
                    bar_colors = [color_map.get(name, Colors.CLIMB) for name in tank_names]
                    
                    fig5 = go.Figure()
                    fig5.add_trace(go.Bar(
                        x=tank_names, y=tank_fuel, name='Tank Fuel', marker_color=bar_colors, marker_opacity=0.7,
                        hovertemplate='<b>%{x}</b><br>Fuel: %{y:.1f} kg<extra></extra>',
                        text=[f'{f:.1f} kg' for f in tank_fuel], textposition='outside', textfont=dict(size=10)))
                    fig5.update_layout(**get_standard_layout("CG ANALYSIS - Current Tank Fuel Levels", subtitle, height=600, width=900))
                    fig5.update_xaxes(**get_axis_config("Tank")); fig5.update_yaxes(**get_axis_config("Fuel (kg)"))
                    fig5.write_image(os.path.join(cg_dir, f'{save_prefix}_tank_levels.png'), width=1200, height=800, scale=2)
            
            # 6. Fuel Tank CG Evolution
            if len(fuel_tank_cg) > 0:
                min_len = min(len(fuel_consumed), len(fuel_tank_cg))
                fig6 = go.Figure()
                fig6.add_trace(go.Scatter(
                    x=fuel_consumed[:min_len], y=fuel_tank_cg[:min_len], mode='lines', name='Fuel Tank CG',
                    line=dict(color=Colors.WARNING, width=LineStyles.THICK),
                    hovertemplate='<b>Fuel Tank CG</b><br>Fuel Consumed: %{x:.1f} kg<br>Fuel CG_X: %{y:.3f} m<extra></extra>'))
                for tank_id, cg_pos in TANK_CG_POSITIONS.items():
                    tank_name = TANK_NAMES.get(tank_id, f'Tank {tank_id}')
                    fig6.add_hline(y=cg_pos, line_dash="dot", line_color=Colors.GRID, line_width=LineStyles.THIN, opacity=0.5,
                        annotation_text=f'{tank_name}: {cg_pos:.2f}m', annotation_position="right", annotation_font_size=8)
                fig6.update_layout(**get_standard_layout("CG ANALYSIS - Fuel Tank CG Evolution", subtitle, height=600, width=900))
                fig6.update_xaxes(**get_axis_config("Fuel Consumed (kg)")); fig6.update_yaxes(**get_axis_config("Fuel Tank x_CG (m)"))
                fig6.write_image(os.path.join(cg_dir, f'{save_prefix}_fuel_tank_cg.png'), width=1200, height=800, scale=2)
            
            print(f"[CG_PLOTTER] Individual CG plots saved as PNG to: {cg_dir}")
        except Exception as e:
            print(f"[WARNING] Could not save individual CG plots: {e}")
    elif show_plots:
        fig.show()
    
    print("[CG_PLOTTER] Dashboard generation complete")
    
    return fig
