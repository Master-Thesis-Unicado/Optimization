"""
Cruise Phase Visualization Module

This module contains all plotting and visualization functions for cruise simulation results.
Provides both interactive Plotly plots and matplotlib fallback options.


"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
import plotly.subplots as sp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cruise import CruiseResults

# Import visualization configuration for consistent styling
from visualization_config import (
    Colors, Typography, Layout, LineStyles,
    get_standard_layout, get_standard_legend, get_axis_config,
    ExportConfig, get_or_create_run_directory
)


def plot_cruise_performance_detailed(cruise_results: 'CruiseResults') -> None:
    """
    Create detailed cruise performance analysis plot (cruise-specific data only).
    
    Shows detailed cruise metrics aligned with descent performance plots:
    - Fuel Flow Rate
    - Thrust vs Drag
    - Weight Evolution
    - Lever Position
    - True Airspeed
    - Fuel Consumption
    
    Each subplot can be exported individually.
    
    Args:
        cruise_results: Complete cruise simulation results
    """
    
    # Extract cruise data (relative time starting from 0)
    cruise_time_min = cruise_results.time_s / 60.0  # Convert to minutes
    cruise_distance_km = cruise_results.distance_km
    cruise_fuel_flow_kgh = cruise_results.fuel_flow_kgps * 3600  # Convert to kg/h
    cruise_thrust_kn = cruise_results.thrust_total_N / 1000  # Convert to kN
    cruise_drag_kn = cruise_results.drag_N / 1000  # Convert to kN
    cruise_weight_kg = cruise_results.weight_kg
    cruise_lever = cruise_results.lever_position * 100  # Convert to percentage
    cruise_fuel_consumed = cruise_results.fuel_consumed_kg
    cruise_tas_ms = cruise_results.true_airspeed_mps
    
    # Create cruise performance plot (3x2 grid) with consistent styling
    fig = sp.make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '<b>Fuel Flow Rate</b>', '<b>Thrust vs Drag</b>', 
            '<b>Weight Evolution</b>', '<b>Lever Position</b>',
            '<b>True Airspeed</b>', '<b>Cumulative Fuel Consumption</b>'
        ),
        vertical_spacing=0.15,  # Increased from 0.12
        horizontal_spacing=0.15  # Increased from 0.12
    )
    
    # 1. Fuel Flow Rate over time (aligned with descent)
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
    
    # 2. Thrust vs Drag over time (aligned with descent)
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
    
    # 3. Weight Evolution over time (aligned with descent)
    fig.add_trace(
        go.Scatter(
            x=cruise_time_min,
            y=cruise_weight_kg,
            mode='lines',
            name='Weight (Cruise)',
            line=dict(color=Colors.CRUISE, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(0, 128, 0, 0.15)',
            hovertemplate='<b>Cruise</b><br>Time: %{x:.1f} min<br>Weight: %{y:.0f} kg<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Engine Lever Position over time (aligned with descent)
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
    
    # Add reference lines for lever position
    fig.add_hline(y=85, line_dash=LineStyles.DASH, line_color=Colors.WARNING, 
                  annotation_text="MCT Limit", annotation_position="top right",
                  row=2, col=2)
    fig.add_hline(y=100, line_dash=LineStyles.DOT, line_color=Colors.CRITICAL,
                  annotation_text="Max Thrust", annotation_position="bottom right",
                  row=2, col=2)
    
    # 5. True Airspeed over time (aligned with descent)
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
    
    # 6. Cumulative Fuel Consumption over time (aligned with descent)
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
    
    # Calculate summary statistics
    total_time_min = cruise_time_min[-1] if len(cruise_time_min) > 0 else 0.0
    total_distance = cruise_distance_km[-1] if len(cruise_distance_km) > 0 else 0.0
    total_fuel = cruise_fuel_consumed[-1] if len(cruise_fuel_consumed) > 0 else 0.0
    avg_fuel_flow = np.mean(cruise_fuel_flow_kgh)
    
    # Update layout with standard configuration
    subtitle = (
        f"Distance: {total_distance:.0f} km | Time: {total_time_min:.1f} min | "
        f"Fuel: {total_fuel:.1f} kg | Avg Fuel Flow: {avg_fuel_flow:.0f} kg/h"
    )
    
    layout_config = get_standard_layout(
        "CRUISE PERFORMANCE ANALYSIS (2D)",
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
    
    # Update axes labels with standard configuration (now using Time instead of Distance for alignment)
    fig.update_xaxes(**get_axis_config("Time (min)"), row=1, col=1)
    fig.update_yaxes(**get_axis_config("Fuel Flow (kg/h)"), row=1, col=1)
    
    fig.update_xaxes(**get_axis_config("Time (min)"), row=1, col=2)
    fig.update_yaxes(**get_axis_config("Force (kN)"), row=1, col=2)
    
    fig.update_xaxes(**get_axis_config("Time (min)"), row=2, col=1)
    # Set y-axis range to zoom in on weight changes for better visibility
    weight_min, weight_max = np.min(cruise_weight_kg), np.max(cruise_weight_kg)
    weight_margin = (weight_max - weight_min) * 0.2  # Add 20% margin
    fig.update_yaxes(**get_axis_config("Weight (kg)"), 
                     range=[weight_min - weight_margin, weight_max + weight_margin], 
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
    config['toImageButtonOptions']['filename'] = 'cruise_performance_2d'
    
    # Save individual plots as separate HTML files in timestamped directory/Cruise subfolder
    try:
        run_dir = get_or_create_run_directory(phase="Cruise")
        save_prefix = "cruise_performance"
        
        # 1. Fuel Flow Rate
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=cruise_time_min, y=cruise_fuel_flow_kgh, mode='lines', name='Fuel Flow (Cruise)',
            line=dict(color=Colors.CRUISE, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(0, 128, 0, 0.15)',
            hovertemplate='<b>Cruise</b><br>Time: %{x:.1f} min<br>Fuel Flow: %{y:.1f} kg/h<extra></extra>'))
        fig1.update_layout(**get_standard_layout("CRUISE PERFORMANCE - Fuel Flow Rate", subtitle, height=600, width=900))
        fig1.update_xaxes(**get_axis_config("Time (min)")); fig1.update_yaxes(**get_axis_config("Fuel Flow (kg/h)"))
        fig1.write_image(os.path.join(run_dir, f'{save_prefix}_fuel_flow.png'), width=1200, height=800, scale=2)
        
        # 2. Thrust vs Drag
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=cruise_time_min, y=cruise_thrust_kn, mode='lines', name='Thrust (Cruise)',
            line=dict(color='darkgreen', width=LineStyles.THICK), hovertemplate='<b>Cruise</b><br>Time: %{x:.1f} min<br>Thrust: %{y:.1f} kN<extra></extra>'))
        fig2.add_trace(go.Scatter(x=cruise_time_min, y=cruise_drag_kn, mode='lines', name='Drag (Cruise)',
            line=dict(color='lightcoral', width=LineStyles.THICK, dash=LineStyles.DASH), hovertemplate='<b>Cruise</b><br>Time: %{x:.1f} min<br>Drag: %{y:.1f} kN<extra></extra>'))
        fig2.update_layout(**get_standard_layout("CRUISE PERFORMANCE - Thrust vs Drag", subtitle, height=600, width=900))
        fig2.update_xaxes(**get_axis_config("Time (min)")); fig2.update_yaxes(**get_axis_config("Force (kN)"))
        fig2.write_image(os.path.join(run_dir, f'{save_prefix}_thrust_drag.png'), width=1200, height=800, scale=2)
        
        # 3. Weight Evolution
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=cruise_time_min, y=cruise_weight_kg, mode='lines', name='Weight (Cruise)',
            line=dict(color=Colors.CRUISE, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(0, 128, 0, 0.15)',
            hovertemplate='<b>Cruise</b><br>Time: %{x:.1f} min<br>Weight: %{y:.0f} kg<extra></extra>'))
        fig3.update_layout(**get_standard_layout("CRUISE PERFORMANCE - Weight Evolution", subtitle, height=600, width=900))
        # Set y-axis range to zoom in on weight changes for better visibility
        weight_min, weight_max = np.min(cruise_weight_kg), np.max(cruise_weight_kg)
        weight_margin = (weight_max - weight_min) * 0.2  # Add 20% margin
        fig3.update_xaxes(**get_axis_config("Time (min)")); 
        fig3.update_yaxes(**get_axis_config("Weight (kg)"), range=[weight_min - weight_margin, weight_max + weight_margin])
        fig3.write_image(os.path.join(run_dir, f'{save_prefix}_weight.png'), width=1200, height=800, scale=2)
        
        # 4. Lever Position
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=cruise_time_min, y=cruise_lever, mode='lines', name='Lever Position (Cruise)',
            line=dict(color='olive', width=LineStyles.THICK), hovertemplate='<b>Cruise</b><br>Time: %{x:.1f} min<br>Lever: %{y:.1f}%<extra></extra>'))
        fig4.add_hline(y=85, line_dash=LineStyles.DASH, line_color=Colors.WARNING, annotation_text="MCT Limit", annotation_position="top right")
        fig4.add_hline(y=100, line_dash=LineStyles.DOT, line_color=Colors.CRITICAL, annotation_text="Max Thrust", annotation_position="bottom right")
        fig4.update_layout(**get_standard_layout("CRUISE PERFORMANCE - Lever Position", subtitle, height=600, width=900))
        fig4.update_xaxes(**get_axis_config("Time (min)")); fig4.update_yaxes(**get_axis_config("Lever Position (%)"))
        fig4.write_image(os.path.join(run_dir, f'{save_prefix}_lever.png'), width=1200, height=800, scale=2)
        
        # 5. True Airspeed
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=cruise_time_min, y=cruise_tas_ms, mode='lines', name='True Airspeed (Cruise)',
            line=dict(color=Colors.CRUISE, width=LineStyles.THICK), hovertemplate='<b>Cruise</b><br>Time: %{x:.1f} min<br>TAS: %{y:.1f} m/s<extra></extra>'))
        fig5.update_layout(**get_standard_layout("CRUISE PERFORMANCE - True Airspeed", subtitle, height=600, width=900))
        fig5.update_xaxes(**get_axis_config("Time (min)")); fig5.update_yaxes(**get_axis_config("True Airspeed (m/s)"))
        fig5.write_image(os.path.join(run_dir, f'{save_prefix}_airspeed.png'), width=1200, height=800, scale=2)
        
        # 6. Cumulative Fuel
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=cruise_time_min, y=cruise_fuel_consumed, mode='lines', name='Fuel Consumed (Cruise)',
            line=dict(color=Colors.CRUISE, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(0, 128, 0, 0.15)',
            hovertemplate='<b>Cruise</b><br>Time: %{x:.1f} min<br>Fuel: %{y:.1f} kg<extra></extra>'))
        fig6.update_layout(**get_standard_layout("CRUISE PERFORMANCE - Cumulative Fuel Consumption", subtitle, height=600, width=900))
        fig6.update_xaxes(**get_axis_config("Time (min)")); fig6.update_yaxes(**get_axis_config("Cumulative Fuel (kg)"))
        fig6.write_image(os.path.join(run_dir, f'{save_prefix}_fuel.png'), width=1200, height=800, scale=2)
        
        print(f"[EXPORT] Individual cruise plots saved as PNG to: {run_dir}")
    except Exception as e:
        print(f"[WARNING] Could not save individual cruise plots: {e}")
    
    # Show the plot
    fig.show(config=config)


# Helper function removed - subplot selection menu no longer needed


def plot_cruise_results_plotly(cruise_results: 'CruiseResults') -> None:
    """
    Create comprehensive interactive plots using Plotly.
    
    Args:
        cruise_results: Complete cruise simulation results
    """
    
    # Create subplots - now 3x2 grid to include lever position
    fig = sp.make_subplots(
        rows=3, cols=2,
        subplot_titles=('Weight and Fuel Consumption', 'Thrust and Drag Balance', 
                       'Fuel Flow Rate', 'Specific Excess Power',
                       'Engine Lever Position', 'Altitude and Mach Profile'),
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": True}]],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    # Convert time to hours for better readability
    time_hours = cruise_results.time_s / 3600.0
    
    # 1. Weight and Fuel Consumption
    fig.add_trace(
        go.Scatter(
            x=time_hours,
            y=cruise_results.weight_kg,
            mode='lines',
            name='Aircraft Weight',
            line=dict(color='blue', width=3),
            hovertemplate='Time: %{x:.2f} h<br>Weight: %{y:.0f} kg<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=time_hours,
            y=cruise_results.fuel_consumed_kg,
            mode='lines',
            name='Fuel Consumed',
            line=dict(color='red', width=3),
            yaxis='y2',
            hovertemplate='Time: %{x:.2f} h<br>Fuel: %{y:.1f} kg<extra></extra>'
        ),
        row=1, col=1, secondary_y=True
    )
    
    # 2. Thrust and Drag Balance  
    fig.add_trace(
        go.Scatter(
            x=time_hours,
            y=cruise_results.thrust_total_N,
            mode='lines',
            name='Total Thrust',
            line=dict(color='green', width=3),
            hovertemplate='Time: %{x:.2f} h<br>Thrust: %{y:.0f} N<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=time_hours,
            y=cruise_results.drag_N,
            mode='lines',
            name='Drag',
            line=dict(color='orange', width=3, dash='dash'),
            hovertemplate='Time: %{x:.2f} h<br>Drag: %{y:.0f} N<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Fuel Flow Rate
    fig.add_trace(
        go.Scatter(
            x=time_hours,
            y=cruise_results.fuel_flow_kgps * 3600,  # Convert to kg/h
            mode='lines',
            name='Fuel Flow Rate',
            line=dict(color='purple', width=3),
            hovertemplate='Time: %{x:.2f} h<br>Fuel Flow: %{y:.0f} kg/h<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Specific Excess Power
    fig.add_trace(
        go.Scatter(
            x=time_hours,
            y=cruise_results.specific_excess_power_mps,
            mode='lines',
            name='Specific Excess Power',
            line=dict(color='brown', width=3),
            hovertemplate='Time: %{x:.2f} h<br>Ps: %{y:.3f} m/s<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Add horizontal line at Ps = 0 for reference
    fig.add_hline(y=0, line_dash="dot", line_color="gray", 
                  annotation_text="Perfect Steady Cruise", row=2, col=2)
    
    # 5. Engine Lever Position
    fig.add_trace(
        go.Scatter(
            x=time_hours,
            y=cruise_results.lever_position * 100,  # Convert to percentage
            mode='lines',
            name='Lever Position',
            line=dict(color='darkred', width=3),
            hovertemplate='Time: %{x:.2f} h<br>Lever: %{y:.1f}%<extra></extra>'
        ),
        row=3, col=1
    )
    
    # Add reference lines for lever position
    fig.add_hline(y=85, line_dash="dash", line_color="orange", 
                  annotation_text="MCT Limit (85%)", row=3, col=1)
    fig.add_hline(y=100, line_dash="dash", line_color="red", 
                  annotation_text="Max Thrust (100%)", row=3, col=1)
    
    # 6. Altitude and Mach Profile
    fig.add_trace(
        go.Scatter(
            x=time_hours,
            y=cruise_results.altitude_m,
            mode='lines',
            name='Altitude',
            line=dict(color='blue', width=3),
            hovertemplate='Time: %{x:.2f} h<br>Altitude: %{y:.0f} m<extra></extra>'
        ),
        row=3, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=time_hours,
            y=cruise_results.mach_number,
            mode='lines',
            name='Mach Number',
            line=dict(color='green', width=3),
            yaxis='y2',
            hovertemplate='Time: %{x:.2f} h<br>Mach: %{y:.3f}<extra></extra>'
        ),
        row=3, col=2, secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=f"Cruise Simulation Results<br>" + 
              f"Distance: {cruise_results.target_distance_km:.0f} km, " +
              f"Altitude: {cruise_results.initial_state.altitude_m:.0f} m, " +
              f"Mach: {cruise_results.initial_state.mach:.3f}",
        title_x=0.5,
        title_font_size=16,
        showlegend=True,
        height=1000,
        width=1200
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (hours)", row=1, col=1)
    fig.update_yaxes(title_text="Aircraft Weight (kg)", row=1, col=1)
    fig.update_yaxes(title_text="Fuel Consumed (kg)", secondary_y=True, row=1, col=1)
    
    fig.update_xaxes(title_text="Time (hours)", row=1, col=2)
    fig.update_yaxes(title_text="Force (N)", row=1, col=2)
    
    fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
    fig.update_yaxes(title_text="Fuel Flow Rate (kg/h)", row=2, col=1)
    
    fig.update_xaxes(title_text="Time (hours)", row=2, col=2)
    fig.update_yaxes(title_text="Specific Excess Power (m/s)", row=2, col=2)
    
    fig.update_xaxes(title_text="Time (hours)", row=3, col=1)
    fig.update_yaxes(title_text="Lever Position (%)", row=3, col=1)
    
    fig.update_xaxes(title_text="Time (hours)", row=3, col=2)
    fig.update_yaxes(title_text="Altitude (m)", row=3, col=2)
    fig.update_yaxes(title_text="Mach Number", secondary_y=True, row=3, col=2)
    
    # Show the plot
    fig.show()
    
    # Create summary table
    create_cruise_summary_table(cruise_results)

def create_cruise_summary_table(cruise_results: 'CruiseResults') -> None:
    """
    Create a summary table of cruise results using Plotly.
    
    Args:
        cruise_results: Complete cruise simulation results
    """
    
    summary = cruise_results.get_summary_dict()
    
    # Prepare table data
    table_data = [
        ["Parameter", "Value", "Unit"],
        ["Cruise Distance", f"{summary['cruise_distance_km']:.1f}", "km"],
        ["Cruise Time", f"{summary['cruise_time_hours']:.2f}", "hours"],
        ["Fuel Consumed", f"{summary['cruise_fuel_kg']:.1f}", "kg"],
        ["Average Fuel Flow", f"{summary['avg_fuel_flow_kg_h']:.0f}", "kg/h"],
        ["Average Thrust", f"{summary['avg_thrust_N']:.0f}", "N"],
        ["Initial Weight", f"{summary['initial_weight_kg']:.0f}", "kg"],
        ["Final Weight", f"{summary['final_weight_kg']:.0f}", "kg"],
        ["Cruise Mach", f"{summary['cruise_mach']:.3f}", "-"],
        ["Cruise Altitude", f"{summary['cruise_altitude_m']:.0f}", "m"],
        ["Average Lever Position", f"{np.mean(cruise_results.lever_position)*100:.1f}", "%"],
        ["Max Lever Position", f"{np.max(cruise_results.lever_position)*100:.1f}", "%"],
    ]
    
    # Create table using Plotly
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=table_data[0],
            fill_color='lightblue',
            align='center',
            font=dict(size=14, color='black', family='Arial Black')
        ),
        cells=dict(
            values=list(zip(*table_data[1:])),
            fill_color=[
                ['lightgray' if i % 2 == 0 else 'white' for i in range(len(table_data)-1)] for _ in range(3)
            ],
            align=['left', 'right', 'center'],
            font=dict(size=12, color='black', family='Arial')
        )
    )])
    
    fig.update_layout(
        title="Cruise Simulation Summary",
        title_x=0.5,
        title_font_size=18,
        height=400,
        width=800,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    fig.show()

def plot_cruise_results_matplotlib(cruise_results: 'CruiseResults') -> None:
    """
    Create plots using matplotlib (fallback option).
    
    Args:
        cruise_results: Complete cruise simulation results
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    time_hours = cruise_results.time_s / 3600.0
    
    # Weight and Fuel
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(time_hours, cruise_results.weight_kg, 'b-', linewidth=2, label='Weight')
    line2 = ax1_twin.plot(time_hours, cruise_results.fuel_consumed_kg, 'r-', linewidth=2, label='Fuel Consumed')
    
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Aircraft Weight (kg)', color='b')
    ax1_twin.set_ylabel('Fuel Consumed (kg)', color='r')
    ax1.set_title('Weight and Fuel Consumption')
    ax1.grid(True, alpha=0.3)
    
    # Thrust and Drag
    axes[0, 1].plot(time_hours, cruise_results.thrust_total_N, 'g-', linewidth=2, label='Thrust')
    axes[0, 1].plot(time_hours, cruise_results.drag_N, 'orange', linewidth=2, linestyle='--', label='Drag')
    axes[0, 1].set_xlabel('Time (hours)')
    axes[0, 1].set_ylabel('Force (N)')
    axes[0, 1].set_title('Thrust and Drag Balance')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Fuel Flow
    axes[1, 0].plot(time_hours, cruise_results.fuel_flow_kgps * 3600, 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Fuel Flow Rate (kg/h)')
    axes[1, 0].set_title('Fuel Flow Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Specific Excess Power
    axes[1, 1].plot(time_hours, cruise_results.specific_excess_power_mps, 'brown', linewidth=2)
    axes[1, 1].axhline(y=0, color='gray', linestyle=':', alpha=0.7, label='Perfect Steady Cruise')
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Specific Excess Power (m/s)')
    axes[1, 1].set_title('Specific Excess Power')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Cruise Simulation Results - Distance: {cruise_results.target_distance_km:.0f} km, '
                f'Altitude: {cruise_results.initial_state.altitude_m:.0f} m, '
                f'Mach: {cruise_results.initial_state.mach:.3f}', fontsize=14)
    
    plt.tight_layout()
    plt.show()

def plot_cruise_performance_comparison(cruise_results_list: list['CruiseResults'], 
                                     labels: list[str] = None) -> None:
    """
    Create comparison plots for multiple cruise simulations.
    
    Args:
        cruise_results_list: List of CruiseResults objects to compare
        labels: Optional list of labels for each simulation
    """
    
    if not cruise_results_list:
        print("[WARNING] No cruise results provided for comparison")
        return
    
    n_sims = len(cruise_results_list)
    if labels is None:
        labels = [f"Simulation {i+1}" for i in range(n_sims)]
    
    # Create comparison plots
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=('Fuel Consumption Comparison', 'Weight Reduction', 
                       'Fuel Flow Rates', 'Flight Efficiency'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Extract comparison data
    distances = [cr.target_distance_km for cr in cruise_results_list]
    fuel_consumed = [cr.total_fuel_consumed_kg for cr in cruise_results_list]
    flight_times = [cr.total_time_s / 3600.0 for cr in cruise_results_list]  # hours
    avg_fuel_flows = [cr.average_fuel_flow_kgps * 3600.0 for cr in cruise_results_list]  # kg/h
    weight_reductions = [cr.initial_state.weight_kg - cr.final_weight_kg for cr in cruise_results_list]
    fuel_efficiency = [dist / fuel for dist, fuel in zip(distances, fuel_consumed)]  # km/kg
    
    # 1. Fuel Consumption Comparison
    fig.add_trace(
        go.Bar(
            x=labels,
            y=fuel_consumed,
            name='Fuel Consumed',
            marker_color='red',
            text=[f'{fuel:.1f} kg' for fuel in fuel_consumed],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # 2. Weight Reduction
    fig.add_trace(
        go.Bar(
            x=labels,
            y=weight_reductions,
            name='Weight Reduction',
            marker_color='blue',
            text=[f'{wr:.1f} kg' for wr in weight_reductions],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # 3. Fuel Flow Rates vs Time
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    for i, (cr, label) in enumerate(zip(cruise_results_list, labels)):
        time_hours = cr.time_s / 3600.0
        fig.add_trace(
            go.Scatter(
                x=time_hours,
                y=cr.fuel_flow_kgps * 3600,
                mode='lines',
                name=label,
                line=dict(color=colors[i % len(colors)], width=2)
            ),
            row=2, col=1
        )
    
    # 4. Flight Efficiency (km per kg fuel)
    fig.add_trace(
        go.Bar(
            x=labels,
            y=fuel_efficiency,
            name='Fuel Efficiency',
            marker_color='green',
            text=[f'{eff:.2f} km/kg' for eff in fuel_efficiency],
            textposition='auto'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Cruise Performance Comparison",
        title_x=0.5,
        title_font_size=18,
        showlegend=True,
        height=800,
        width=1200
    )
    
    # Update axes
    fig.update_yaxes(title_text="Fuel Consumed (kg)", row=1, col=1)
    fig.update_yaxes(title_text="Weight Reduction (kg)", row=1, col=2)
    fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
    fig.update_yaxes(title_text="Fuel Flow Rate (kg/h)", row=2, col=1)
    fig.update_yaxes(title_text="Efficiency (km/kg)", row=2, col=2)
    
    fig.show()

def plot_cruise_trajectory_3d(cruise_results: 'CruiseResults') -> None:
    """
    Create 3D trajectory plot showing distance vs time vs fuel consumption.
    
    Args:
        cruise_results: Complete cruise simulation results
    """
    
    time_hours = cruise_results.time_s / 3600.0
    
    fig = go.Figure(data=[go.Scatter3d(
        x=cruise_results.distance_km,
        y=time_hours,
        z=cruise_results.fuel_consumed_kg,
        mode='markers+lines',
        marker=dict(
            size=4,
            color=cruise_results.fuel_flow_kgps * 3600,  # Color by fuel flow rate
            colorscale='Viridis',
            colorbar=dict(title="Fuel Flow Rate (kg/h)"),
            showscale=True
        ),
        line=dict(
            color='blue',
            width=6
        ),
        text=[f'Distance: {d:.0f} km<br>Time: {t:.2f} h<br>Fuel: {f:.1f} kg<br>Flow: {ff*3600:.0f} kg/h'
              for d, t, f, ff in zip(cruise_results.distance_km, time_hours, 
                                   cruise_results.fuel_consumed_kg, cruise_results.fuel_flow_kgps)],
        hovertemplate='%{text}<extra></extra>'
    )])
    
    fig.update_layout(
        title=f'3D Cruise Trajectory<br>Altitude: {cruise_results.initial_state.altitude_m:.0f} m, '
              f'Mach: {cruise_results.initial_state.mach:.3f}',
        scene=dict(
            xaxis_title='Distance (km)',
            yaxis_title='Time (hours)',
            zaxis_title='Fuel Consumed (kg)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1000,
        height=800
    )
    
    fig.show()

def plot_cruise_results(cruise_results: 'CruiseResults', show_browser: bool = True) -> None:
    """
    Main plotting interface function for cruise results.
    
    Args:
        cruise_results: Complete cruise simulation results
        show_browser: Whether to show plots in browser (Plotly) or matplotlib
    """
    
    if show_browser:
        print("[CRUISE-PLOT] Creating interactive Plotly visualizations...")
        plot_cruise_results_plotly(cruise_results)
    else:
        print("[CRUISE-PLOT] Creating matplotlib plots...")
        plot_cruise_results_matplotlib(cruise_results)

def plot_mission_overview(climb_fuel_kg: float, climb_time_hours: float,
                         cruise_results: 'CruiseResults') -> None:
    """
    Create overview plot combining climb and cruise phases.
    
    Args:
        climb_fuel_kg: Fuel consumed during climb phase
        climb_time_hours: Time spent in climb phase
        cruise_results: Complete cruise simulation results
    """
    
    # Prepare data
    phases = ['Climb', 'Cruise']
    fuel_consumption = [climb_fuel_kg, cruise_results.total_fuel_consumed_kg]
    time_spent = [climb_time_hours, cruise_results.total_time_s / 3600.0]
    
    # Create subplots
    fig = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=('Fuel Consumption by Phase', 'Time Distribution'),
        specs=[[{"type": "bar"}, {"type": "pie"}]]
    )
    
    # Fuel consumption bar chart
    fig.add_trace(
        go.Bar(
            x=phases,
            y=fuel_consumption,
            name='Fuel Consumption',
            marker_color=['skyblue', 'lightgreen'],
            text=[f'{fuel:.1f} kg' for fuel in fuel_consumption],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Time distribution pie chart
    fig.add_trace(
        go.Pie(
            labels=phases,
            values=time_spent,
            name='Time Distribution',
            marker_colors=['skyblue', 'lightgreen'],
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{value:.2f} h<br>(%{percent})'
        ),
        row=1, col=2
    )
    
    # Update layout
    total_fuel = sum(fuel_consumption)
    total_time = sum(time_spent)
    
    fig.update_layout(
        title=f'Mission Overview<br>Total Fuel: {total_fuel:.1f} kg, Total Time: {total_time:.2f} hours',
        title_x=0.5,
        title_font_size=16,
        showlegend=False,
        height=500,
        width=1200
    )
    
    fig.update_yaxes(title_text="Fuel Consumed (kg)", row=1, col=1)
    
    fig.show()

def plot_combined_climb_cruise_trajectory(climb_result, cruise_results: 'CruiseResults') -> None:
    """
    Create combined trajectory plot showing both climb and cruise phases.
    
    Args:
        climb_result: ClimbingCore.MinFuelSchedule object from climb optimization
        cruise_results: Complete cruise simulation results
    """
    
    # Extract climb data
    climb_time_s = np.cumsum(np.nan_to_num(climb_result.dt_s, nan=0.0, posinf=0.0, neginf=0.0))
    climb_time_hours = climb_time_s / 3600.0
    climb_alt_m = np.asarray(climb_result.alt_m, float)
    climb_mach = np.asarray(climb_result.mach, float)
    climb_fuel_kg = np.asarray(climb_result.cumFuel_kg, float)
    climb_lever = np.asarray(climb_result.lever, float)
    
    # Extract cruise data (shift time to continue from climb)
    climb_end_time_hours = climb_time_hours[-1] if len(climb_time_hours) > 0 else 0.0
    cruise_time_hours = (cruise_results.time_s / 3600.0) + climb_end_time_hours
    cruise_alt_m = cruise_results.altitude_m
    cruise_mach = cruise_results.mach_number
    cruise_fuel_kg = cruise_results.fuel_consumed_kg + climb_fuel_kg[-1]  # Add climb fuel
    cruise_lever = cruise_results.lever_position
    
    # Create comprehensive trajectory plot
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=('Altitude Profile', 'Mach Number Profile', 
                       'Fuel Consumption', 'Engine Lever Position'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Altitude Profile
    fig.add_trace(
        go.Scatter(
            x=climb_time_hours,
            y=climb_alt_m,
            mode='lines',
            name='Climb Phase',
            line=dict(color='blue', width=3),
            hovertemplate='Time: %{x:.2f} h<br>Altitude: %{y:.0f} m<br>Phase: Climb<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=cruise_time_hours,
            y=cruise_alt_m,
            mode='lines',
            name='Cruise Phase',
            line=dict(color='green', width=3),
            hovertemplate='Time: %{x:.2f} h<br>Altitude: %{y:.0f} m<br>Phase: Cruise<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Mach Number Profile
    fig.add_trace(
        go.Scatter(
            x=climb_time_hours,
            y=climb_mach,
            mode='lines',
            name='Climb Mach',
            line=dict(color='blue', width=3),
            showlegend=False,
            hovertemplate='Time: %{x:.2f} h<br>Mach: %{y:.3f}<br>Phase: Climb<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=cruise_time_hours,
            y=cruise_mach,
            mode='lines',
            name='Cruise Mach',
            line=dict(color='green', width=3),
            showlegend=False,
            hovertemplate='Time: %{x:.2f} h<br>Mach: %{y:.3f}<br>Phase: Cruise<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Fuel Consumption
    fig.add_trace(
        go.Scatter(
            x=climb_time_hours,
            y=climb_fuel_kg,
            mode='lines',
            name='Climb Fuel',
            line=dict(color='blue', width=3),
            showlegend=False,
            hovertemplate='Time: %{x:.2f} h<br>Fuel: %{y:.1f} kg<br>Phase: Climb<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=cruise_time_hours,
            y=cruise_fuel_kg,
            mode='lines',
            name='Cruise Fuel',
            line=dict(color='green', width=3),
            showlegend=False,
            hovertemplate='Time: %{x:.2f} h<br>Fuel: %{y:.1f} kg<br>Phase: Cruise<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Engine Lever Position
    fig.add_trace(
        go.Scatter(
            x=climb_time_hours,
            y=climb_lever * 100,  # Convert to percentage
            mode='lines',
            name='Climb Lever',
            line=dict(color='blue', width=3),
            showlegend=False,
            hovertemplate='Time: %{x:.2f} h<br>Lever: %{y:.1f}%<br>Phase: Climb<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=cruise_time_hours,
            y=cruise_lever * 100,  # Convert to percentage
            mode='lines',
            name='Cruise Lever',
            line=dict(color='green', width=3),
            showlegend=False,
            hovertemplate='Time: %{x:.2f} h<br>Lever: %{y:.1f}%<br>Phase: Cruise<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Add reference lines for lever position
    fig.add_hline(y=85, line_dash="dash", line_color="orange", 
                  annotation_text="MCT Limit (85%)", row=2, col=2)
    fig.add_hline(y=100, line_dash="dash", line_color="red", 
                  annotation_text="Max Thrust (100%)", row=2, col=2)
    
    # Add vertical line to separate phases
    if len(climb_time_hours) > 0:
        phase_transition_time = climb_time_hours[-1]
        for row in range(1, 3):
            for col in range(1, 3):
                fig.add_vline(x=phase_transition_time, line_dash="dot", line_color="red",
                             annotation_text="Cruise Start", row=row, col=col)
    
    # Update layout
    total_time = cruise_time_hours[-1] if len(cruise_time_hours) > 0 else 0.0
    total_fuel = cruise_fuel_kg[-1] if len(cruise_fuel_kg) > 0 else 0.0
    
    fig.update_layout(
        title=f"Complete Mission Trajectory (Climb + Cruise)<br>" + 
              f"Total Time: {total_time:.2f} hours, Total Fuel: {total_fuel:.1f} kg",
        title_x=0.5,
        title_font_size=16,
        showlegend=True,
        height=800,
        width=1200,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (hours)", row=1, col=1)
    fig.update_yaxes(title_text="Altitude (m)", row=1, col=1)
    
    fig.update_xaxes(title_text="Time (hours)", row=1, col=2)
    fig.update_yaxes(title_text="Mach Number", row=1, col=2)
    
    fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Fuel Consumed (kg)", row=2, col=1)
    
    fig.update_xaxes(title_text="Time (hours)", row=2, col=2)
    fig.update_yaxes(title_text="Engine Lever Position (%)", row=2, col=2)
    
    # Show the plot
    fig.show()
