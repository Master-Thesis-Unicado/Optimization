"""
Fuel Optimization Plotting Module

This module provides visualization capabilities for the fuel optimization convergence
process, tracking the evolution of Key Performance Parameters (KPPs) throughout
iterations and creating publication-quality convergence analysis plots.

Features:
- Convergence trajectory visualization
- KPP evolution tracking (fuel, time, weight, efficiency)
- Iteration-by-iteration performance comparison
- Performance improvement quantification
"""

from __future__ import annotations
import numpy as np
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os

# Suppress choreographer JSONError warnings (non-critical browser communication errors)
warnings.filterwarnings('ignore', category=UserWarning, module='choreographer')
try:
    import logging
    logging.getLogger('choreographer').setLevel(logging.ERROR)
except:
    pass

# Set Plotly to open in browser
pio.renderers.default = "browser"

from dataclasses import dataclass
from typing import List, Optional

# Import optimization module
from mission_fuel_optimizer import ConvergenceHistory, MissionIterationResults, SAFETY_BUFFER_PERCENT

# Import aircraft config for reference values
from aircraft_config import MAX_FUEL_KG

# Import visualization config
from visualization_config import (
    Colors, Typography, Layout, LineStyles,
    get_standard_layout, get_or_create_run_directory,
    get_standard_legend, get_axis_config, ExportConfig
)


def plot_convergence_trajectory(history: ConvergenceHistory, save_plots: bool = True):
    """
    Plot the convergence trajectory showing fuel consumption evolution using Plotly.
    Uses bisection bounds and fuel deficit tracking.
    
    Args:
        history: Convergence history containing all iterations
        save_plots: Whether to save plots to disk
    """
    
    if len(history.iterations) < 2:
        print("[WARNING] Need at least 2 iterations to plot convergence")
        return
    
    # Extract data
    iterations = [r.iteration for r in history.iterations]
    initial_fuels = [r.initial_fuel_kg for r in history.iterations]
    consumed_fuels = [r.fuel_consumed_kg for r in history.iterations]
    fuel_deficits = [r.fuel_deficit_kg for r in history.iterations]
    
    # Extract bisection bounds
    fuel_lows = [bounds[0] for bounds in history.fuel_bounds_history]
    fuel_highs = [bounds[1] for bounds in history.fuel_bounds_history]
    bound_ranges = [high - low for low, high in history.fuel_bounds_history]
    
    # Calculate summary metrics
    first_iter = history.iterations[0]
    last_iter = history.iterations[-1]
    optimized_fuel = last_iter.fuel_consumed_kg * (1.0 + SAFETY_BUFFER_PERCENT)
    convergence_iterations = len(history.iterations)
    final_deficit = abs(last_iter.fuel_deficit_kg)
    final_range = bound_ranges[-1] if bound_ranges else 0
    
    # Normalize to first iteration for percentage change
    first_fuel = consumed_fuels[0]
    fuel_percent_change = [(f - first_fuel) / first_fuel * 100 for f in consumed_fuels]
    
    # Calculate fuel savings
    fuel_reduction = MAX_FUEL_KG - optimized_fuel
    fuel_reduction_pct = (fuel_reduction / MAX_FUEL_KG) * 100
    
    # Create summary table data
    summary_data = [
        ['Iterations', f'{convergence_iterations}'],
        ['Final Deficit', f'{final_deficit:.2f} kg'],
        ['Final Range', f'{final_range:.2f} kg'],
        ['', ''],
        ['Initial Capacity', f'{MAX_FUEL_KG:.1f} kg'],
        ['Optimized Capacity', f'{optimized_fuel:.1f} kg'],
        ['Fuel Savings', f'{fuel_reduction:.1f} kg ({fuel_reduction_pct:.1f}%)'],
        ['', ''],
        ['Final Time', f'{last_iter.total_time_s/60:.0f} min'],
        ['Final Weight', f'{last_iter.final_weight_kg:.0f} kg']
    ]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('<b>Bisection Bounds Evolution</b>', '<b>Fuel Deficit Convergence</b>',
                       '<b>Fuel Change Evolution</b>', '<b>Optimization Summary</b>'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "table"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    # Plot 1: Bisection bounds evolution
    fig.add_trace(go.Scatter(
        x=iterations, y=fuel_highs,
        mode='lines+markers',
        name='Upper Bound',
        line=dict(color='red', width=3),
        marker=dict(size=10, symbol='triangle-up'),
        hovertemplate='<b>Iter %{x}</b><br>Upper: %{y:.1f} kg<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=iterations, y=initial_fuels,
        mode='lines+markers',
        name='Tested Fuel',
        line=dict(color=Colors.CRUISE, width=3),
        marker=dict(size=10, symbol='circle'),
        hovertemplate='<b>Iter %{x}</b><br>Tested: %{y:.1f} kg<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=iterations, y=fuel_lows,
        mode='lines+markers',
        name='Lower Bound',
        line=dict(color='blue', width=3),
        marker=dict(size=10, symbol='triangle-down'),
        hovertemplate='<b>Iter %{x}</b><br>Lower: %{y:.1f} kg<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=iterations, y=consumed_fuels,
        mode='lines+markers',
        name='Consumed',
        line=dict(color='green', width=2, dash='dot'),
        marker=dict(size=8),
        hovertemplate='<b>Iter %{x}</b><br>Consumed: %{y:.1f} kg<extra></extra>'
    ), row=1, col=1)
    
    # Plot 2: Fuel deficit (abs value, log scale)
    abs_deficits = [abs(d) for d in fuel_deficits]
    fig.add_trace(go.Scatter(
        x=iterations, y=abs_deficits,
        mode='lines+markers',
        name='Fuel Deficit',
        line=dict(color=Colors.DESCENT, width=3),
        marker=dict(size=10),
        hovertemplate='<b>Iter %{x}</b><br>|Deficit|: %{y:.2f} kg<extra></extra>',
        showlegend=False
    ), row=1, col=2)
    
    # Note: Convergence stopping rule is based on bounds width, not deficit.
    # We intentionally do NOT draw a 10 kg dashed line here to avoid implying
    # that |deficit| < 10 kg is the termination criterion.
    fig.add_annotation(
        xref='x2', yref='y2',
        x=iterations[-1] if len(iterations) > 0 else 1, 
        y=max(abs_deficits) if len(abs_deficits) > 0 else 1,
        text="<b>Note</b>: Stopping uses bounds range < 10 kg;<br>|deficit| is diagnostic only.",
        showarrow=False, align='right',
        xanchor='right', yanchor='top',
        font=dict(size=10, color='gray')
    )
    
    # Plot 3: Performance evolution
    fig.add_trace(go.Scatter(
        x=iterations, y=fuel_percent_change,
        mode='lines+markers',
        name='Fuel Consumption',
        line=dict(color=Colors.CLIMB, width=3),
        marker=dict(size=10),
        hovertemplate='<b>Iter %{x}</b><br>Change: %{y:.2f}%<extra></extra>',
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=iterations, y=[0] * len(iterations),
        mode='lines',
        line=dict(color='black', width=1),
        showlegend=False,
        hoverinfo='skip'
    ), row=2, col=1)
    
    # Plot 4: Summary table
    fig.add_trace(go.Table(
        header=dict(values=['<b>Parameter</b>', '<b>Value</b>'],
                   fill_color='lightblue',
                   align='center',
                   font=dict(size=12, color='black')),
        cells=dict(values=[[row[0] for row in summary_data], [row[1] for row in summary_data]],
                  fill_color='white',
                  align=['left', 'right'],
                  font=dict(size=10),
                  height=25)
    ), row=2, col=2)
    
    # Update axes
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_yaxes(title_text="Fuel (kg)", row=1, col=1)
    
    fig.update_xaxes(title_text="Iteration", row=1, col=2)
    fig.update_yaxes(title_text="|Deficit| (kg)", type="log", row=1, col=2)
    
    fig.update_xaxes(title_text="Iteration", row=2, col=1)
    fig.update_yaxes(title_text="Change from Iter 1 (%)", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>Bisection Fuel Optimization Convergence Analysis</b>",
            x=0.5, xanchor='center',
            font=dict(size=18)
        ),
        height=900, width=1400,
        template='plotly_white',
        showlegend=True,
        # Move legend outside to the right to avoid overlapping with plots
        legend=dict(orientation='v', yanchor='top', y=1.0, xanchor='left', x=1.02),
        margin=dict(r=200)
    )
    
    if save_plots:
        run_dir = get_or_create_run_directory(phase="Optimized")
        html_path = os.path.join(run_dir, 'fuel_convergence.html')
        fig.write_html(html_path)
        print(f"[EXPORT] Convergence analysis saved to: {html_path}")
    
    fig.show()


def plot_optimization_comparison(history: ConvergenceHistory, save_plots: bool = True):
    """
    Create a comprehensive comparison showing optimization results using Plotly.
    
    Shows the optimized mission performance and capacity savings compared to original MAX_FUEL_KG.
    
    Args:
        history: Convergence history containing all iterations
        save_plots: Whether to save plots to disk
    """
    
    if len(history.iterations) < 1:
        print("[WARNING] Need at least 1 iteration for comparison")
        return
    
    # Use the final converged result
    final = history.iterations[-1]
    
    # Calculate optimized capacity with safety buffer
    optimized_capacity = final.fuel_consumed_kg * (1 + SAFETY_BUFFER_PERCENT)
    capacity_reduction = MAX_FUEL_KG - optimized_capacity
    capacity_reduction_pct = (capacity_reduction / MAX_FUEL_KG) * 100
    
    # Phase-wise data for optimized mission
    phases = ['Climb', 'Cruise', 'Descent', 'Total']
    optimized_fuels = [final.climb_fuel_kg, final.cruise_fuel_kg, final.descent_fuel_kg, final.fuel_consumed_kg]
    optimized_times = [final.climb_time_s / 3600, final.cruise_time_s / 3600, final.descent_time_s / 3600, final.total_time_s / 3600]
    
    # Fuel breakdown percentages
    fuel_percentages = [(f / final.fuel_consumed_kg * 100) if final.fuel_consumed_kg > 0 else 0.0 for f in optimized_fuels[:-1]]
    fuel_percentages.append(100.0)  # Total is always 100%
    
    # Create summary table
    summary_data = [
        ['Iterations', f'{len(history.iterations)}'],
        ['Final Deficit', f'{abs(final.fuel_deficit_kg):.2f} kg'],
        ['Convergence Status', 'Equilibrium Achieved' if abs(final.fuel_deficit_kg) < 50 else 'Near Equilibrium'],
        ['', ''],
        ['Original Capacity', f'{MAX_FUEL_KG:.1f} kg'],
        ['Optimized Capacity', f'{optimized_capacity:.1f} kg'],
        ['Capacity Savings', f'{capacity_reduction:.1f} kg ({capacity_reduction_pct:.1f}%)'],
        ['', ''],
        ['Mission Duration', f'{final.total_time_s/3600:.2f} hours'],
        ['Final Weight', f'{final.final_weight_kg:.0f} kg']
    ]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('<b>Capacity Comparison</b>', '<b>Fuel Breakdown by Phase</b>',
                       '<b>Mission Duration by Phase</b>', '<b>Optimization Summary</b>'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "table"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Plot 1: Capacity comparison (Original vs Optimized)
    capacity_labels = ['Original\nCapacity', 'Optimized\nCapacity', 'Savings']
    capacity_values = [MAX_FUEL_KG, optimized_capacity, capacity_reduction]
    capacity_colors = ['lightcoral', 'lightgreen', 'gold']
    
    fig.add_trace(go.Bar(
        x=capacity_labels, 
        y=capacity_values,
        marker_color=capacity_colors,
        text=[f'{v:.1f} kg' for v in capacity_values],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Fuel: %{y:.1f} kg<extra></extra>',
        showlegend=False
    ), row=1, col=1)
    
    # Plot 2: Fuel breakdown by phase (optimized mission)
    fig.add_trace(go.Bar(
        x=phases, 
        y=optimized_fuels,
        marker_color=[Colors.CLIMB, Colors.CRUISE, Colors.DESCENT, 'navy'],
        text=[f'{f:.1f} kg<br>({p:.1f}%)' for f, p in zip(optimized_fuels, fuel_percentages)],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Fuel: %{y:.1f} kg<extra></extra>',
        showlegend=False
    ), row=1, col=2)
    
    # Plot 3: Mission duration by phase (optimized mission)
    fig.add_trace(go.Bar(
        x=phases, 
        y=optimized_times,
        marker_color=[Colors.CLIMB, Colors.CRUISE, Colors.DESCENT, 'navy'],
        text=[f'{t:.2f} h' for t in optimized_times],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Time: %{y:.2f} h<extra></extra>',
        showlegend=False
    ), row=2, col=1)
    
    # Plot 4: Summary table
    fig.add_trace(go.Table(
        header=dict(values=['<b>Parameter</b>', '<b>Value</b>'],
                   fill_color='lightgreen',
                   align='center',
                   font=dict(size=12, color='black')),
        cells=dict(values=[[row[0] for row in summary_data], [row[1] for row in summary_data]],
                  fill_color='white',
                  align=['left', 'right'],
                  font=dict(size=10),
                  height=25)
    ), row=2, col=2)
    
    # Update axes
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="Fuel Capacity (kg)", row=1, col=1)
    
    fig.update_xaxes(title_text="Phase", row=1, col=2)
    fig.update_yaxes(title_text="Fuel Consumed (kg)", row=1, col=2)
    
    fig.update_xaxes(title_text="Phase", row=2, col=1)
    fig.update_yaxes(title_text="Duration (hours)", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>Optimization Results Summary</b><br><sup>Optimized Mission Performance</sup>",
            x=0.5, xanchor='center',
            font=dict(size=18)
        ),
        height=900, width=1400,
        template='plotly_white',
        showlegend=False,
        barmode='group'
    )
    
    if save_plots:
        run_dir = get_or_create_run_directory(phase="Optimized")
        html_path = os.path.join(run_dir, 'optimization_results_summary.html')
        fig.write_html(html_path)
        print(f"[EXPORT] Optimization results summary saved to: {html_path}")
    
    fig.show()


def plot_aerodynamic_performance_analysis(history: ConvergenceHistory, save_plots: bool = True):
    """
    Comprehensive aerodynamic performance analysis combining L/D ratios and thrust lever evolution.
    
    Args:
        history: Convergence history containing all iterations
        save_plots: Whether to save plots to disk
    """
    
    if len(history.iterations) < 2:
        print("[WARNING] Need at least 2 iterations for aerodynamic analysis")
        return
    
    iterations = [r.iteration for r in history.iterations]
    
    # Extract L/D ratios
    ld_climb = [r.avg_ld_climb for r in history.iterations]
    ld_cruise = [r.avg_ld_cruise for r in history.iterations]
    ld_descent = [r.avg_ld_descent for r in history.iterations]
    
    # Extract thrust lever positions
    lever_climb = [r.avg_lever_climb for r in history.iterations]
    lever_cruise = [r.avg_lever_cruise for r in history.iterations]
    lever_descent = [r.avg_lever_descent for r in history.iterations]
    
    # Create subplot figure with 2 rows, 1 column
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Aerodynamic Efficiency (L/D Ratio) Evolution', 'Thrust Lever Position Evolution'),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # L/D Ratio traces (subplot 1)
    fig.add_trace(go.Scatter(
        x=iterations, y=ld_climb,
        mode='lines+markers',
        name='Climb L/D',
        line=dict(color=Colors.CLIMB, width=3),
        marker=dict(size=10),
        hovertemplate='<b>Iteration %{x}</b><br>Climb L/D: %{y:.2f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=iterations, y=ld_cruise,
        mode='lines+markers',
        name='Cruise L/D',
        line=dict(color=Colors.CRUISE, width=3),
        marker=dict(size=10),
        hovertemplate='<b>Iteration %{x}</b><br>Cruise L/D: %{y:.2f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=iterations, y=ld_descent,
        mode='lines+markers',
        name='Descent L/D',
        line=dict(color=Colors.DESCENT, width=3),
        marker=dict(size=10),
        hovertemplate='<b>Iteration %{x}</b><br>Descent L/D: %{y:.2f}<extra></extra>'
    ), row=1, col=1)
    
    # Thrust Lever traces (subplot 2)
    fig.add_trace(go.Scatter(
        x=iterations, y=lever_climb,
        mode='lines+markers',
        name='Climb Lever',
        line=dict(color=Colors.CLIMB, width=3, dash='dot'),
        marker=dict(size=10),
        hovertemplate='<b>Iteration %{x}</b><br>Climb Lever: %{y:.3f}<extra></extra>'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=iterations, y=lever_cruise,
        mode='lines+markers',
        name='Cruise Lever',
        line=dict(color=Colors.CRUISE, width=3, dash='dot'),
        marker=dict(size=10),
        hovertemplate='<b>Iteration %{x}</b><br>Cruise Lever: %{y:.3f}<extra></extra>'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=iterations, y=lever_descent,
        mode='lines+markers',
        name='Descent Lever',
        line=dict(color=Colors.DESCENT, width=3, dash='dot'),
        marker=dict(size=10),
        hovertemplate='<b>Iteration %{x}</b><br>Descent Lever: %{y:.3f}<extra></extra>'
    ), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>Aerodynamic Performance Analysis</b><br><sup>L/D Ratios and Thrust Lever Evolution</sup>",
            x=0.5, xanchor='center',
            font=dict(size=18)
        ),
        height=900, width=1200,
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation='v', yanchor='top', y=0.98, xanchor='right', x=0.98)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=2, col=1)
    fig.update_yaxes(title_text="L/D Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Lever Position (0-1)", row=2, col=1)
    
    if save_plots:
        run_dir = get_or_create_run_directory(phase="Optimized")
        html_path = os.path.join(run_dir, 'aerodynamic_performance_analysis.html')
        fig.write_html(html_path)
        print(f"[EXPORT] Aerodynamic performance analysis saved to: {html_path}")
    
    fig.show()


def plot_3d_trajectory_comparison(history: ConvergenceHistory, save_plots: bool = True):
    """
    Plot 3D trajectory comparison (altitude-distance-time) for first vs final iteration.
    
    Args:
        history: Convergence history containing all iterations
        save_plots: Whether to save plots to disk
    """
    
    if len(history.iterations) < 2:
        print("[WARNING] Need at least 2 iterations for trajectory comparison")
        return
    
    first = history.iterations[0]
    final = history.iterations[-1]
    
    # Create figure
    fig = go.Figure()
    
    # First iteration trajectory
    # Climb
    climb_dist_first = np.linspace(0, 50, len(first.climb_result.alt_m))  # Estimated distance in km
    climb_time_first = np.cumsum(first.climb_result.dt_s) / 60  # Minutes
    
    fig.add_trace(go.Scatter3d(
        x=climb_dist_first,
        y=climb_time_first,
        z=first.climb_result.alt_m,
        mode='lines',
        name='Iter 1 - Climb',
        line=dict(color=Colors.CLIMB, width=6),
        hovertemplate='<b>Iteration 1 - Climb</b><br>Distance: %{x:.1f} km<br>Time: %{y:.1f} min<br>Altitude: %{z:.0f} m<extra></extra>'
    ))
    
    # Cruise
    cruise_dist_first = np.linspace(climb_dist_first[-1], climb_dist_first[-1] + first.cruise_result.distance_km[-1], 
                                   len(first.cruise_result.distance_km))
    cruise_time_first = climb_time_first[-1] + np.array(first.cruise_result.time_s) / 60
    cruise_alt_first = first.cruise_result.altitude_m
    
    fig.add_trace(go.Scatter3d(
        x=cruise_dist_first,
        y=cruise_time_first,
        z=cruise_alt_first,
        mode='lines',
        name='Iter 1 - Cruise',
        line=dict(color=Colors.CRUISE, width=6),
        hovertemplate='<b>Iteration 1 - Cruise</b><br>Distance: %{x:.1f} km<br>Time: %{y:.1f} min<br>Altitude: %{z:.0f} m<extra></extra>'
    ))
    
    # Descent
    if hasattr(first.descent_result, 'alt_m') and len(first.descent_result.alt_m) > 0:
        descent_dist_first = np.linspace(cruise_dist_first[-1], 
                                         cruise_dist_first[-1] + 50,
                                         len(first.descent_result.alt_m))
        descent_time_first = cruise_time_first[-1] + np.cumsum(first.descent_result.dt_s) / 60
        
        fig.add_trace(go.Scatter3d(
            x=descent_dist_first,
            y=descent_time_first,
            z=first.descent_result.alt_m,
            mode='lines',
            name='Iter 1 - Descent',
            line=dict(color=Colors.DESCENT, width=6),
            hovertemplate='<b>Iteration 1 - Descent</b><br>Distance: %{x:.1f} km<br>Time: %{y:.1f} min<br>Altitude: %{z:.0f} m<extra></extra>'
        ))
    
    # Final iteration trajectory
    # Climb
    climb_dist_final = np.linspace(0, 50, len(final.climb_result.alt_m))
    climb_time_final = np.cumsum(final.climb_result.dt_s) / 60
    
    fig.add_trace(go.Scatter3d(
        x=climb_dist_final,
        y=climb_time_final,
        z=final.climb_result.alt_m,
        mode='lines',
        name=f'Iter {final.iteration} - Climb',
        line=dict(color=Colors.CLIMB, width=6, dash='dash'),
        hovertemplate=f'<b>Iteration {final.iteration} - Climb</b><br>Distance: %{{x:.1f}} km<br>Time: %{{y:.1f}} min<br>Altitude: %{{z:.0f}} m<extra></extra>'
    ))
    
    # Cruise
    cruise_dist_final = np.linspace(climb_dist_final[-1], climb_dist_final[-1] + final.cruise_result.distance_km[-1],
                                   len(final.cruise_result.distance_km))
    cruise_time_final = climb_time_final[-1] + np.array(final.cruise_result.time_s) / 60
    cruise_alt_final = final.cruise_result.altitude_m
    
    fig.add_trace(go.Scatter3d(
        x=cruise_dist_final,
        y=cruise_time_final,
        z=cruise_alt_final,
        mode='lines',
        name=f'Iter {final.iteration} - Cruise',
        line=dict(color=Colors.CRUISE, width=6, dash='dash'),
        hovertemplate=f'<b>Iteration {final.iteration} - Cruise</b><br>Distance: %{{x:.1f}} km<br>Time: %{{y:.1f}} min<br>Altitude: %{{z:.0f}} m<extra></extra>'
    ))
    
    # Descent
    if hasattr(final.descent_result, 'alt_m') and len(final.descent_result.alt_m) > 0:
        descent_dist_final = np.linspace(cruise_dist_final[-1],
                                         cruise_dist_final[-1] + 50,
                                         len(final.descent_result.alt_m))
        descent_time_final = cruise_time_final[-1] + np.cumsum(final.descent_result.dt_s) / 60
        
        fig.add_trace(go.Scatter3d(
            x=descent_dist_final,
            y=descent_time_final,
            z=final.descent_result.alt_m,
            mode='lines',
            name=f'Iter {final.iteration} - Descent',
            line=dict(color=Colors.DESCENT, width=6, dash='dash'),
            hovertemplate=f'<b>Iteration {final.iteration} - Descent</b><br>Distance: %{{x:.1f}} km<br>Time: %{{y:.1f}} min<br>Altitude: %{{z:.0f}} m<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text="<b>3D Mission Trajectory Comparison</b><br><sup>First vs Final Iteration (Solid: Iter 1, Dashed: Final)</sup>",
            x=0.5, xanchor='center',
            font=dict(size=18)
        ),
        scene=dict(
            xaxis_title="Distance (km)",
            yaxis_title="Time (min)",
            zaxis_title="Altitude (m)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=800, width=1200,
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation='v', yanchor='top', y=0.98, xanchor='left', x=0.02)
    )
    
    if save_plots:
        run_dir = get_or_create_run_directory(phase="Optimized")
        html_path = os.path.join(run_dir, '3d_trajectory_comparison.html')
        fig.write_html(html_path)
        print(f"[EXPORT] 3D trajectory comparison saved to: {html_path}")
    
    fig.show()


def plot_specific_energy_evolution(history: ConvergenceHistory, save_plots: bool = True):
    """
    Plot specific energy and energy height evolution across iterations.
    
    Args:
        history: Convergence history containing all iterations
        save_plots: Whether to save plots to disk
    """
    
    if len(history.iterations) < 2:
        print("[WARNING] Need at least 2 iterations to plot specific energy evolution")
        return
    
    iterations = [r.iteration for r in history.iterations]
    
    # Extract specific energies (J/kg)
    se_climb = [r.avg_specific_energy_climb_J_kg / 1000 for r in history.iterations]  # Convert to kJ/kg
    se_cruise = [r.avg_specific_energy_cruise_J_kg / 1000 for r in history.iterations]
    se_descent = [r.avg_specific_energy_descent_J_kg / 1000 for r in history.iterations]
    
    # Convert to energy height (m) = Specific Energy / g
    eh_climb = [r.avg_specific_energy_climb_J_kg / 9.81 for r in history.iterations]
    eh_cruise = [r.avg_specific_energy_cruise_J_kg / 9.81 for r in history.iterations]
    eh_descent = [r.avg_specific_energy_descent_J_kg / 9.81 for r in history.iterations]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('<b>Specific Energy Evolution</b>', '<b>Energy Height Evolution</b>'),
        horizontal_spacing=0.12
    )
    
    # Plot 1: Specific Energy
    fig.add_trace(go.Scatter(
        x=iterations, y=se_climb,
        mode='lines+markers',
        name='Climb',
        line=dict(color=Colors.CLIMB, width=3),
        marker=dict(size=10),
        hovertemplate='<b>Climb</b><br>Iter: %{x}<br>SE: %{y:.1f} kJ/kg<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=iterations, y=se_cruise,
        mode='lines+markers',
        name='Cruise',
        line=dict(color=Colors.CRUISE, width=3),
        marker=dict(size=10),
        hovertemplate='<b>Cruise</b><br>Iter: %{x}<br>SE: %{y:.1f} kJ/kg<extra></extra>',
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=iterations, y=se_descent,
        mode='lines+markers',
        name='Descent',
        line=dict(color=Colors.DESCENT, width=3),
        marker=dict(size=10),
        hovertemplate='<b>Descent</b><br>Iter: %{x}<br>SE: %{y:.1f} kJ/kg<extra></extra>',
        showlegend=False
    ), row=1, col=1)
    
    # Plot 2: Energy Height
    fig.add_trace(go.Scatter(
        x=iterations, y=eh_climb,
        mode='lines+markers',
        name='Climb',
        line=dict(color=Colors.CLIMB, width=3),
        marker=dict(size=10),
        hovertemplate='<b>Climb</b><br>Iter: %{x}<br>EH: %{y:.0f} m<extra></extra>',
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=iterations, y=eh_cruise,
        mode='lines+markers',
        name='Cruise',
        line=dict(color=Colors.CRUISE, width=3),
        marker=dict(size=10),
        hovertemplate='<b>Cruise</b><br>Iter: %{x}<br>EH: %{y:.0f} m<extra></extra>',
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=iterations, y=eh_descent,
        mode='lines+markers',
        name='Descent',
        line=dict(color=Colors.DESCENT, width=3),
        marker=dict(size=10),
        hovertemplate='<b>Descent</b><br>Iter: %{x}<br>EH: %{y:.0f} m<extra></extra>',
        showlegend=False
    ), row=1, col=2)
    
    # Update axes
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_yaxes(title_text="Specific Energy (kJ/kg)", row=1, col=1)
    
    fig.update_xaxes(title_text="Iteration", row=1, col=2)
    fig.update_yaxes(title_text="Energy Height (m)", row=1, col=2)
    
    fig.update_layout(
        title=dict(
            text="<b>Specific Energy & Energy Height Evolution</b><br><sup>Total Energy State = Potential + Kinetic Energy</sup>",
            x=0.5, xanchor='center',
            font=dict(size=18)
        ),
        height=600, width=1400,
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation='v', yanchor='top', y=0.98, xanchor='right', x=0.48)
    )
    
    if save_plots:
        run_dir = get_or_create_run_directory(phase="Optimized")
        html_path = os.path.join(run_dir, 'specific_energy_evolution.html')
        fig.write_html(html_path)
        print(f"[EXPORT] Specific energy evolution saved to: {html_path}")
    
    fig.show()


def visualize_convergence_analysis(history: ConvergenceHistory, save_plots: bool = True):
    """
    Master function to create convergence analysis visualizations.
    
    Args:
        history: Convergence history containing all iterations
        save_plots: Whether to save plots to disk
    """
    
    print("\n[PLOTTING] Creating convergence analysis visualizations...")
    
    # Create consolidated plots (simplified for bisection method)
    plot_convergence_trajectory(history, save_plots)
    plot_optimization_comparison(history, save_plots)
    
    # Advanced plots removed: aerodynamic metrics, specific energy, KPP evolution, etc.
    # These require performance calculator which was removed in bisection refactoring
    # Note: plot_3d_trajectory_comparison also disabled (requires deleted fields)
    
    print("[PLOTTING] Convergence analysis complete!")

