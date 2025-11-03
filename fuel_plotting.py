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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os

# Set Plotly to open in browser
pio.renderers.default = "browser"

from dataclasses import dataclass
from typing import List, Optional

# Import optimization module
from fuel_optimizer import ConvergenceHistory, MissionIterationResults, SAFETY_BUFFER_PERCENT

# Import visualization config
from visualization_config import (
    Colors, Typography, Layout, LineStyles,
    get_standard_layout, get_or_create_run_directory,
    get_standard_legend, get_axis_config, ExportConfig
)


def plot_convergence_trajectory(history: ConvergenceHistory, save_plots: bool = True):
    """
    Plot the convergence trajectory showing fuel consumption evolution using Plotly.
    
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
    deltas = [abs(r.convergence_delta_percent) for r in history.iterations[1:]]
    delta_iterations = [r.iteration for r in history.iterations[1:]]
    
    # Calculate summary metrics
    first_iter = history.iterations[0]
    last_iter = history.iterations[-1]
    # Use A320 typical maximum fuel capacity as baseline for comparison
    original_max_fuel = 23860.0  # A320 maximum fuel capacity in kg
    optimized_fuel = last_iter.fuel_consumed_kg * (1.0 + SAFETY_BUFFER_PERCENT)
    fuel_savings = original_max_fuel - optimized_fuel
    percent_savings = (fuel_savings / original_max_fuel) * 100
    convergence_iterations = len(history.iterations)
    final_convergence = abs(last_iter.convergence_delta_percent) if convergence_iterations > 1 else 0
    
    # Normalize to first iteration for percentage change
    first_fuel = consumed_fuels[0]
    fuel_percent_change = [(f - first_fuel) / first_fuel * 100 for f in consumed_fuels]
    
    # Create summary table data
    summary_data = [
        ['Iterations', f'{convergence_iterations}'],
        ['Final Delta', f'{final_convergence:.4f}%'],
        ['', ''],
        ['A320 Max Capacity', f'{original_max_fuel:.1f} kg'],
        ['Optimized Capacity', f'{optimized_fuel:.1f} kg'],
        ['Fuel Saved', f'{fuel_savings:.1f} kg ({percent_savings:.1f}%)'],
        ['', ''],
        ['Final Time', f'{last_iter.total_time_s/60:.0f} min'],
        ['Total Distance', f'{last_iter.total_distance_km:.0f} km'],
        ['Final Weight', f'{last_iter.final_weight_kg:.0f} kg']
    ]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('<b>Fuel Capacity Evolution</b>', '<b>Convergence Trajectory</b>',
                       '<b>Performance Evolution</b>', '<b>Optimization Summary</b>'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "table"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    # Plot 1: Fuel evolution
    fig.add_trace(go.Scatter(
        x=iterations, y=initial_fuels,
        mode='lines+markers',
        name='Initial Fuel Estimate',
        line=dict(color=Colors.CLIMB, width=3),
        marker=dict(size=10, symbol='circle'),
        hovertemplate='<b>Iter %{x}</b><br>Initial: %{y:.1f} kg<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=iterations, y=consumed_fuels,
        mode='lines+markers',
        name='Consumed Fuel',
        line=dict(color=Colors.CRUISE, width=3),
        marker=dict(size=10, symbol='square'),
        hovertemplate='<b>Iter %{x}</b><br>Consumed: %{y:.1f} kg<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=iterations, y=[consumed_fuels[-1]] * len(iterations),
        mode='lines',
        name='Converged Value',
        line=dict(color='red', width=2, dash='dash'),
        hovertemplate='Converged: %{y:.1f} kg<extra></extra>'
    ), row=1, col=1)
    
    # Plot 2: Convergence delta (log scale)
    fig.add_trace(go.Scatter(
        x=delta_iterations, y=deltas,
        mode='lines+markers',
        name='Convergence Delta',
        line=dict(color=Colors.DESCENT, width=3),
        marker=dict(size=10),
        hovertemplate='<b>Iter %{x}</b><br>Delta: %{y:.4f}%<extra></extra>',
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=delta_iterations, y=[0.1] * len(delta_iterations),
        mode='lines',
        name='0.1% Tolerance',
        line=dict(color='green', width=2, dash='dash'),
        hovertemplate='Tolerance: 0.1%<extra></extra>',
        showlegend=False
    ), row=1, col=2)
    
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
    fig.update_yaxes(title_text="Delta (%)", type="log", row=1, col=2)
    
    fig.update_xaxes(title_text="Iteration", row=2, col=1)
    fig.update_yaxes(title_text="Change from Iter 1 (%)", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>Fuel Capacity Optimization Convergence Analysis</b>",
            x=0.5, xanchor='center',
            font=dict(size=18)
        ),
        height=900, width=1400,
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation='v', yanchor='top', y=0.98, xanchor='right', x=0.48)
    )
    
    if save_plots:
        run_dir = get_or_create_run_directory(phase="Optimized")
        html_path = os.path.join(run_dir, 'fuel_convergence.html')
        fig.write_html(html_path)
        print(f"[EXPORT] Convergence analysis saved to: {html_path}")
    
    fig.show()


def plot_kpp_evolution(history: ConvergenceHistory, save_plots: bool = True):
    """
    Plot evolution of Key Performance Parameters across iterations.
    
    Args:
        history: Convergence history containing all iterations
        save_plots: Whether to save plots to disk
    """
    
    if len(history.iterations) < 2:
        print("[WARNING] Need at least 2 iterations to plot KPP evolution")
        return
    
    # Extract KPPs
    iterations = [r.iteration for r in history.iterations]
    
    climb_fuels = [r.climb_fuel_kg for r in history.iterations]
    cruise_fuels = [r.cruise_fuel_kg for r in history.iterations]
    descent_fuels = [r.descent_fuel_kg for r in history.iterations]
    
    climb_times = [r.climb_time_s / 60 for r in history.iterations]
    cruise_times = [r.cruise_time_s / 60 for r in history.iterations]
    descent_times = [r.descent_time_s / 60 for r in history.iterations]
    
    initial_masses = [r.initial_mass_kg for r in history.iterations]
    final_weights = [r.final_weight_kg for r in history.iterations]
    
    # Extract aerodynamic data
    avg_lift_climb = [r.avg_lift_climb_N / 1000 for r in history.iterations]  # Convert to kN
    avg_drag_climb = [r.avg_drag_climb_N / 1000 for r in history.iterations]
    avg_lift_cruise = [r.avg_lift_cruise_N / 1000 for r in history.iterations]
    avg_drag_cruise = [r.avg_drag_cruise_N / 1000 for r in history.iterations]
    avg_lift_descent = [r.avg_lift_descent_N / 1000 for r in history.iterations]
    avg_drag_descent = [r.avg_drag_descent_N / 1000 for r in history.iterations]
    
    # Calculate efficiencies
    fuel_efficiencies = []
    for r in history.iterations:
        if r.total_distance_km > 0:
            fuel_efficiencies.append(r.fuel_consumed_kg / r.total_distance_km)
        else:
            fuel_efficiencies.append(0.0)
    
    # Create interactive Plotly figure with lift and drag
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            '<b>Phase Fuel Consumption</b>',
            '<b>Phase Duration</b>',
            '<b>Average Lift Evolution</b>',
            '<b>Average Drag Evolution</b>',
            '<b>Weight Evolution</b>',
            '<b>Fuel Efficiency</b>',
            '<b>Total Fuel vs Initial Mass</b>',
            '<b>Mission Overview</b>'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "table"}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.10
    )
    
    # Plot 1: Phase fuel consumption
    fig.add_trace(
        go.Scatter(x=iterations, y=climb_fuels, mode='lines+markers', name='Climb',
                   line=dict(color=Colors.CLIMB, width=3),
                   marker=dict(size=8)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=cruise_fuels, mode='lines+markers', name='Cruise',
                   line=dict(color=Colors.CRUISE, width=3),
                   marker=dict(size=8)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=descent_fuels, mode='lines+markers', name='Descent',
                   line=dict(color=Colors.DESCENT, width=3),
                   marker=dict(size=8)),
        row=1, col=1
    )
    
    # Plot 2: Phase duration
    fig.add_trace(
        go.Scatter(x=iterations, y=climb_times, mode='lines+markers', name='Climb Time',
                   line=dict(color=Colors.CLIMB, width=3),
                   marker=dict(size=8), showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=cruise_times, mode='lines+markers', name='Cruise Time',
                   line=dict(color=Colors.CRUISE, width=3),
                   marker=dict(size=8), showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=descent_times, mode='lines+markers', name='Descent Time',
                   line=dict(color=Colors.DESCENT, width=3),
                   marker=dict(size=8), showlegend=False),
        row=1, col=2
    )
    
    # Plot 3: Average Lift Evolution
    fig.add_trace(
        go.Scatter(x=iterations, y=avg_lift_climb, mode='lines+markers', name='Climb Lift',
                   line=dict(color=Colors.CLIMB, width=3),
                   marker=dict(size=8)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=avg_lift_cruise, mode='lines+markers', name='Cruise Lift',
                   line=dict(color=Colors.CRUISE, width=3),
                   marker=dict(size=8)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=avg_lift_descent, mode='lines+markers', name='Descent Lift',
                   line=dict(color=Colors.DESCENT, width=3),
                   marker=dict(size=8)),
        row=2, col=1
    )
    
    # Plot 4: Average Drag Evolution
    fig.add_trace(
        go.Scatter(x=iterations, y=avg_drag_climb, mode='lines+markers', name='Climb Drag',
                   line=dict(color=Colors.CLIMB, width=3),
                   marker=dict(size=8), showlegend=False),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=avg_drag_cruise, mode='lines+markers', name='Cruise Drag',
                   line=dict(color=Colors.CRUISE, width=3),
                   marker=dict(size=8), showlegend=False),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=avg_drag_descent, mode='lines+markers', name='Descent Drag',
                   line=dict(color=Colors.DESCENT, width=3),
                   marker=dict(size=8), showlegend=False),
        row=2, col=2
    )
    
    # Plot 5: Weight evolution
    fig.add_trace(
        go.Scatter(x=iterations, y=initial_masses, mode='lines+markers', name='Initial Mass',
                   line=dict(color='blue', width=3), marker=dict(size=8)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=final_weights, mode='lines+markers', name='Final Weight',
                   line=dict(color='green', width=3), marker=dict(size=8)),
        row=3, col=1
    )
    
    # Plot 6: Fuel efficiency
    fig.add_trace(
        go.Scatter(x=iterations, y=fuel_efficiencies, mode='lines+markers',
                   line=dict(color='purple', width=3), marker=dict(size=8),
                   name='Fuel per Distance'),
        row=3, col=2
    )
    
    # Plot 7: Fuel vs mass correlation
    consumed_fuels = [r.fuel_consumed_kg for r in history.iterations]
    fig.add_trace(
        go.Scatter(x=initial_masses, y=consumed_fuels, mode='lines+markers',
                   line=dict(color='orange', width=3), marker=dict(size=8),
                   name='Fuel vs Mass'),
        row=4, col=1
    )
    
    # Plot 8: Enhanced Summary table
    last_result = history.iterations[-1]
    first_result = history.iterations[0]
    
    optimized_capacity = last_result.fuel_consumed_kg * (1 + SAFETY_BUFFER_PERCENT)
    fuel_savings = first_result.initial_fuel_kg - optimized_capacity
    percent_savings = (fuel_savings / first_result.initial_fuel_kg) * 100
    
    # Calculate phase percentages
    total_fuel = last_result.fuel_consumed_kg
    climb_pct = (last_result.climb_fuel_kg / total_fuel) * 100
    cruise_pct = (last_result.cruise_fuel_kg / total_fuel) * 100
    descent_pct = (last_result.descent_fuel_kg / total_fuel) * 100
    
    # Calculate averages
    avg_speed_kmh = last_result.total_distance_km / (last_result.total_time_s / 3600)
    fuel_efficiency = last_result.fuel_consumed_kg / last_result.total_distance_km
    
    summary_data = [
        ['━━━ CONVERGENCE ━━━', ''],
        ['Iterations', f'{len(history.iterations)}'],
        ['Final Delta', f'{abs(last_result.convergence_delta_percent):.4f}%'],
        ['', ''],
        ['━━━ FUEL OPTIMIZATION ━━━', ''],
        ['Original Capacity', f'{first_result.initial_fuel_kg:.1f} kg'],
        ['Optimized Capacity', f'{optimized_capacity:.1f} kg'],
        ['Fuel Savings', f'{fuel_savings:.1f} kg ({percent_savings:.1f}%)'],
        ['', ''],
        ['━━━ PHASE BREAKDOWN ━━━', ''],
        ['Climb', f'{last_result.climb_fuel_kg:.1f} kg ({climb_pct:.1f}%)'],
        ['Cruise', f'{last_result.cruise_fuel_kg:.1f} kg ({cruise_pct:.1f}%)'],
        ['Descent', f'{last_result.descent_fuel_kg:.2f} kg ({descent_pct:.1f}%)'],
        ['Total Consumed', f'{last_result.fuel_consumed_kg:.1f} kg'],
        ['', ''],
        ['━━━ MISSION PERFORMANCE ━━━', ''],
        ['Total Time', f'{last_result.total_time_s/3600:.2f} h ({last_result.total_time_s/60:.0f} min)'],
        ['Distance', f'{last_result.total_distance_km:.0f} km'],
        ['Avg Speed', f'{avg_speed_kmh:.0f} km/h'],
        ['Fuel Efficiency', f'{fuel_efficiency:.3f} kg/km'],
        ['', ''],
        ['━━━ WEIGHT ANALYSIS ━━━', ''],
        ['Initial Mass', f'{last_result.initial_mass_kg:.0f} kg'],
        ['Final Weight', f'{last_result.final_weight_kg:.0f} kg'],
        ['Mass Reduction', f'{last_result.initial_mass_kg - last_result.final_weight_kg:.0f} kg']
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(values=['<b>Parameter</b>', '<b>Value</b>'],
                       fill_color='paleturquoise',
                       align='center',
                       font=dict(size=12, color='black', family='Arial')),
            cells=dict(values=[[row[0] for row in summary_data], [row[1] for row in summary_data]],
                      fill_color='white',
                      align=['left', 'right'],
                      font=dict(size=10, family='Arial'),
                      height=25)  # Compact row height to fit more data
        ),
        row=4, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_yaxes(title_text="Fuel (kg)", row=1, col=1)
    
    fig.update_xaxes(title_text="Iteration", row=1, col=2)
    fig.update_yaxes(title_text="Time (min)", row=1, col=2)
    
    fig.update_xaxes(title_text="Iteration", row=2, col=1)
    fig.update_yaxes(title_text="Lift (kN)", row=2, col=1)
    
    fig.update_xaxes(title_text="Iteration", row=2, col=2)
    fig.update_yaxes(title_text="Drag (kN)", row=2, col=2)
    
    fig.update_xaxes(title_text="Iteration", row=3, col=1)
    fig.update_yaxes(title_text="Weight (kg)", row=3, col=1)
    
    fig.update_xaxes(title_text="Iteration", row=3, col=2)
    fig.update_yaxes(title_text="kg/km", row=3, col=2)
    
    fig.update_xaxes(title_text="Initial Mass (kg)", row=4, col=1)
    fig.update_yaxes(title_text="Consumed Fuel (kg)", row=4, col=1)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>Key Performance Parameter (KPP) Evolution with Aerodynamics</b>",
            x=0.5, xanchor='center',
            font=dict(size=18)
        ),
        height=1400, width=1600,  # Increased height for 4 rows
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    if save_plots:
        run_dir = get_or_create_run_directory(phase="Optimized")
        html_path = os.path.join(run_dir, 'kpp_evolution.html')
        png_path = os.path.join(run_dir, 'kpp_evolution.png')
        fig.write_html(html_path)
        try:
            fig.write_image(png_path, width=1600, height=1000, scale=2)
            print(f"[EXPORT] KPP evolution saved to: {html_path} and {png_path}")
        except Exception as e:
            print(f"[EXPORT] KPP evolution saved to: {html_path} (HTML only)")
            print(f"[WARNING] Could not save PNG: {e}")
    
    fig.show()


def plot_optimization_comparison(history: ConvergenceHistory, save_plots: bool = True):
    """
    Create a comprehensive comparison of first vs final iteration performance using Plotly.
    
    Args:
        history: Convergence history containing all iterations
        save_plots: Whether to save plots to disk
    """
    
    if len(history.iterations) < 2:
        print("[WARNING] Need at least 2 iterations for comparison")
        return
    
    first = history.iterations[0]
    final = history.iterations[-1]
    
    # Calculate data
    phases = ['Climb', 'Cruise', 'Descent', 'Total']
    first_fuels = [first.climb_fuel_kg, first.cruise_fuel_kg, first.descent_fuel_kg, first.fuel_consumed_kg]
    final_fuels = [final.climb_fuel_kg, final.cruise_fuel_kg, final.descent_fuel_kg, final.fuel_consumed_kg]
    
    first_times = [first.climb_time_s / 3600, first.cruise_time_s / 3600, first.descent_time_s / 3600, first.total_time_s / 3600]
    final_times = [final.climb_time_s / 3600, final.cruise_time_s / 3600, final.descent_time_s / 3600, final.total_time_s / 3600]
    
    improvements = [(f - fin) / f * 100 if f > 0 else 0.0 for f, fin in zip(first_fuels, final_fuels)]
    
    # Calculate summary metrics
    optimized_capacity = final.fuel_consumed_kg * (1 + SAFETY_BUFFER_PERCENT)
    # Use A320 typical maximum fuel capacity as baseline for comparison
    original_max_capacity = 23860.0  # A320 maximum fuel capacity in kg
    capacity_savings = original_max_capacity - optimized_capacity
    capacity_savings_pct = (capacity_savings / original_max_capacity) * 100
    fuel_reduction = first.fuel_consumed_kg - final.fuel_consumed_kg
    fuel_reduction_pct = (fuel_reduction / first.fuel_consumed_kg) * 100
    mass_reduction = first.initial_mass_kg - final.initial_mass_kg
    mass_reduction_pct = (mass_reduction / first.initial_mass_kg) * 100
    time_diff = final.total_time_s - first.total_time_s
    first_efficiency = first.fuel_consumed_kg / first.total_distance_km
    final_efficiency = final.fuel_consumed_kg / final.total_distance_km
    efficiency_improvement = ((first_efficiency - final_efficiency) / first_efficiency) * 100
    
    # Create summary table
    summary_data = [
        ['Iterations', f'{len(history.iterations)}'],
        ['Final Delta', f'{abs(final.convergence_delta_percent):.4f}%'],
        ['', ''],
        ['A320 Max Capacity', f'{original_max_capacity:.1f} kg'],
        ['Optimized Capacity', f'{optimized_capacity:.1f} kg'],
        ['Capacity Savings', f'{capacity_savings:.1f} kg ({capacity_savings_pct:.1f}%)'],
        ['', ''],
        ['Fuel Reduction', f'{fuel_reduction:.1f} kg ({fuel_reduction_pct:.2f}%)'],
        ['Mass Reduction', f'{mass_reduction:.1f} kg ({mass_reduction_pct:.1f}%)'],
        ['Time Change', f'{time_diff/60:.1f} min'],
        ['', ''],
        ['Efficiency Improvement', f'{efficiency_improvement:.2f}%']
    ]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('<b>Fuel Consumption Comparison</b>', '<b>Fuel Savings by Phase</b>',
                       '<b>Mission Duration Comparison</b>', '<b>Optimization Impact</b>'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "table"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Plot 1: Fuel comparison
    fig.add_trace(go.Bar(
        x=phases, y=first_fuels,
        name='Iteration 1',
        marker_color='lightblue',
        hovertemplate='<b>Iter 1 - %{x}</b><br>Fuel: %{y:.1f} kg<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=phases, y=final_fuels,
        name='Final Iteration',
        marker_color='darkblue',
        hovertemplate=f'<b>Iter {final.iteration} - %{{x}}</b><br>Fuel: %{{y:.1f}} kg<extra></extra>'
    ), row=1, col=1)
    
    # Plot 2: Improvement percentages
    fig.add_trace(go.Bar(
        x=phases, y=improvements,
        name='Fuel Savings',
        marker_color='green',
        hovertemplate='<b>%{x}</b><br>Savings: %{y:.2f}%<extra></extra>',
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=phases, y=[0]*len(phases),
        mode='lines',
        line=dict(color='black', width=1),
        showlegend=False,
        hoverinfo='skip'
    ), row=1, col=2)
    
    # Plot 3: Time comparison
    fig.add_trace(go.Bar(
        x=phases, y=first_times,
        name='Iteration 1',
        marker_color='lightcoral',
        hovertemplate='<b>Iter 1 - %{x}</b><br>Time: %{y:.2f} h<extra></extra>',
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Bar(
        x=phases, y=final_times,
        name='Final Iteration',
        marker_color='darkred',
        hovertemplate=f'<b>Iter {final.iteration} - %{{x}}</b><br>Time: %{{y:.2f}} h<extra></extra>',
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
    fig.update_xaxes(title_text="Phase", row=1, col=1)
    fig.update_yaxes(title_text="Fuel (kg)", row=1, col=1)
    
    fig.update_xaxes(title_text="Phase", row=1, col=2)
    fig.update_yaxes(title_text="Improvement (%)", row=1, col=2)
    
    fig.update_xaxes(title_text="Phase", row=2, col=1)
    fig.update_yaxes(title_text="Time (hours)", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>Optimization Performance Comparison</b><br><sup>Iteration 1 vs Final Iteration</sup>",
            x=0.5, xanchor='center',
            font=dict(size=18)
        ),
        height=900, width=1400,
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation='v', yanchor='top', y=0.98, xanchor='right', x=0.48),
        barmode='group'
    )
    
    if save_plots:
        run_dir = get_or_create_run_directory(phase="Optimized")
        html_path = os.path.join(run_dir, 'optimization_comparison.html')
        fig.write_html(html_path)
        print(f"[EXPORT] Optimization comparison saved to: {html_path}")
    
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
    Master function to create all convergence analysis visualizations.
    
    Args:
        history: Convergence history containing all iterations
        save_plots: Whether to save plots to disk
    """
    
    print("\n[PLOTTING] Creating convergence analysis visualizations...")
    
    # Create consolidated plots (fewer windows, more content per window)
    plot_convergence_trajectory(history, save_plots)
    plot_kpp_evolution(history, save_plots)
    plot_optimization_comparison(history, save_plots)
    
    # Consolidated advanced plots (multiple plots per window)
    plot_aerodynamic_performance_analysis(history, save_plots)  # Combines L/D + Thrust Lever
    plot_3d_trajectory_comparison(history, save_plots)          # Keep 3D trajectory separate
    plot_specific_energy_evolution(history, save_plots)        # Keep energy analysis separate
    
    print("[PLOTTING] Convergence analysis complete!")

