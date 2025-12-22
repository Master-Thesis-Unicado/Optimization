# ========================================================================
# MISSION FUEL OPTIMIZATION VISUALIZATION MODULE
# ========================================================================
"""
Convergence analysis visualization for bisection-based fuel optimization.

Mathematical context:
    Bisection iteration: [m_fuel,low, m_fuel,high] → m_fuel,test = (low + high)/2
    Deficit: Δm = m_fuel,initial - m_fuel,consumed
    Convergence: |m_fuel,high - m_fuel,low| < ε_tol
    Optimized capacity: m_fuel,opt = m_fuel,consumed × (1 + β_safety)

Visualization domains:
    1. Bisection bounds evolution: [m_low, m_high, m_test] vs. iteration
    2. Deficit convergence: |Δm| vs. iteration (log scale)
    3. Performance change: Percentage deviation from first iteration
    4. Capacity comparison: Original vs. optimized vs. savings
    5. Phase breakdown: m_fuel and t by phase (climb, cruise, descent)

All plots use interactive Plotly with HTML export capability.
"""

from __future__ import annotations
import numpy as np
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
from pathlib import Path

# Suppress non-critical browser communication warnings
warnings.filterwarnings('ignore', category=UserWarning, module='choreographer')
try:
    import logging
    logging.getLogger('choreographer').setLevel(logging.ERROR)
except:
    pass

pio.renderers.default = "browser"

# Fuel optimization data structures and parameters
from mission_fuel_optimizer import ConvergenceHistory, SAFETY_BUFFER_PERCENT

# Aircraft reference capacity
from aircraft_config import W_FUEL_KG

# Visualization styling
from visualization_config import (
    Colors,
    get_or_create_run_directory
)


# ========================================================================
# SECTION 1: CONVERGENCE TRAJECTORY VISUALIZATION
# ========================================================================

def plot_convergence_trajectory(history: ConvergenceHistory, save_plots: bool = True):
    """
    Generate 4-panel bisection convergence analysis.
    
    Panels:
        1. Bisection bounds evolution: [m_low, m_high, m_test, m_consumed] vs. k
        2. Deficit convergence: |Δm| vs. k (log scale)
        3. Performance change: (m_k - m_1)/m_1 × 100% vs. k
        4. Summary table: Iterations, deficit, range, capacity, savings
    
    Note: Convergence criterion is |m_high - m_low| < ε_tol, not |Δm| < ε_tol.
          Deficit is diagnostic; bounds range determines termination.
    
    Parameters:
        history: ConvergenceHistory - bisection iteration sequence
        save_plots: bool - export to HTML
    """
    
    if len(history.iterations) < 2:
        print("[WARNING] Need at least 2 iterations to plot convergence")
        return
    
    # ────────────────────────────────────────────────────────────────────
    # Data Extraction from Convergence History
    # ────────────────────────────────────────────────────────────────────
    iterations = [r.iteration for r in history.iterations]                # k
    initial_fuels = [r.initial_fuel_kg for r in history.iterations]       # m_test,k [kg]
    consumed_fuels = [r.fuel_consumed_kg for r in history.iterations]     # m_consumed,k [kg]
    fuel_deficits = [r.fuel_deficit_kg for r in history.iterations]       # Δm_k = m_test - m_consumed [kg]
    
    # Bisection bounds: [m_low,k, m_high,k]
    fuel_lows = [bounds[0] for bounds in history.fuel_bounds_history]     # m_low,k [kg]
    fuel_highs = [bounds[1] for bounds in history.fuel_bounds_history]    # m_high,k [kg]
    bound_ranges = [high - low for low, high in history.fuel_bounds_history]  # m_high - m_low [kg]
    
    # ────────────────────────────────────────────────────────────────────
    # Convergence Metrics
    # ────────────────────────────────────────────────────────────────────
    first_iter = history.iterations[0]
    last_iter = history.iterations[-1]
    optimized_fuel = last_iter.fuel_consumed_kg * (1.0 + SAFETY_BUFFER_PERCENT)  # m_opt = m_consumed × (1 + β)
    convergence_iterations = len(history.iterations)                              # N_iter
    final_deficit = abs(last_iter.fuel_deficit_kg)                                # |Δm_final| [kg]
    final_range = bound_ranges[-1] if bound_ranges else 0                         # m_high,N - m_low,N [kg]
    
    # Relative change from first iteration: (m_k - m_1)/m_1 × 100%
    first_fuel = consumed_fuels[0]
    fuel_percent_change = [(f - first_fuel) / first_fuel * 100 for f in consumed_fuels]
    
    # Capacity savings: Δm_capacity = m_ref - m_opt
    fuel_reduction = W_FUEL_KG - optimized_fuel                  # Δm_capacity [kg]
    fuel_reduction_pct = (fuel_reduction / W_FUEL_KG) * 100      # Δm_capacity / m_ref × 100%
    
    # Create summary table data
    summary_data = [
        ['Iterations', f'{convergence_iterations}'],
        ['Final Deficit', f'{final_deficit:.2f} kg'],
        ['Final Range', f'{final_range:.2f} kg'],
        ['', ''],
        ['Initial Capacity', f'{W_FUEL_KG:.1f} kg'],
        ['Optimized Capacity', f'{optimized_fuel:.1f} kg'],
        ['Fuel Savings', f'{fuel_reduction:.1f} kg ({fuel_reduction_pct:.1f}%)'],
        ['', ''],
        ['Final Time', f'{last_iter.total_time_s/60:.0f} min'],
        ['Final Mass', f'{last_iter.final_mass_kg:.0f} kg']
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
        fuel_opt_path = Path(run_dir) / "Fuel_Optimization"
        fuel_opt_path.mkdir(parents=True, exist_ok=True)
        html_path = str(fuel_opt_path / 'fuel_convergence.html')
        fig.write_html(html_path)
        print(f"[EXPORT] Convergence analysis saved to: {html_path}")
    
    fig.show()


# ========================================================================
# SECTION 2: OPTIMIZATION RESULTS COMPARISON
# ========================================================================

def plot_optimization_comparison(history: ConvergenceHistory, save_plots: bool = True):
    """
    Generate 4-panel optimization results summary.
    
    Panels:
        1. Capacity comparison: m_ref vs. m_opt vs. Δm_capacity (bar chart)
        2. Fuel breakdown: m_fuel by phase (climb, cruise, descent, total)
        3. Duration breakdown: t by phase (hours)
        4. Summary table: Iterations, deficit, capacity, savings, duration, final mass
    
    Optimized capacity: m_opt = m_consumed,final × (1 + β_safety)
    Savings: Δm_capacity = m_ref - m_opt
    
    Parameters:
        history: ConvergenceHistory - final converged iteration
        save_plots: bool - export to HTML
    """
    
    if len(history.iterations) < 1:
        print("[WARNING] Need at least 1 iteration for comparison")
        return
    
    # ────────────────────────────────────────────────────────────────────
    # Final Converged Result
    # ────────────────────────────────────────────────────────────────────
    final = history.iterations[-1]
    
    # Optimized capacity: m_opt = m_consumed,final × (1 + β_safety)
    optimized_capacity = final.fuel_consumed_kg * (1 + SAFETY_BUFFER_PERCENT)      # m_opt [kg]
    capacity_reduction = W_FUEL_KG - optimized_capacity                             # Δm_capacity [kg]
    capacity_reduction_pct = (capacity_reduction / W_FUEL_KG) * 100                 # Δm/m_ref × 100%
    
    # ────────────────────────────────────────────────────────────────────
    # Phase-Wise Breakdown
    # ────────────────────────────────────────────────────────────────────
    phases = ['Climb', 'Cruise', 'Descent', 'Total']
    optimized_fuels = [final.climb_fuel_kg, final.cruise_fuel_kg, 
                      final.descent_fuel_kg, final.fuel_consumed_kg]               # m_phase [kg]
    optimized_times = [final.climb_time_s / 3600, final.cruise_time_s / 3600, 
                      final.descent_time_s / 3600, final.total_time_s / 3600]      # t_phase [h]
    
    # Relative fuel distribution: m_phase / m_total × 100%
    fuel_percentages = [(f / final.fuel_consumed_kg * 100) if final.fuel_consumed_kg > 0 else 0.0 
                       for f in optimized_fuels[:-1]]
    fuel_percentages.append(100.0)  # Total always 100%
    
    # Create summary table
    summary_data = [
        ['Iterations', f'{len(history.iterations)}'],
        ['Final Deficit', f'{abs(final.fuel_deficit_kg):.2f} kg'],
        ['Convergence Status', 'Equilibrium Achieved' if abs(final.fuel_deficit_kg) < 50 else 'Near Equilibrium'],
        ['', ''],
        ['Original Capacity', f'{W_FUEL_KG:.1f} kg'],
        ['Optimized Capacity', f'{optimized_capacity:.1f} kg'],
        ['Capacity Savings', f'{capacity_reduction:.1f} kg ({capacity_reduction_pct:.1f}%)'],
        ['', ''],
        ['Mission Duration', f'{final.total_time_s/3600:.2f} hours'],
        ['Final Mass', f'{final.final_mass_kg:.0f} kg']
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
    capacity_values = [W_FUEL_KG, optimized_capacity, capacity_reduction]
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
        fuel_opt_path = Path(run_dir) / "Fuel_Optimization"
        fuel_opt_path.mkdir(parents=True, exist_ok=True)
        html_path = str(fuel_opt_path / 'optimization_results_summary.html')
        fig.write_html(html_path)
        print(f"[EXPORT] Optimization results summary saved to: {html_path}")
    
    fig.show()


# ========================================================================
# SECTION 3: MASTER VISUALIZATION INTERFACE
# ========================================================================

def visualize_convergence_analysis(history: ConvergenceHistory, save_plots: bool = True):
    """
    Generate complete fuel optimization visualization suite.
    
    Produces:
        - fuel_convergence.html: 4-panel bisection convergence analysis
        - optimization_results_summary.html: 4-panel optimized mission comparison
    
    Parameters:
        history: ConvergenceHistory - complete iteration sequence
        save_plots: bool - export to HTML in "Optimized" subdirectory
    """
    
    print("\n[PLOTTING] Creating convergence analysis visualizations...")
    
    # Panel set 1: Bisection convergence trajectory
    plot_convergence_trajectory(history, save_plots)
    
    # Panel set 2: Optimized mission comparison
    plot_optimization_comparison(history, save_plots)
    
    print("[PLOTTING] Convergence analysis complete!")

