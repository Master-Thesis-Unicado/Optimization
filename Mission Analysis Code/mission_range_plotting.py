# =========  1 - MODULE INITIALIZATION =================
# ========= IMPORTS AND BASIC SETUP ===========================================
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Import visualization configuration
from visualization_config import (
    Colors,
    get_or_create_run_directory,
    get_standard_layout,
    get_standard_legend,
    get_axis_config,
    get_table_header_style,
    get_table_cell_style
)

# Set Plotly to open in browser
pio.renderers.default = "browser"


# =========  2 - VISUALIZATION CORE SYSTEM =================
class RangeOptimizationVisualization:
    """
    Range optimization visualization framework for convergence analysis.
    
    This class implements a complete visualization system for mission range optimization
    through four integrated subsystems: convergence history plotting, cruise adjustment
    visualization, distance breakdown analysis, and comprehensive dashboard generation.
    
    System Components:
    - ConvergencePlotter: Generates interactive plots showing optimization convergence
      with error evolution, tolerance bands, and convergence point marking
    - AdjustmentPlotter: Visualizes cruise distance adjustment strategy and damping effects
      through evolution plots and magnitude analysis
    - BreakdownPlotter: Displays phase-wise distance contributions with absolute and
      relative views across optimization iterations
    - DashboardGenerator: Creates comprehensive multi-plot dashboards with integrated
      phase visualizations and summary statistics
    
    Visualization Features:
    - Interactive Plotly plots with hover tooltips and zoom capabilities
    - Standardized styling from visualization_config.py for consistency
    - Convergence criteria visualization with tolerance bands
    - Phase contribution analysis with stacked bar charts
    
    Implementation:
        fig = RangeOptimizationVisualization.ConvergencePlotter.plot_convergence_history(...)
        figures = RangeOptimizationVisualization.DashboardGenerator.create_optimization_dashboard(...)
    """
    
    # ========= COLOR CONFIGURATION =========
    class ColorPalette:
        """Color palette for optimization visualizations."""
        PRIMARY = Colors.CLIMB
        SECONDARY = Colors.CRUISE
        ACCENT_GREEN = Colors.CRUISE
        ACCENT_RED = Colors.DESCENT
        GRAY = 'gray'
        SUCCESS = 'rgba(76, 175, 80, 0.15)'
    
    # ========= CONVERGENCE VISUALIZATION SYSTEM =========
    class ConvergencePlotter:
        """Convergence history visualization and error evolution plotting."""
        
        @staticmethod
        def plot_convergence_history(
            iteration_history: List,
            target_range_km: float,
            tolerance_km: float,
            save_path: Optional[str] = None
        ) -> go.Figure:
            """
            Visualize convergence history showing distance error evolution.
            
            Creates an interactive plot demonstrating how the optimization error
            decreases over iterations, with clear indication of convergence criteria
            and tolerance bounds.
            
            Args:
                iteration_history: List of OptimizationIteration objects
                target_range_km: Target mission range [km]
                tolerance_km: Convergence tolerance [km]
                save_path: Optional path to save the figure
                
            Returns:
                Plotly figure object
            """
            if not iteration_history:
                print("[WARNING] No iteration history available for plotting")
                return None
            
            iterations = [record.iteration for record in iteration_history]
            errors = [record.distance_error_km for record in iteration_history]
            total_distances = [record.total_distance_km for record in iteration_history]
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    '<b>Mission Range Convergence to Target</b>',
                    '<b>Distance Error Evolution</b>'
                ),
                vertical_spacing=0.15,
                row_heights=[0.5, 0.5]
            )
            
            # Plot 1: Total Distance vs Target
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=total_distances,
                    mode='lines+markers',
                    name='Actual Mission Range',
                    line=dict(color=RangeOptimizationVisualization.ColorPalette.PRIMARY, width=3),
                    marker=dict(size=8, symbol='circle'),
                    hovertemplate='<b>Iteration %{x}</b><br>' +
                                 'Total Distance: %{y:.2f} km<br>' +
                                 '<extra></extra>'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[iterations[0], iterations[-1]],
                    y=[target_range_km, target_range_km],
                    mode='lines',
                    name='Target Range',
                    line=dict(color=RangeOptimizationVisualization.ColorPalette.ACCENT_GREEN, width=2, dash='dash'),
                    hovertemplate='<b>Target: %{y:.2f} km</b><extra></extra>'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=iterations + iterations[::-1],
                    y=[target_range_km + tolerance_km] * len(iterations) + 
                      [target_range_km - tolerance_km] * len(iterations),
                    fill='toself',
                    fillcolor=RangeOptimizationVisualization.ColorPalette.SUCCESS,
                    line=dict(width=0),
                    name='Tolerance Band (±' + f'{tolerance_km:.0f} km)',
                    showlegend=True,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            # Plot 2: Error Evolution
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=errors,
                    mode='lines+markers',
                    name='Distance Error',
                    line=dict(color=RangeOptimizationVisualization.ColorPalette.ACCENT_RED, width=3),
                    marker=dict(size=8, symbol='diamond'),
                    hovertemplate='<b>Iteration %{x}</b><br>' +
                                 'Error: %{y:+.2f} km<br>' +
                                 '<extra></extra>'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[iterations[0], iterations[-1]],
                    y=[0, 0],
                    mode='lines',
                    name='Zero Error',
                    line=dict(color='gray', width=1, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=2, col=1
            )
            
            fig.add_hline(
                y=tolerance_km, 
                line_dash="dash", 
                line_color=RangeOptimizationVisualization.ColorPalette.ACCENT_GREEN,
                annotation_text=f"+{tolerance_km} km",
                annotation_position="right",
                row=2, col=1
            )
            fig.add_hline(
                y=-tolerance_km, 
                line_dash="dash", 
                line_color=RangeOptimizationVisualization.ColorPalette.ACCENT_GREEN,
                annotation_text=f"-{tolerance_km} km",
                annotation_position="right",
                row=2, col=1
            )
            
            # Mark convergence point if achieved
            converged_idx = None
            for i, record in enumerate(iteration_history):
                if record.converged:
                    converged_idx = i
                    break
            
            if converged_idx is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[iterations[converged_idx]],
                        y=[total_distances[converged_idx]],
                        mode='markers',
                        name='Convergence Point',
                        marker=dict(
                            size=15,
                            color=RangeOptimizationVisualization.ColorPalette.ACCENT_GREEN,
                            symbol='star',
                            line=dict(width=2, color='white')
                        ),
                        hovertemplate='<b>Converged!</b><br>' +
                                     'Iteration: %{x}<br>' +
                                     'Distance: %{y:.2f} km<br>' +
                                     '<extra></extra>',
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            fig.update_xaxes(get_axis_config("Iteration"), row=1, col=1)
            fig.update_xaxes(get_axis_config("Iteration"), row=2, col=1)
            fig.update_yaxes(get_axis_config("Mission Range [km]"), row=1, col=1)
            fig.update_yaxes(get_axis_config("Distance Error [km]"), row=2, col=1)
            
            layout_config = get_standard_layout(
                title="Mission Range Optimization: Convergence Analysis",
                subtitle=f"Target: {target_range_km:.0f} km | Tolerance: ±{tolerance_km:.0f} km",
                height=900
            )
            layout_config.update({
                'showlegend': True,
                'hovermode': 'x unified',
                'legend': get_standard_legend()
            })
            fig.update_layout(**layout_config)
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(save_path)
                print(f"[EXPORT] Convergence history saved to: {save_path}")
            
            return fig
    
    # ========= ADJUSTMENT STRATEGY VISUALIZATION =========
    class AdjustmentPlotter:
        """Cruise distance adjustment strategy visualization."""
        
        @staticmethod
        def plot_cruise_adjustment_strategy(
            iteration_history: List,
            save_path: Optional[str] = None
        ) -> go.Figure:
            """
            Visualize cruise distance adjustment strategy throughout optimization.
            
            Demonstrates how the optimizer adjusts cruise distance in response to
            distance errors, showing the damping effect and convergence behavior.
            
            Args:
                iteration_history: List of OptimizationIteration objects
                save_path: Optional path to save the figure
                
            Returns:
                Plotly figure object
            """
            if not iteration_history:
                print("[WARNING] No iteration history available for plotting")
                return None
            
            iterations = [record.iteration for record in iteration_history]
            cruise_distances = [record.cruise_distance_km for record in iteration_history]
            
            adjustments = [0.0]
            for i in range(1, len(cruise_distances)):
                adj = cruise_distances[i] - cruise_distances[i-1]
                adjustments.append(adj)
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    '<b>Cruise Distance Evolution</b>',
                    '<b>Adjustment Magnitude per Iteration</b>'
                ),
                vertical_spacing=0.15
            )
            
            # Plot 1: Cruise Distance Evolution
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=cruise_distances,
                    mode='lines+markers',
                    name='Cruise Distance',
                    line=dict(color=RangeOptimizationVisualization.ColorPalette.SECONDARY, width=3),
                    marker=dict(size=10, symbol='square'),
                    hovertemplate='<b>Iteration %{x}</b><br>' +
                                 'Cruise Distance: %{y:.2f} km<br>' +
                                 '<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Plot 2: Adjustment Magnitude (Bar Chart)
            colors = [RangeOptimizationVisualization.ColorPalette.ACCENT_GREEN if adj >= 0 
                     else RangeOptimizationVisualization.ColorPalette.ACCENT_RED 
                     for adj in adjustments]
            
            fig.add_trace(
                go.Bar(
                    x=iterations,
                    y=adjustments,
                    name='Adjustment',
                    marker=dict(
                        color=colors,
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate='<b>Iteration %{x}</b><br>' +
                                 'Adjustment: %{y:+.2f} km<br>' +
                                 '<extra></extra>'
                ),
                row=2, col=1
            )
            
            fig.add_hline(
                y=0, 
                line_dash="solid", 
                line_color="gray",
                line_width=1,
                row=2, col=1
            )
            
            fig.update_xaxes(get_axis_config("Iteration"), row=1, col=1)
            fig.update_xaxes(get_axis_config("Iteration"), row=2, col=1)
            fig.update_yaxes(get_axis_config("Cruise Distance [km]"), row=1, col=1)
            fig.update_yaxes(get_axis_config("Adjustment [km]"), row=2, col=1)
            
            layout_config = get_standard_layout(
                title="Cruise Distance Adjustment Strategy",
                subtitle="Iterative Optimization with Damping",
                height=800
            )
            layout_config.update({
                'showlegend': True,
                'hovermode': 'x unified',
                'legend': get_standard_legend()
            })
            fig.update_layout(**layout_config)
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(save_path)
                print(f"[EXPORT] Cruise adjustment plot saved to: {save_path}")
            
            return fig
    
    # ========= DISTANCE BREAKDOWN VISUALIZATION =========
    class BreakdownPlotter:
        """Phase-wise distance breakdown visualization."""
        
        @staticmethod
        def plot_distance_breakdown_evolution(
            iteration_data: List[Dict[str, Any]],
            save_path: Optional[str] = None
        ) -> go.Figure:
            """
            Visualize phase-wise distance contributions throughout optimization.
            
            Shows how climb, cruise, and descent distances contribute to total
            mission range, and how these contributions evolve during optimization.
            
            Args:
                iteration_data: List of dictionaries containing phase breakdowns
                    Each dict should have: iteration, climb_km, cruise_km, descent_km
                save_path: Optional path to save the figure
                
            Returns:
                Plotly figure object
            """
            if not iteration_data:
                print("[WARNING] No iteration data available for plotting")
                return None
            
            iterations = [data['iteration'] for data in iteration_data]
            climb_distances = [data['climb_km'] for data in iteration_data]
            cruise_distances = [data['cruise_km'] for data in iteration_data]
            descent_distances = [data['descent_km'] for data in iteration_data]
            total_distances = [data['total_km'] for data in iteration_data]
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    '<b>Distance Breakdown (Absolute)</b>',
                    '<b>Distance Breakdown (Percentage)</b>'
                ),
                specs=[[{'type': 'bar'}, {'type': 'bar'}]]
            )
            
            # Plot 1: Absolute distances (stacked bar)
            fig.add_trace(
                go.Bar(
                    x=iterations,
                    y=climb_distances,
                    name='Climb',
                    marker=dict(color=RangeOptimizationVisualization.ColorPalette.PRIMARY),
                    hovertemplate='Climb: %{y:.2f} km<extra></extra>'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=iterations,
                    y=cruise_distances,
                    name='Cruise',
                    marker=dict(color=RangeOptimizationVisualization.ColorPalette.SECONDARY),
                    hovertemplate='Cruise: %{y:.2f} km<extra></extra>'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=iterations,
                    y=descent_distances,
                    name='Descent',
                    marker=dict(color=RangeOptimizationVisualization.ColorPalette.ACCENT_RED),
                    hovertemplate='Descent: %{y:.2f} km<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Plot 2: Percentage breakdown
            climb_pct = [c/t*100 for c, t in zip(climb_distances, total_distances)]
            cruise_pct = [c/t*100 for c, t in zip(cruise_distances, total_distances)]
            descent_pct = [d/t*100 for d, t in zip(descent_distances, total_distances)]
            
            fig.add_trace(
                go.Bar(
                    x=iterations,
                    y=climb_pct,
                    name='Climb %',
                    marker=dict(color=RangeOptimizationVisualization.ColorPalette.PRIMARY),
                    hovertemplate='Climb: %{y:.1f}%<extra></extra>',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(
                    x=iterations,
                    y=cruise_pct,
                    name='Cruise %',
                    marker=dict(color=RangeOptimizationVisualization.ColorPalette.SECONDARY),
                    hovertemplate='Cruise: %{y:.1f}%<extra></extra>',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(
                    x=iterations,
                    y=descent_pct,
                    name='Descent %',
                    marker=dict(color=RangeOptimizationVisualization.ColorPalette.ACCENT_RED),
                    hovertemplate='Descent: %{y:.1f}%<extra></extra>',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            fig.update_xaxes(get_axis_config("Iteration"), row=1, col=1)
            fig.update_xaxes(get_axis_config("Iteration"), row=1, col=2)
            fig.update_yaxes(get_axis_config("Distance [km]"), row=1, col=1)
            fig.update_yaxes(get_axis_config("Percentage [%]"), row=1, col=2)
            
            layout_config = get_standard_layout(
                title="Mission Distance Phase Breakdown Evolution",
                subtitle="Climb | Cruise | Descent Contributions",
                height=500
            )
            layout_config.update({
                'barmode': 'stack',
                'legend': get_standard_legend()
            })
            fig.update_layout(**layout_config)
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(save_path)
                print(f"[EXPORT] Distance breakdown plot saved to: {save_path}")
            
            return fig
    
    # ========= DASHBOARD GENERATION SYSTEM =========
    class DashboardGenerator:
        """Comprehensive dashboard creation and phase plot integration."""
        
        @staticmethod
        def save_converged_mission_plots(
            climb_result,
            cruise_result,
            descent_result,
            climb_info,
            descent_info,
            aero,
            engine,
            initial_mass_kg: float,
            base_output_dir: Path
        ):
            """
            Save all detailed phase plots from the converged mission.
            
            This function generates and saves comprehensive visualizations for climb,
            cruise, and descent phases after optimization convergence.
            
            Args:
                climb_result: Climb optimization result (MinFuelSchedule)
                cruise_result: Cruise simulation result (CruiseResults)
                descent_result: Descent optimization result (DescentResults)
                climb_info: Climb DP optimization info
                descent_info: Descent DP optimization info
                aero: Aerodynamics wrapper
                engine: Engine wrapper
                initial_mass_kg: Initial aircraft mass [kg]
                base_output_dir: Base directory for saving plots
            """
            print("\n" + "="*80)
            print("SAVING CONVERGED MISSION PHASE PLOTS")
            print("="*80)
            
            try:
                from climb_plotting import plot_climb_performance_detailed
                from cruise_plotting import plot_cruise_performance_detailed
                from descent_plotting import plot_descent_trajectory_interactive
                from mission_summary import plot_mission_summary_dashboard, plot_combined_performance_analysis
            except ImportError as e:
                print(f"[WARNING] Could not import phase plotting functions: {e}")
                return
            
            climb_dir = base_output_dir / "Climb"
            cruise_dir = base_output_dir / "Cruise"
            descent_dir = base_output_dir / "Descent"
            mission_dir = base_output_dir / "Mission_Summary"
            
            for directory in [climb_dir, cruise_dir, descent_dir, mission_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            
            print(f"\n[CLIMB] Generating and saving climb phase plots...")
            try:
                plot_climb_performance_detailed(climb_result, climb_info)
                print(f"[CLIMB] Climb plots saved to: {climb_dir}")
            except Exception as e:
                print(f"[ERROR] Failed to save climb plots: {e}")
            
            print(f"\n[CRUISE] Generating and saving cruise phase plots...")
            try:
                plot_cruise_performance_detailed(cruise_result)
                print(f"[CRUISE] Cruise plots saved to: {cruise_dir}")
            except Exception as e:
                print(f"[ERROR] Failed to save cruise plots: {e}")
            
            print(f"\n[DESCENT] Generating and saving descent phase plots...")
            try:
                plot_descent_trajectory_interactive(descent_result)
                print(f"[DESCENT] Descent plots saved to: {descent_dir}")
            except Exception as e:
                print(f"[ERROR] Failed to save descent plots: {e}")
            
            print(f"\n[MISSION] Generating and saving complete mission summary...")
            try:
                plot_mission_summary_dashboard(
                    climb_result=climb_result,
                    cruise_result=cruise_result,
                    descent_result=descent_result,
                    initial_mass_kg=initial_mass_kg
                )
                
                plot_combined_performance_analysis(
                    climb_result=climb_result,
                    cruise_result=cruise_result,
                    descent_result=descent_result,
                    initial_mass_kg=initial_mass_kg
                )
                print(f"[MISSION] Mission summary saved to: {mission_dir}")
            except Exception as e:
                print(f"[ERROR] Failed to save mission summary: {e}")
            
            print("\n" + "="*80)
            print("ALL CONVERGED MISSION PLOTS SAVED SUCCESSFULLY")
            print("="*80)
        
        @staticmethod
        def create_optimization_dashboard(
            iteration_history: List,
            iteration_data: List[Dict[str, Any]],
            optimization_summary: Dict[str, Any],
            save_dir: Optional[str] = None,
            climb_result = None,
            cruise_result = None,
            descent_result = None,
            climb_info = None,
            descent_info = None,
            aero = None,
            engine = None,
            initial_mass_kg: float = None
        ) -> Dict[str, go.Figure]:
            """
            Create comprehensive optimization analysis dashboard.
            
            Generates complete set of visualizations for range optimization analysis,
            including convergence history, adjustment strategy, phase breakdowns, and
            detailed phase plots from the converged mission.
            
            Args:
                iteration_history: List of OptimizationIteration objects
                iteration_data: List of phase breakdown dictionaries
                optimization_summary: Summary statistics dictionary
                save_dir: Optional directory to save all figures
                climb_result: Optional climb results for detailed plotting
                cruise_result: Optional cruise results for detailed plotting
                descent_result: Optional descent results for detailed plotting
                climb_info: Optional climb DP info for detailed plotting
                descent_info: Optional descent DP info for detailed plotting
                aero: Optional aerodynamics wrapper for detailed plotting
                engine: Optional engine wrapper for detailed plotting
                initial_mass_kg: Optional initial mass for detailed plotting
                
            Returns:
                Dictionary of figure objects keyed by plot name
            """
            figures = {}
            
            if save_dir:
                output_path = Path(save_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                print(f"\n[DASHBOARD] Creating optimization dashboard in: {output_path}")
            else:
                output_path = Path(get_or_create_run_directory())
                output_path.mkdir(parents=True, exist_ok=True)
            
            range_opt_path = output_path / "Range_Optimization"
            range_opt_path.mkdir(parents=True, exist_ok=True)
            
            print("[DASHBOARD] Generating convergence history plot...")
            fig_convergence = RangeOptimizationVisualization.ConvergencePlotter.plot_convergence_history(
                iteration_history,
                optimization_summary['target_range_km'],
                optimization_summary['tolerance_km'],
                save_path=str(range_opt_path / "convergence_history.html")
            )
            figures['convergence'] = fig_convergence
            
            print("[DASHBOARD] Generating cruise adjustment strategy plot...")
            fig_adjustment = RangeOptimizationVisualization.AdjustmentPlotter.plot_cruise_adjustment_strategy(
                iteration_history,
                save_path=str(range_opt_path / "cruise_adjustment.html")
            )
            figures['adjustment'] = fig_adjustment
            
            if iteration_data:
                print("[DASHBOARD] Generating distance breakdown plot...")
                fig_breakdown = RangeOptimizationVisualization.BreakdownPlotter.plot_distance_breakdown_evolution(
                    iteration_data,
                    save_path=str(range_opt_path / "distance_breakdown.html")
                )
                figures['breakdown'] = fig_breakdown
            
            print("[DASHBOARD] Generating optimization summary table...")
            fig_summary = RangeOptimizationVisualization.SummaryGenerator.create_optimization_summary_table(
                optimization_summary,
                save_path=str(range_opt_path / "optimization_summary.html")
            )
            figures['summary'] = fig_summary
            
            print(f"[DASHBOARD] Range optimization plots saved to: {range_opt_path}")
            
            if (climb_result is not None and cruise_result is not None and 
                descent_result is not None and optimization_summary.get('converged', False)):
                print("\n[DASHBOARD] Generating detailed phase plots from converged mission...")
                RangeOptimizationVisualization.DashboardGenerator.save_converged_mission_plots(
                    climb_result=climb_result,
                    cruise_result=cruise_result,
                    descent_result=descent_result,
                    climb_info=climb_info,
                    descent_info=descent_info,
                    aero=aero,
                    engine=engine,
                    initial_mass_kg=initial_mass_kg,
                    base_output_dir=output_path
                )
            else:
                if not optimization_summary.get('converged', False):
                    print("\n[DASHBOARD] Skipping detailed phase plots (optimization did not converge)")
                else:
                    print("\n[DASHBOARD] Skipping detailed phase plots (phase results not provided)")
            
            print(f"\n[DASHBOARD] Complete dashboard saved to: {output_path}")
            
            return figures
    
    # ========= SUMMARY TABLE GENERATION =========
    class SummaryGenerator:
        """Summary table creation and formatting."""
        
        @staticmethod
        def create_optimization_summary_table(
            summary: Dict[str, Any],
            save_path: Optional[str] = None
        ) -> go.Figure:
            """
            Create formatted summary table of optimization results.
            
            Args:
                summary: Optimization summary dictionary
                save_path: Optional path to save the figure
                
            Returns:
                Plotly figure object with formatted table
            """
            labels = [
                '<b>Target Range</b>',
                '<b>Tolerance</b>',
                '<b>Damping Factor</b>',
                '<b>Total Iterations</b>',
                '<b>Converged</b>',
                '<b>Final Cruise Distance</b>',
                '<b>Final Total Distance</b>',
                '<b>Final Error</b>',
                '<b>Convergence Rate</b>'
            ]
            
            if len(summary.get('iteration_history', [])) > 1:
                first_error = abs(summary['iteration_history'][0].distance_error_km)
                last_error = abs(summary['iteration_history'][-1].distance_error_km)
                if first_error > 0:
                    convergence_rate = (1 - last_error/first_error) * 100
                else:
                    convergence_rate = 100.0
            else:
                convergence_rate = 0.0
            
            values = [
                f"{summary['target_range_km']:.1f} km",
                f"±{summary['tolerance_km']:.1f} km",
                f"{summary['damping_factor']:.2f}",
                f"{summary['total_iterations']}",
                "Yes" if summary['converged'] else "No",
                f"{summary['final_cruise_distance_km']:.2f} km",
                f"{summary['final_total_distance_km']:.2f} km",
                f"{summary['final_error_km']:+.2f} km",
                f"{convergence_rate:.1f}%"
            ]
            
            header_style = get_table_header_style()
            cell_style = get_table_cell_style()
            
            header_style['values'] = ['<b>Parameter</b>', '<b>Value</b>']
            header_style['align'] = 'left'
            
            cell_style['values'] = [labels, values]
            cell_style['fill_color'] = [['white']*len(labels)]
            cell_style['align'] = 'left'
            cell_style['height'] = 30
            
            fig = go.Figure(data=[go.Table(
                header=header_style,
                cells=cell_style
            )])
            
            layout_config = get_standard_layout(
                title="Range Optimization Summary",
                subtitle="Convergence Statistics and Results",
                height=450
            )
            layout_config['margin'] = dict(l=20, r=20, t=80, b=20)
            fig.update_layout(**layout_config)
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(save_path)
                print(f"[EXPORT] Summary table saved to: {save_path}")
            
            return fig
    
    # ========= UTILITY FUNCTIONS =========
    class Utilities:
        """Utility functions for reporting and analysis."""
        
        @staticmethod
        def print_optimization_report(optimization_summary: Dict[str, Any]):
            """
            Print formatted text report of optimization results.
            
            Args:
                optimization_summary: Optimization summary dictionary
            """
            print("\n" + "="*80)
            print("RANGE OPTIMIZATION REPORT")
            print("="*80)
            print(f"Target Range:           {optimization_summary['target_range_km']:.1f} km")
            print(f"Tolerance:              +/-{optimization_summary['tolerance_km']:.1f} km")
            print(f"Damping Factor:         {optimization_summary['damping_factor']:.3f}")
            print(f"Total Iterations:       {optimization_summary['total_iterations']}")
            print(f"Convergence Status:     {'CONVERGED [YES]' if optimization_summary['converged'] else 'NOT CONVERGED [NO]'}")
            print(f"\nFinal Results:")
            print(f"  Cruise Distance:      {optimization_summary['final_cruise_distance_km']:.2f} km")
            print(f"  Total Mission Range:  {optimization_summary['final_total_distance_km']:.2f} km")
            print(f"  Distance Error:       {optimization_summary['final_error_km']:+.2f} km")
            print(f"  Relative Error:       {abs(optimization_summary['final_error_km']/optimization_summary['target_range_km'])*100:.2f}%")
            
            if optimization_summary['iteration_history']:
                print(f"\nIteration History:")
                print(f"{'Iter':>4} | {'Cruise [km]':>12} | {'Total [km]':>11} | {'Error [km]':>11} | Status")
                print("-" * 60)
                for record in optimization_summary['iteration_history']:
                    status = "[OK]" if record.converged else "[->]"
                    print(f"{record.iteration:4d} | {record.cruise_distance_km:12.2f} | "
                          f"{record.total_distance_km:11.2f} | {record.distance_error_km:+11.2f} | {status}")
            
            print("="*80)


# =========  3 - BACKWARD COMPATIBILITY WRAPPERS =================
def plot_convergence_history(
    iteration_history: List,
    target_range_km: float,
    tolerance_km: float,
    save_path: Optional[str] = None
) -> go.Figure:
    """Backward compatibility wrapper for RangeOptimizationVisualization.ConvergencePlotter.plot_convergence_history"""
    return RangeOptimizationVisualization.ConvergencePlotter.plot_convergence_history(
        iteration_history, target_range_km, tolerance_km, save_path
    )

def plot_cruise_adjustment_strategy(
    iteration_history: List,
    save_path: Optional[str] = None
) -> go.Figure:
    """Backward compatibility wrapper for RangeOptimizationVisualization.AdjustmentPlotter.plot_cruise_adjustment_strategy"""
    return RangeOptimizationVisualization.AdjustmentPlotter.plot_cruise_adjustment_strategy(
        iteration_history, save_path
    )

def plot_distance_breakdown_evolution(
    iteration_data: List[Dict[str, Any]],
    save_path: Optional[str] = None
) -> go.Figure:
    """Backward compatibility wrapper for RangeOptimizationVisualization.BreakdownPlotter.plot_distance_breakdown_evolution"""
    return RangeOptimizationVisualization.BreakdownPlotter.plot_distance_breakdown_evolution(
        iteration_data, save_path
    )

def create_optimization_dashboard(
    iteration_history: List,
    iteration_data: List[Dict[str, Any]],
    optimization_summary: Dict[str, Any],
    save_dir: Optional[str] = None,
    climb_result = None,
    cruise_result = None,
    descent_result = None,
    climb_info = None,
    descent_info = None,
    aero = None,
    engine = None,
    initial_mass_kg: float = None
) -> Dict[str, go.Figure]:
    """Backward compatibility wrapper for RangeOptimizationVisualization.DashboardGenerator.create_optimization_dashboard"""
    return RangeOptimizationVisualization.DashboardGenerator.create_optimization_dashboard(
        iteration_history, iteration_data, optimization_summary, save_dir,
        climb_result, cruise_result, descent_result,
        climb_info, descent_info, aero, engine, initial_mass_kg
    )

def create_optimization_summary_table(
    summary: Dict[str, Any],
    save_path: Optional[str] = None
) -> go.Figure:
    """Backward compatibility wrapper for RangeOptimizationVisualization.SummaryGenerator.create_optimization_summary_table"""
    return RangeOptimizationVisualization.SummaryGenerator.create_optimization_summary_table(
        summary, save_path
    )

def print_optimization_report(optimization_summary: Dict[str, Any]):
    """Backward compatibility wrapper for RangeOptimizationVisualization.Utilities.print_optimization_report"""
    return RangeOptimizationVisualization.Utilities.print_optimization_report(optimization_summary)

def save_converged_mission_plots(
    climb_result,
    cruise_result,
    descent_result,
    climb_info,
    descent_info,
    aero,
    engine,
    initial_mass_kg: float,
    base_output_dir: Path
):
    """Backward compatibility wrapper for RangeOptimizationVisualization.DashboardGenerator.save_converged_mission_plots"""
    return RangeOptimizationVisualization.DashboardGenerator.save_converged_mission_plots(
        climb_result, cruise_result, descent_result,
        climb_info, descent_info, aero, engine,
        initial_mass_kg, base_output_dir
    )
