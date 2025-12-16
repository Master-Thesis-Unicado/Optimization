# ========================================================================
# MISSION RANGE OPTIMIZATION VISUALIZATION MODULE
# ========================================================================
"""
Convergence analysis visualization for iterative range optimization.

Visualization domains:
    1. Convergence history: e_k vs. k, s_total vs. s_target
    2. Adjustment strategy: s_cruise evolution, Δs_cruise per iteration
    3. Phase breakdown: s_climb, s_cruise, s_descent contributions
    4. Summary tables: Convergence statistics, final results
    5. Dashboard: Multi-panel comprehensive analysis

Mathematical context:
    Error evolution: e_k = s_target - s_total,k
    Convergence: |e_k| < ε_tol
    Update rule: s_cruise,k+1 = s_cruise,k + α·e_k

All plots use interactive Plotly with hover tooltips and export capability.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Visualization styling and utilities
from visualization_config import (
    Colors,
    get_or_create_run_directory,
    get_standard_layout,
    get_standard_legend,
    get_axis_config,
    get_table_header_style,
    get_table_cell_style
)

pio.renderers.default = "browser"


# ========================================================================
# SECTION 1: RANGE OPTIMIZATION VISUALIZATION FRAMEWORK
# ========================================================================

class RangeOptimizationVisualization:
    """
    Visualization suite for range optimization convergence analysis.
    
    Subsystems:
        - ConvergencePlotter: e_k evolution, s_total vs. s_target
        - AdjustmentPlotter: s_cruise,k trajectory, Δs_cruise,k
        - BreakdownPlotter: Phase contributions (s_climb, s_cruise, s_descent)
        - DashboardGenerator: Multi-panel dashboard with phase plots
        - SummaryGenerator: Tabular convergence statistics
        - Utilities: Text reporting
    
    All visualizations use Plotly interactive graphics with HTML export.
    """
    
    # ────────────────────────────────────────────────────────────────────
    # Color Scheme
    # ────────────────────────────────────────────────────────────────────
    
    class ColorPalette:
        """Color assignments for optimization plots."""
        PRIMARY = Colors.CLIMB          # Climb phase / primary line
        SECONDARY = Colors.CRUISE        # Cruise phase / secondary line
        ACCENT_GREEN = Colors.CRUISE     # Success / positive adjustment
        ACCENT_RED = Colors.DESCENT      # Error / negative adjustment
        GRAY = 'gray'                    # Reference lines
        SUCCESS = 'rgba(76, 175, 80, 0.15)'  # Tolerance band fill
    
    # ────────────────────────────────────────────────────────────────────
    # Convergence History Visualization
    # ────────────────────────────────────────────────────────────────────
    
    class ConvergencePlotter:
        """
        Convergence trajectory visualization: e_k vs. k, s_total vs. s_target.
        
        Plots:
            Panel 1: s_total,k vs. s_target with tolerance band [s_target ± ε_tol]
            Panel 2: e_k = s_target - s_total,k evolution
        """
        
        @staticmethod
        def plot_convergence_history(
            iteration_history: List,
            target_range_km: float,
            tolerance_km: float,
            save_path: Optional[str] = None
        ) -> go.Figure:
            """
            Generate 2-panel convergence analysis figure.
            
            Panel 1: s_total,k vs. k with s_target reference and tolerance band
            Panel 2: e_k vs. k with ±ε_tol bounds
            
            Parameters:
                iteration_history: List[OptimizationIteration] - convergence sequence
                target_range_km: s_target [km] - target range
                tolerance_km: ε_tol [km] - convergence tolerance
                save_path: str - HTML export path (optional)
                
            Returns:
                go.Figure: interactive Plotly figure
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
    
    # ────────────────────────────────────────────────────────────────────
    # Adjustment Strategy Visualization
    # ────────────────────────────────────────────────────────────────────
    
    class AdjustmentPlotter:
        """
        Cruise distance adjustment visualization: s_cruise,k and Δs_cruise,k.
        
        Plots:
            Panel 1: s_cruise,k trajectory
            Panel 2: Δs_cruise,k = s_cruise,k - s_cruise,k-1 per iteration
        """
        
        @staticmethod
        def plot_cruise_adjustment_strategy(
            iteration_history: List,
            save_path: Optional[str] = None
        ) -> go.Figure:
            """
            Generate 2-panel adjustment strategy figure.
            
            Panel 1: s_cruise,k vs. k (evolution)
            Panel 2: Δs_cruise,k vs. k (bar chart, color-coded by sign)
            
            Parameters:
                iteration_history: List[OptimizationIteration] - adjustment sequence
                save_path: str - HTML export path (optional)
                
            Returns:
                go.Figure: interactive Plotly figure
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
    
    # ────────────────────────────────────────────────────────────────────
    # Phase Breakdown Visualization
    # ────────────────────────────────────────────────────────────────────
    
    class BreakdownPlotter:
        """
        Phase-wise distance contributions: s_climb, s_cruise, s_descent.
        
        Plots:
            Panel 1: Stacked bar chart (absolute distances [km])
            Panel 2: Stacked bar chart (relative percentages [%])
        """
        
        @staticmethod
        def plot_distance_breakdown_evolution(
            iteration_data: List[Dict[str, Any]],
            save_path: Optional[str] = None
        ) -> go.Figure:
            """
            Generate 2-panel phase breakdown figure.
            
            Panel 1: s_phase vs. k (absolute [km], stacked)
            Panel 2: s_phase/s_total vs. k (percentage [%], stacked)
            
            Parameters:
                iteration_data: List[Dict] - phase breakdown per iteration
                    Each dict: {'iteration', 'climb_km', 'cruise_km', 'descent_km', 'total_km'}
                save_path: str - HTML export path (optional)
                
            Returns:
                go.Figure: interactive Plotly figure
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
    
    # ────────────────────────────────────────────────────────────────────
    # Dashboard Generation
    # ────────────────────────────────────────────────────────────────────
    
    class DashboardGenerator:
        """
        Comprehensive dashboard generation with phase plots.
        
        Operations:
            - Create optimization visualizations (convergence, adjustment, breakdown)
            - Generate phase-specific plots (climb, cruise, descent)
            - Export mission summary dashboards
        """
        
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
            Generate and export all phase plots for converged mission.
            
            Directory structure:
                base_output_dir/
                    Climb/          - climb phase plots
                    Cruise/         - cruise phase plots
                    Descent/        - descent phase plots
                    Mission_Summary/ - combined mission analysis
            
            Parameters:
                climb_result: MinFuelSchedule - climb trajectory
                cruise_result: CruiseResults - cruise trajectory
                descent_result: DescentResults - descent trajectory
                climb_info: dict - climb DP metadata
                descent_info: dict - descent DP metadata
                aero: aerodynamics model
                engine: propulsion model
                initial_mass_kg: m_0 [kg] - initial mass
                base_output_dir: Path - output directory root
            """
            print("\n" + "="*80)
            print("SAVING CONVERGED MISSION PHASE PLOTS")
            print("="*80)
            
            try:
                from climb_plotting import plot_performance_2d as plot_climb_performance_2d
                from cruise_plotting import plot_performance_2d as plot_cruise_performance_2d
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
                plot_climb_performance_2d(climb_result, climb_info)
                print(f"[CLIMB] Climb plots saved to: {climb_dir}")
            except Exception as e:
                print(f"[ERROR] Failed to save climb plots: {e}")
            
            print(f"\n[CRUISE] Generating and saving cruise phase plots...")
            try:
                plot_cruise_performance_2d(cruise_result)
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
                    initial_mass_kg=initial_mass_kg,
                    save_to_optimized=True
                )
                
                plot_combined_performance_analysis(
                    climb_result=climb_result,
                    cruise_result=cruise_result,
                    descent_result=descent_result,
                    initial_mass_kg=initial_mass_kg,
                    save_to_optimized=True
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
            Generate complete optimization dashboard with all visualizations.
            
            Generated plots:
                - convergence.html: e_k evolution and s_total vs. s_target
                - cruise_adjustment.html: s_cruise,k trajectory and Δs_cruise,k
                - distance_breakdown.html: Phase contributions (absolute and %)
                - optimization_summary.html: Convergence statistics table
                - Phase plots: If converged (climb, cruise, descent, mission summary)
            
            Parameters:
                iteration_history: List[OptimizationIteration] - convergence data
                iteration_data: List[Dict] - phase breakdown per iteration
                optimization_summary: Dict - convergence statistics
                save_dir: str - output directory (optional, uses default if None)
                climb_result: MinFuelSchedule - climb trajectory (optional)
                cruise_result: CruiseResults - cruise trajectory (optional)
                descent_result: DescentResults - descent trajectory (optional)
                climb_info: dict - climb metadata (optional)
                descent_info: dict - descent metadata (optional)
                aero: aerodynamics model (optional)
                engine: propulsion model (optional)
                initial_mass_kg: m_0 [kg] (optional)
                
            Returns:
                Dict[str, go.Figure]: {'convergence', 'adjustment', 'breakdown', 'summary'}
            """
            figures = {}
            
            # Check if called from main_range_optimizer to save to Optimized folder
            import inspect
            frame = inspect.currentframe()
            try:
                caller_frame = frame.f_back
                caller_file = caller_frame.f_globals.get('__file__', '')
                save_to_optimized = 'main_range_optimizer' in caller_file
            except:
                save_to_optimized = False
            finally:
                del frame
            
            if save_dir:
                output_path = Path(save_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                print(f"\n[DASHBOARD] Creating optimization dashboard in: {output_path}")
            elif save_to_optimized:
                output_path = Path(get_or_create_run_directory(phase="Optimized"))
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
            # Save PNG version
            try:
                fig_convergence.write_image(str(range_opt_path / "convergence_history.png"), width=1600, height=1000, scale=2)
                print(f"[EXPORT] Convergence history PNG saved to: {range_opt_path / 'convergence_history.png'}")
            except Exception as e:
                print(f"[WARNING] Could not save convergence history PNG: {e}")
            
            print("[DASHBOARD] Generating cruise adjustment strategy plot...")
            fig_adjustment = RangeOptimizationVisualization.AdjustmentPlotter.plot_cruise_adjustment_strategy(
                iteration_history,
                save_path=str(range_opt_path / "cruise_adjustment.html")
            )
            figures['adjustment'] = fig_adjustment
            # Save PNG version
            try:
                fig_adjustment.write_image(str(range_opt_path / "cruise_adjustment.png"), width=1600, height=1000, scale=2)
                print(f"[EXPORT] Cruise adjustment PNG saved to: {range_opt_path / 'cruise_adjustment.png'}")
            except Exception as e:
                print(f"[WARNING] Could not save cruise adjustment PNG: {e}")
            
            if iteration_data:
                print("[DASHBOARD] Generating distance breakdown plot...")
                fig_breakdown = RangeOptimizationVisualization.BreakdownPlotter.plot_distance_breakdown_evolution(
                    iteration_data,
                    save_path=str(range_opt_path / "distance_breakdown.html")
                )
                figures['breakdown'] = fig_breakdown
                # Save PNG version
                try:
                    fig_breakdown.write_image(str(range_opt_path / "distance_breakdown.png"), width=1600, height=1000, scale=2)
                    print(f"[EXPORT] Distance breakdown PNG saved to: {range_opt_path / 'distance_breakdown.png'}")
                except Exception as e:
                    print(f"[WARNING] Could not save distance breakdown PNG: {e}")
            
            print("[DASHBOARD] Generating optimization summary table...")
            fig_summary = RangeOptimizationVisualization.SummaryGenerator.create_optimization_summary_table(
                optimization_summary,
                save_path=str(range_opt_path / "optimization_summary.html")
            )
            figures['summary'] = fig_summary
            # Save PNG version
            try:
                fig_summary.write_image(str(range_opt_path / "optimization_summary.png"), width=1600, height=800, scale=2)
                print(f"[EXPORT] Optimization summary PNG saved to: {range_opt_path / 'optimization_summary.png'}")
            except Exception as e:
                print(f"[WARNING] Could not save optimization summary PNG: {e}")
            
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
    
    # ────────────────────────────────────────────────────────────────────
    # Summary Table Generation
    # ────────────────────────────────────────────────────────────────────
    
    class SummaryGenerator:
        """
        Tabular summary generation for convergence statistics.
        
        Table rows:
            - Target range: s_target [km]
            - Tolerance: ε_tol [km]
            - Damping factor: α
            - Total iterations: N_iter
            - Convergence status: |e_final| < ε_tol
            - Final results: s_cruise, s_total, e_final
            - Convergence rate: (1 - |e_final|/|e_0|)·100%
        """
        
        @staticmethod
        def create_optimization_summary_table(
            summary: Dict[str, Any],
            save_path: Optional[str] = None
        ) -> go.Figure:
            """
            Generate formatted convergence summary table.
            
            Parameters:
                summary: Dict - optimization results
                    {'target_range_km', 'tolerance_km', 'damping_factor', 
                     'total_iterations', 'converged', 'final_cruise_distance_km',
                     'final_total_distance_km', 'final_error_km', 'iteration_history'}
                save_path: str - HTML export path (optional)
                
            Returns:
                go.Figure: Plotly table figure
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
    
    # ────────────────────────────────────────────────────────────────────
    # Text Reporting Utilities
    # ────────────────────────────────────────────────────────────────────
    
    class Utilities:
        """Console output utilities for optimization reporting."""
        
        @staticmethod
        def print_optimization_report(optimization_summary: Dict[str, Any]):
            """
            Print formatted convergence report to console.
            
            Output:
                - Header: Target, tolerance, damping, iterations
                - Final results: s_cruise, s_total, e_final, relative error
                - Iteration table: k, s_cruise,k, s_total,k, e_k, convergence status
            
            Parameters:
                optimization_summary: Dict - convergence statistics
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


# ========================================================================
# SECTION 2: PUBLIC API
# ========================================================================

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
    Generate complete optimization dashboard with all visualizations.
    
    Wrapper for RangeOptimizationVisualization.DashboardGenerator.create_optimization_dashboard.
    """
    return RangeOptimizationVisualization.DashboardGenerator.create_optimization_dashboard(
        iteration_history, iteration_data, optimization_summary, save_dir,
        climb_result, cruise_result, descent_result,
        climb_info, descent_info, aero, engine, initial_mass_kg
    )

def print_optimization_report(optimization_summary: Dict[str, Any]):
    """
    Print formatted convergence report to console.
    
    Wrapper for RangeOptimizationVisualization.Utilities.print_optimization_report.
    """
    return RangeOptimizationVisualization.Utilities.print_optimization_report(optimization_summary)
