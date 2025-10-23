# =========================================================================
# FUEL OPTIMIZATION MODULE
# =========================================================================
"""
Iterative fuel load optimization for mission planning.

This module implements an iterative approach to optimize the initial fuel load
based on actual mission fuel consumption. The optimization continues until
convergence is achieved, ensuring realistic fuel planning with safety margins.
"""

from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time

from aircraft_config import SystemConfiguration, INITIAL_MASS_KG, W_OE_KG, W_PL_KG, MAX_FUEL_KG

@dataclass
class FuelOptimizationResult:
    """Results from fuel optimization iteration."""
    iteration: int
    initial_fuel_kg: float
    fuel_consumed_kg: float
    fuel_remaining_kg: float
    initial_mass_kg: float
    final_mass_kg: float
    convergence_error: float
    converged: bool
    mission_success: bool
    error_message: Optional[str] = None

@dataclass
class FuelOptimizationConfig:
    """Configuration for fuel optimization."""
    max_iterations: int = 20
    convergence_tolerance_percent: float = 5.0  # 5% weight difference convergence
    safety_factor: float = 1.2
    min_fuel_kg: float = 1000.0  # Minimum fuel load
    max_fuel_kg: float = MAX_FUEL_KG  # Maximum fuel load
    enable_visualization: bool = True
    visualization_update_interval: int = 1  # Update every N iterations

class FuelOptimizer:
    """Iterative fuel load optimizer for mission planning."""
    
    def __init__(self, config: FuelOptimizationConfig = None):
        """Initialize the fuel optimizer."""
        self.config = config or FuelOptimizationConfig()
        self.aircraft_config = SystemConfiguration()
        self.optimization_history: List[FuelOptimizationResult] = []
        self.convergence_fig = None
        
    def optimize_fuel_load(self, mission_simulator_func, initial_fuel_kg: float = None) -> FuelOptimizationResult:
        """
        Optimize fuel load through iterative mission simulation.
        
        Args:
            mission_simulator_func: Function that simulates mission and returns fuel consumption
            initial_fuel_kg: Starting fuel load (default: max fuel capacity)
            
        Returns:
            Final optimization result
        """
        if initial_fuel_kg is None:
            initial_fuel_kg = self.config.max_fuel_kg
            
        print(f"\n{'='*80}")
        print("FUEL LOAD OPTIMIZATION")
        print(f"{'='*80}")
        print(f"Starting fuel load: {initial_fuel_kg:.1f} kg")
        print(f"Convergence tolerance: {self.config.convergence_tolerance_percent:.1f}% weight difference")
        print(f"Safety factor: {self.config.safety_factor:.2f}")
        print(f"Max iterations: {self.config.max_iterations}")
        print(f"{'='*80}")
        
        current_fuel_kg = initial_fuel_kg
        self.optimization_history = []
        
        # Initialize convergence visualization
        if self.config.enable_visualization:
            print(f"[DEBUG] Initializing convergence visualization...")
            self._initialize_convergence_plot()
            print(f"[DEBUG] Convergence plot initialized")
        
        for iteration in range(self.config.max_iterations):
            print(f"\n[ITERATION {iteration + 1}/{self.config.max_iterations}]")
            print(f"Current fuel load: {current_fuel_kg:.1f} kg")
            
            # Calculate initial mass with current fuel load
            initial_mass_kg = self.aircraft_config.get_initial_mass_with_fuel(current_fuel_kg)
            print(f"Initial mass: {initial_mass_kg:.1f} kg")
            
            try:
                # Run mission simulation
                mission_result = mission_simulator_func(initial_mass_kg)
                
                # Extract fuel consumption from mission result
                fuel_consumed_kg = self._extract_fuel_consumption(mission_result)
                fuel_remaining_kg = self.aircraft_config.get_fuel_remaining_after_mission(
                    current_fuel_kg, fuel_consumed_kg
                )
                final_mass_kg = initial_mass_kg - fuel_consumed_kg
                
                # Calculate convergence error based on weight difference percentage
                if iteration == 0:
                    convergence_error = float('inf')
                else:
                    prev_mass = self.optimization_history[-1].initial_mass_kg
                    current_mass = initial_mass_kg
                    weight_difference = abs(current_mass - prev_mass)
                    convergence_error = (weight_difference / prev_mass) * 100.0  # Percentage
                
                # Check convergence (5% weight difference)
                converged = convergence_error <= self.config.convergence_tolerance_percent
                mission_success = fuel_consumed_kg > 0 and fuel_remaining_kg >= 0
                
                # Create result
                result = FuelOptimizationResult(
                    iteration=iteration + 1,
                    initial_fuel_kg=current_fuel_kg,
                    fuel_consumed_kg=fuel_consumed_kg,
                    fuel_remaining_kg=fuel_remaining_kg,
                    initial_mass_kg=initial_mass_kg,
                    final_mass_kg=final_mass_kg,
                    convergence_error=convergence_error,
                    converged=converged,
                    mission_success=mission_success
                )
                
                self.optimization_history.append(result)
                
                # Print iteration results
                print(f"Fuel consumed: {fuel_consumed_kg:.1f} kg")
                print(f"Fuel remaining: {fuel_remaining_kg:.1f} kg")
                print(f"Weight difference: {convergence_error:.2f}%")
                print(f"Converged: {converged}")
                print(f"Mission success: {mission_success}")
                
                # Update convergence plot
                if self.config.enable_visualization and (iteration + 1) % self.config.visualization_update_interval == 0:
                    print(f"[DEBUG] Updating convergence plot for iteration {iteration + 1}")
                    self._update_convergence_plot()
                
                # Check if converged
                if converged and mission_success:
                    print(f"\n✓ CONVERGENCE ACHIEVED after {iteration + 1} iterations!")
                    print(f"Optimal fuel load: {current_fuel_kg:.1f} kg")
                    print(f"Fuel consumed: {fuel_consumed_kg:.1f} kg")
                    print(f"Safety margin: {fuel_remaining_kg:.1f} kg")
                    break
                
                # Calculate next fuel load with safety factor
                if mission_success:
                    next_fuel_kg = fuel_consumed_kg * self.config.safety_factor
                    next_fuel_kg = np.clip(next_fuel_kg, self.config.min_fuel_kg, self.config.max_fuel_kg)
                    print(f"Next fuel load: {next_fuel_kg:.1f} kg (consumed × {self.config.safety_factor:.2f})")
                    current_fuel_kg = next_fuel_kg
                else:
                    print("❌ Mission failed - stopping optimization")
                    result.error_message = "Mission simulation failed"
                    break
                    
            except Exception as e:
                print(f"❌ Error in iteration {iteration + 1}: {str(e)}")
                result = FuelOptimizationResult(
                    iteration=iteration + 1,
                    initial_fuel_kg=current_fuel_kg,
                    fuel_consumed_kg=0.0,
                    fuel_remaining_kg=current_fuel_kg,
                    initial_mass_kg=initial_mass_kg,
                    final_mass_kg=initial_mass_kg,
                    convergence_error=float('inf'),
                    converged=False,
                    mission_success=False,
                    error_message=str(e)
                )
                self.optimization_history.append(result)
                break
        
        # Final summary
        self._print_final_summary()
        
        # Save convergence plot
        if self.config.enable_visualization:
            print(f"[DEBUG] Saving convergence plot...")
            self._save_convergence_plot()
        
        return self.optimization_history[-1] if self.optimization_history else None
    
    def _extract_fuel_consumption(self, mission_result) -> float:
        """Extract fuel consumption from mission result."""
        # This method needs to be adapted based on your mission result structure
        if hasattr(mission_result, 'total_fuel_kg'):
            return mission_result.total_fuel_kg
        elif hasattr(mission_result, 'fuel_consumed_kg'):
            return mission_result.fuel_consumed_kg
        elif hasattr(mission_result, 'cumFuel_kg'):
            return float(mission_result.cumFuel_kg[-1]) if len(mission_result.cumFuel_kg) > 0 else 0.0
        else:
            # Try to extract from climb + cruise + descent
            total_fuel = 0.0
            if hasattr(mission_result, 'climb_fuel'):
                total_fuel += mission_result.climb_fuel
            if hasattr(mission_result, 'cruise_fuel'):
                total_fuel += mission_result.cruise_fuel
            if hasattr(mission_result, 'descent_fuel'):
                total_fuel += mission_result.descent_fuel
            return total_fuel
    
    def _initialize_convergence_plot(self):
        """Initialize the convergence visualization plot."""
        self.convergence_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Fuel Load Convergence',
                'Fuel Consumption vs Load',
                'Mass Evolution',
                'Convergence Error'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Initialize empty traces
        self.convergence_fig.add_trace(
            go.Scatter(x=[], y=[], name='Fuel Load', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        self.convergence_fig.add_trace(
            go.Scatter(x=[], y=[], name='Fuel Consumed', line=dict(color='red', width=2)),
            row=1, col=2
        )
        self.convergence_fig.add_trace(
            go.Scatter(x=[], y=[], name='Initial Mass', line=dict(color='green', width=2)),
            row=2, col=1
        )
        self.convergence_fig.add_trace(
            go.Scatter(x=[], y=[], name='Convergence Error', line=dict(color='orange', width=2)),
            row=2, col=2
        )
        
        # Update layout
        self.convergence_fig.update_layout(
            title="Fuel Optimization Convergence",
            height=600,
            showlegend=True
        )
        
        # Update axes
        self.convergence_fig.update_xaxes(title_text="Iteration", row=2, col=1)
        self.convergence_fig.update_xaxes(title_text="Iteration", row=2, col=2)
        self.convergence_fig.update_yaxes(title_text="Fuel (kg)", row=1, col=1)
        self.convergence_fig.update_yaxes(title_text="Fuel (kg)", row=1, col=2)
        self.convergence_fig.update_yaxes(title_text="Mass (kg)", row=2, col=1)
        self.convergence_fig.update_yaxes(title_text="Error (kg)", row=2, col=2)
    
    def _update_convergence_plot(self):
        """Update the convergence plot with current data."""
        if not self.convergence_fig or not self.optimization_history:
            return
            
        iterations = [r.iteration for r in self.optimization_history]
        fuel_loads = [r.initial_fuel_kg for r in self.optimization_history]
        fuel_consumed = [r.fuel_consumed_kg for r in self.optimization_history]
        initial_masses = [r.initial_mass_kg for r in self.optimization_history]
        convergence_errors = [r.convergence_error for r in self.optimization_history]
        
        # Update traces
        self.convergence_fig.data[0].x = iterations
        self.convergence_fig.data[0].y = fuel_loads
        self.convergence_fig.data[1].x = iterations
        self.convergence_fig.data[1].y = fuel_consumed
        self.convergence_fig.data[2].x = iterations
        self.convergence_fig.data[2].y = initial_masses
        self.convergence_fig.data[3].x = iterations
        self.convergence_fig.data[3].y = convergence_errors
        
        # Add convergence line
        if len(iterations) > 1:
            tolerance_line = [self.config.convergence_tolerance] * len(iterations)
            if len(self.convergence_fig.data) < 5:
                self.convergence_fig.add_trace(
                    go.Scatter(x=iterations, y=tolerance_line, 
                             name='Tolerance', line=dict(color='red', dash='dash')),
                    row=2, col=2
                )
            else:
                self.convergence_fig.data[4].x = iterations
                self.convergence_fig.data[4].y = tolerance_line
    
    def _save_convergence_plot(self):
        """Save the convergence plot to HTML file and show it."""
        if self.convergence_fig:
            filename = f"fuel_optimization_convergence_{int(time.time())}.html"
            self.convergence_fig.write_html(filename)
            print(f"Convergence plot saved: {filename}")
            
            # Also show the plot in browser
            try:
                self.convergence_fig.show()
                print(f"Convergence plot opened in browser")
            except Exception as e:
                print(f"Could not open convergence plot in browser: {e}")
                print(f"Please open the file manually: {filename}")
    
    def _print_final_summary(self):
        """Print final optimization summary."""
        if not self.optimization_history:
            print("❌ No optimization history available")
            return
            
        final_result = self.optimization_history[-1]
        
        print(f"\n{'='*80}")
        print("FINAL OPTIMIZATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total iterations: {len(self.optimization_history)}")
        print(f"Converged: {final_result.converged}")
        print(f"Mission success: {final_result.mission_success}")
        
        if final_result.converged and final_result.mission_success:
            print(f"Optimal fuel load: {final_result.initial_fuel_kg:.1f} kg")
            print(f"Fuel consumed: {final_result.fuel_consumed_kg:.1f} kg")
            print(f"Fuel remaining: {final_result.fuel_remaining_kg:.1f} kg")
            print(f"Safety margin: {final_result.fuel_remaining_kg:.1f} kg")
            print(f"Final convergence error: {final_result.convergence_error:.1f} kg")
            
            # Calculate efficiency metrics
            fuel_efficiency = final_result.fuel_consumed_kg / final_result.initial_fuel_kg * 100
            print(f"Fuel efficiency: {fuel_efficiency:.1f}%")
        else:
            print(f"❌ Optimization failed: {final_result.error_message or 'Unknown error'}")
        
        print(f"{'='*80}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        if not self.optimization_history:
            return {}
            
        final_result = self.optimization_history[-1]
        
        return {
            'total_iterations': len(self.optimization_history),
            'converged': final_result.converged,
            'mission_success': final_result.mission_success,
            'optimal_fuel_kg': final_result.initial_fuel_kg,
            'fuel_consumed_kg': final_result.fuel_consumed_kg,
            'fuel_remaining_kg': final_result.fuel_remaining_kg,
            'final_convergence_error': final_result.convergence_error,
            'optimization_history': [
                {
                    'iteration': r.iteration,
                    'fuel_load': r.initial_fuel_kg,
                    'fuel_consumed': r.fuel_consumed_kg,
                    'convergence_error': r.convergence_error
                }
                for r in self.optimization_history
            ]
        }
