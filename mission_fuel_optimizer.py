# =========  1 - MODULE INITIALIZATION =================
# ========= IMPORTS AND BASIC SETUP ===========================================
"""
Fuel Capacity Optimization Module

This module implements convergent iterative optimization to determine the minimum 
required fuel capacity for mission completion. The optimization process employs
damped fixed-point iteration enhanced with Aitken's Δ² acceleration to iteratively 
refine the initial fuel load until convergence is achieved.

Scientific Approach:
- Fixed-point iteration with successive underrelaxation
- Aitken's Δ² acceleration method for adaptive convergence enhancement
- Dynamic mass evolution tracking throughout mission phases
- Comprehensive performance metrics calculation
- Safety buffer application to converged results

Module Structure:
1. Data Structures (iteration results, convergence history)
2. Mission Iteration Executor
3. Convergence Controller (optimization loop)
4. Configuration Manager (fuel parameter updates)

Key Features:
- Damped fixed-point iteration with initial damping factor (0.4)
- Aitken Δ² adaptive acceleration for convergence enhancement
- Fuel consumption tracking and convergence analysis
- Intermediate results storage without plot generation
- Automatic convergence detection with relative tolerance (0.5%)
- Safety buffer application (5%) after convergence
- Comprehensive error handling and recovery mechanisms

Author: Mission Analysis System
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import time

# Import mission phase modules
import climb
from climb import ClimbingCore, MinFuelSchedule
import cruise
from cruise import run_cruise_simulation, CruiseResults
import descent
from descent import run_descent_dp_optimization, DescentResults

# Import configuration
from aircraft_config import INITIAL_MASS_KG, MAX_FUEL_KG, W_OE_KG, W_PL_KG, AtmosphericProperties
from mission_config import (
    TARGET_ALT_CLIMB_M, START_ALTITUDE_CLIMB_M, START_VELOCITY_CLIMB_MS, START_LEVER_CLIMB,
    CRUISE_DISTANCE_KM, CRUISE_TIME_STEP_S,
    TARGET_DESCENT_ALT_M, TARGET_DESCENT_MACH,
    TARGET_MACH_CRUISE, TARGET_MACH_TOLERANCE_CLIMB,
    N_MACH_SAMPLES_CLIMB, N_ALTITUDE_STEPS_CLIMB, N_LEVER_SAMPLES_CLIMB,
    N_MACH_SAMPLES_DESCENT, N_ALTITUDE_STEPS_DESCENT, N_LEVER_SAMPLES_DESCENT,
    ALT_STEP_M
)

# Import wrappers
from pyaerodynamics_wrapper import PyAerodynamicsWrapper
from pyengine_wrapper import EngineWrapper


# =========  2 - CONVERGENCE PARAMETERS =================
class ConvergenceParameters:
    """Centralized convergence control parameters for fuel optimization."""
    
    CONVERGENCE_TOLERANCE_RELATIVE = 0.005  # 0.5% relative tolerance
    CONVERGENCE_TOLERANCE_PERCENT = CONVERGENCE_TOLERANCE_RELATIVE * 100.0  # 0.5% in percentage units
    SAFETY_BUFFER_PERCENT = 0.05  # 5% safety buffer
    MAX_ITERATIONS = 5  # Safety limit to prevent infinite loops
    DAMPING_FACTOR = 0.4  # Initial relaxation parameter for fixed-point iteration
    USE_AITKEN_ACCELERATION = True  # Enable Aitken's Δ² acceleration method
    AITKEN_MIN_DAMPING = 0.1  # Minimum damping factor to prevent excessive updates
    AITKEN_MAX_DAMPING = 0.9  # Maximum damping factor to maintain stability


# Create global convergence parameters instance
_convergence_params = ConvergenceParameters()

# Backward compatibility constants
CONVERGENCE_TOLERANCE_RELATIVE = _convergence_params.CONVERGENCE_TOLERANCE_RELATIVE
CONVERGENCE_TOLERANCE_PERCENT = _convergence_params.CONVERGENCE_TOLERANCE_PERCENT
SAFETY_BUFFER_PERCENT = _convergence_params.SAFETY_BUFFER_PERCENT
MAX_ITERATIONS = _convergence_params.MAX_ITERATIONS
DAMPING_FACTOR = _convergence_params.DAMPING_FACTOR
USE_AITKEN_ACCELERATION = _convergence_params.USE_AITKEN_ACCELERATION
AITKEN_MIN_DAMPING = _convergence_params.AITKEN_MIN_DAMPING
AITKEN_MAX_DAMPING = _convergence_params.AITKEN_MAX_DAMPING


# =========  3 - DATA STRUCTURES =================
@dataclass
class MissionIterationResults:
    """
    Results from a single mission iteration.
    
    This structure encapsulates complete mission simulation results including
    fuel consumption, performance metrics, and phase-wise detailed results.
    
    Attributes:
        iteration: Iteration number
        initial_fuel_kg: Initial fuel load for this iteration [kg]
        initial_mass_kg: Initial total aircraft mass [kg]
        fuel_consumed_kg: Total fuel consumed across all phases [kg]
        convergence_delta_percent: Relative change from previous iteration [%]
        climb_result: Climb phase optimization results
        cruise_result: Cruise phase simulation results
        descent_result: Descent phase optimization results
        total_time_s: Total mission duration [s]
        total_distance_km: Total mission ground distance [km]
        final_weight_kg: Final aircraft weight after mission [kg]
        climb_fuel_kg: Fuel consumed in climb phase [kg]
        cruise_fuel_kg: Fuel consumed in cruise phase [kg]
        descent_fuel_kg: Fuel consumed in descent phase [kg]
        climb_time_s: Climb phase duration [s]
        cruise_time_s: Cruise phase duration [s]
        descent_time_s: Descent phase duration [s]
        avg_lift_climb_N: Average lift force during climb [N]
        avg_drag_climb_N: Average drag force during climb [N]
        avg_lift_cruise_N: Average lift force during cruise [N]
        avg_drag_cruise_N: Average drag force during cruise [N]
        avg_lift_descent_N: Average lift force during descent [N]
        avg_drag_descent_N: Average drag force during descent [N]
        avg_ld_climb: Average lift-to-drag ratio during climb
        avg_ld_cruise: Average lift-to-drag ratio during cruise
        avg_ld_descent: Average lift-to-drag ratio during descent
        avg_lever_climb: Average thrust lever position during climb
        avg_lever_cruise: Average thrust lever position during cruise
        avg_lever_descent: Average thrust lever position during descent
        avg_specific_energy_climb_J_kg: Average specific energy during climb [J/kg]
        avg_specific_energy_cruise_J_kg: Average specific energy during cruise [J/kg]
        avg_specific_energy_descent_J_kg: Average specific energy during descent [J/kg]
    """
    # Required fields (no defaults)
    iteration: int
    initial_fuel_kg: float
    initial_mass_kg: float
    fuel_consumed_kg: float
    convergence_delta_percent: float
    
    # Phase-wise results (no defaults)
    climb_result: MinFuelSchedule
    cruise_result: CruiseResults
    descent_result: DescentResults
    
    # Mission totals (no defaults)
    total_time_s: float
    total_distance_km: float
    
    # Key performance parameters (no defaults)
    final_weight_kg: float
    climb_fuel_kg: float
    cruise_fuel_kg: float
    descent_fuel_kg: float
    climb_time_s: float
    cruise_time_s: float
    descent_time_s: float
    
    # Optional fields with defaults (must come after fields without defaults)
    # Aerodynamic performance tracking
    avg_lift_climb_N: float = 0.0
    avg_drag_climb_N: float = 0.0
    avg_lift_cruise_N: float = 0.0
    avg_drag_cruise_N: float = 0.0
    avg_lift_descent_N: float = 0.0
    avg_drag_descent_N: float = 0.0
    
    # L/D ratios
    avg_ld_climb: float = 0.0
    avg_ld_cruise: float = 0.0
    avg_ld_descent: float = 0.0
    
    # Thrust lever positions
    avg_lever_climb: float = 0.0
    avg_lever_cruise: float = 0.0
    avg_lever_descent: float = 0.0
    
    # Specific energy (J/kg)
    avg_specific_energy_climb_J_kg: float = 0.0
    avg_specific_energy_cruise_J_kg: float = 0.0
    avg_specific_energy_descent_J_kg: float = 0.0


@dataclass
class ConvergenceHistory:
    """
    Tracking structure for convergence analysis.
    
    This structure maintains complete optimization history for analysis,
    diagnostics, and visualization purposes.
    
    Attributes:
        iterations: List of all mission iteration results
    """
    iterations: List[MissionIterationResults]
    
    def __init__(self):
        """Initialize empty convergence history."""
        self.iterations = []
    
    def add_iteration(self, result: MissionIterationResults):
        """
        Add iteration result to history.
        
        Args:
            result: Mission iteration results to add
        """
        self.iterations.append(result)
    
    def get_last_two_iterations(self) -> Tuple[MissionIterationResults, MissionIterationResults]:
        """
        Get the last two iterations for convergence analysis.
        
        Returns:
            Tuple of (previous_iteration, current_iteration)
            
        Raises:
            ValueError: If fewer than 2 iterations available
        """
        if len(self.iterations) < 2:
            raise ValueError("Need at least 2 iterations for convergence analysis")
        return self.iterations[-2], self.iterations[-1]
    
    def is_converged(self) -> bool:
        """
        Check if convergence has been achieved.
        
        Convergence criterion:
            |Δf_rel| < ε_tolerance
        
        where:
            Δf_rel = (F_consumed,k - F_consumed,k-1) / F_consumed,k-1
            ε_tolerance = CONVERGENCE_TOLERANCE_PERCENT
        
        Returns:
            bool: True if converged, False otherwise
        """
        if len(self.iterations) < 2:
            return False
        
        prev, curr = self.get_last_two_iterations()
        delta = curr.convergence_delta_percent
        
        return abs(delta) < CONVERGENCE_TOLERANCE_PERCENT


# =========  4 - FUEL OPTIMIZATION CORE SYSTEM =================
class FuelOptimizationCore:
    """
    Comprehensive fuel capacity optimization framework for minimum fuel determination.
    
    This class implements a complete computational framework for aircraft fuel capacity
    optimization through three integrated subsystems: mission iteration execution,
    convergence control with Aitken acceleration, and configuration management. The
    system determines minimum required fuel capacity through iterative convergence.
    
    System Components:
    - IterationExecutor: Executes complete mission simulations (climb + cruise + descent)
      with dynamic mass tracking and performance metrics calculation
    - ConvergenceController: Manages optimization loop with Aitken's Δ² acceleration,
      adaptive damping, and convergence detection for stable, rapid convergence
    - ConfigurationManager: Updates aircraft configuration files with optimized fuel
      capacity values for subsequent mission analysis
    
    Computational Features:
    - Fixed-point iteration with successive underrelaxation for stability
    - Aitken's Δ² acceleration for quadratic convergence enhancement
    - Dynamic mass evolution with fuel burn tracking
    - Comprehensive performance metrics (L/D, specific energy, lever positions)
    - Physical anomaly detection and diagnostic reporting
    - Safety buffer application to converged results
    
    Implementation:
        # Execute single mission iteration
        result = FuelOptimizationCore.IterationExecutor.run_single_mission_iteration(...)
        
        # Run complete optimization
        optimal, history = FuelOptimizationCore.ConvergenceController.optimize_fuel_capacity(...)
        
        # Update configuration
        FuelOptimizationCore.ConfigurationManager.apply_optimized_fuel_to_configuration(...)
    """
    
    # ========= MISSION ITERATION EXECUTOR =========
    class IterationExecutor:
        """Manages execution of complete mission iterations with performance tracking."""
        
        @staticmethod
        def run_single_mission_iteration(
            initial_mass_kg: float,
            aero: PyAerodynamicsWrapper,
            eng: EngineWrapper,
            M_grid: np.ndarray,
            H_plot: np.ndarray,
            lever_samples: int,
            print_progress: bool = True
        ) -> MissionIterationResults:
            """
            Execute a complete mission iteration (climb + cruise + descent).
            
            This function runs a full mission simulation with the specified initial mass,
            tracking fuel consumption, performance metrics, and trajectory data across
            all three flight phases.
            
            Args:
                initial_mass_kg: Initial aircraft mass for this iteration
                aero: Aerodynamics wrapper instance
                eng: Engine wrapper instance
                M_grid: Mach grid for optimization
                H_plot: Altitude grid for plotting
                lever_samples: Number of lever samples for DP optimization
                print_progress: Whether to print progress messages
                    
            Returns:
                MissionIterationResults containing all phase results and metrics
                
            Notes:
                - Mass decreases dynamically as fuel is consumed
                - Performance metrics calculated using instantaneous aircraft state
                - Distance integration accounts for climb and descent angles
            """
            atmospheric_props = AtmosphericProperties()
            iteration_start_time = time.time()
            
            if print_progress:
                print(f"\n[MISSION ITERATION] Running full mission with initial mass: {initial_mass_kg:.1f} kg")
            
            # ========= CLIMB PHASE =========================================
            if print_progress:
                print("[CLIMB] Computing optimal climb trajectory...")
            
            # Calculate starting Mach from takeoff velocity at start altitude
            a = atmospheric_props.a_from_altitude(START_ALTITUDE_CLIMB_M)
            start_mach = START_VELOCITY_CLIMB_MS / a
            
            # Create uniform altitude steps
            uniform_step_size = TARGET_ALT_CLIMB_M / len(H_plot)
            H_sched = np.arange(START_ALTITUDE_CLIMB_M, 
                                TARGET_ALT_CLIMB_M + uniform_step_size, 
                                uniform_step_size)
            
            # Solve 3D DP for climb
            dp_sched, dp_info = ClimbingCore.DynamicProgrammingOptimizer.solve_3d_fixed_mass(
                aero, eng, M_grid, H_sched, 
                lever_samples=lever_samples,
                target_mach=TARGET_MACH_CRUISE,
                target_mach_tolerance=TARGET_MACH_TOLERANCE_CLIMB,
                start_mach=start_mach,
                start_lever=START_LEVER_CLIMB,
                mass_kg=initial_mass_kg
            )
            
            climb_fuel = float(np.nan_to_num(dp_sched.cumFuel_kg, nan=0.0)[-1])
            climb_time_s = float(np.sum(np.nan_to_num(dp_sched.dt_s, nan=0.0)))
            climb_mass_end = initial_mass_kg - climb_fuel
            
            if print_progress:
                print(f"[CLIMB] Completed: {climb_time_s/60:.1f} min, {climb_fuel:.1f} kg fuel")
                print(f"[CLIMB] Mass: {initial_mass_kg:.1f} kg → {climb_mass_end:.1f} kg "
                      f"(burned {climb_fuel:.1f} kg, {climb_fuel/initial_mass_kg*100:.2f}%)")
            
            # ========= CRUISE PHASE =========================================
            if print_progress:
                print(f"[CRUISE] Starting cruise from altitude {dp_sched.alt_m[-1]:.0f}m, "
                      f"Mach {dp_sched.mach[-1]:.3f}")
            
            cruise_results = run_cruise_simulation(
                climb_result=dp_sched,
                initial_mass_kg=climb_mass_end,
                target_distance_km=CRUISE_DISTANCE_KM,
                aero=aero,
                engine=eng,
                time_step_s=CRUISE_TIME_STEP_S,
                create_plots=False
            )
            
            cruise_fuel = cruise_results.total_fuel_consumed_kg
            cruise_time_s = cruise_results.total_time_s
            cruise_mass_end = climb_mass_end - cruise_fuel
            
            if print_progress:
                print(f"[CRUISE] Completed: {cruise_time_s/3600:.2f} hours, {cruise_fuel:.1f} kg fuel")
                print(f"[CRUISE] Mass: {climb_mass_end:.1f} kg → {cruise_mass_end:.1f} kg "
                      f"(burned {cruise_fuel:.1f} kg, {cruise_fuel/climb_mass_end*100:.2f}%)")
            
            # ========= DESCENT PHASE =========================================
            if print_progress:
                print("[DESCENT] Computing optimal descent trajectory...")
            
            H_descent = np.linspace(cruise_results.altitude_m[-1], 
                                   TARGET_DESCENT_ALT_M, 
                                   N_ALTITUDE_STEPS_DESCENT)
            
            M_min_descent = max(0.2, TARGET_DESCENT_MACH - 0.1)
            M_max_descent = min(0.85, cruise_results.mach_number[-1] + 0.05)
            M_grid_descent = np.linspace(M_min_descent, M_max_descent, N_MACH_SAMPLES_DESCENT)
            
            descent_result, descent_info = run_descent_dp_optimization(
                cruise_results=cruise_results,
                climb_fuel_kg=climb_fuel,
                climb_time_s=climb_time_s,
                aero=aero,
                engine=eng,
                target_altitude_m=TARGET_DESCENT_ALT_M,
                target_mach=TARGET_DESCENT_MACH,
                n_altitude_steps=N_ALTITUDE_STEPS_DESCENT,
                n_mach_samples=N_MACH_SAMPLES_DESCENT,
                lever_samples=N_LEVER_SAMPLES_DESCENT
            )
            
            descent_fuel = descent_result.total_fuel_consumed_kg
            descent_time_s = descent_result.total_time_s
            descent_mass_end = cruise_mass_end - descent_fuel
            
            if print_progress:
                print(f"[DESCENT] Completed: {descent_time_s/60:.1f} min, {descent_fuel:.2f} kg fuel")
                print(f"[DESCENT] Mass: {cruise_mass_end:.1f} kg → {descent_mass_end:.1f} kg "
                      f"(burned {descent_fuel:.1f} kg, {descent_fuel/cruise_mass_end*100:.2f}%)")
            
            # ========= COMPUTE SUMMARY =========================================
            total_fuel = climb_fuel + cruise_fuel + descent_fuel
            total_time_s = climb_time_s + cruise_time_s + descent_time_s
            
            # Calculate mission distance and performance metrics
            total_distance_km = FuelOptimizationCore.PerformanceCalculator.calculate_mission_distance(
                dp_sched, cruise_results, descent_result, atmospheric_props
            )
            
            performance_metrics = FuelOptimizationCore.PerformanceCalculator.calculate_performance_metrics(
                dp_sched, cruise_results, descent_result, 
                initial_mass_kg, aero, atmospheric_props
            )
            
            # Warn if fuel consumed exceeds initial fuel significantly
            initial_fuel_kg = initial_mass_kg - W_OE_KG - W_PL_KG
            fuel_overage_percent = (total_fuel / initial_fuel_kg - 1) * 100.0 if initial_fuel_kg > 0 else 0.0
            
            if total_fuel > initial_fuel_kg and fuel_overage_percent > 100.0:
                if print_progress:
                    print(f"[WARNING] Large fuel deficit: consumed {total_fuel:.1f} kg, "
                          f"initial {initial_fuel_kg:.1f} kg")
                    print(f"[WARNING] Overage: {total_fuel - initial_fuel_kg:.1f} kg "
                          f"({fuel_overage_percent:.1f}%)")
                    print(f"[INFO] This is expected in early iterations - optimization will correct this")
            
            iteration_time = time.time() - iteration_start_time
            
            if print_progress:
                print(f"[ITERATION] Completed in {iteration_time:.1f}s")
                print(f"[MISSION TOTALS] Fuel: {total_fuel:.1f} kg, Time: {total_time_s/3600:.2f} hours")
            
            return MissionIterationResults(
                iteration=-1,  # Will be set by caller
                initial_fuel_kg=initial_fuel_kg,
                initial_mass_kg=initial_mass_kg,
                fuel_consumed_kg=total_fuel,
                convergence_delta_percent=0.0,  # Will be computed by caller
                climb_result=dp_sched,
                cruise_result=cruise_results,
                descent_result=descent_result,
                total_time_s=total_time_s,
                total_distance_km=total_distance_km,
                final_weight_kg=descent_result.final_weight_kg,
                climb_fuel_kg=climb_fuel,
                cruise_fuel_kg=cruise_fuel,
                descent_fuel_kg=descent_fuel,
                climb_time_s=climb_time_s,
                cruise_time_s=cruise_time_s,
                descent_time_s=descent_time_s,
                **performance_metrics
            )
    
    # ========= PERFORMANCE CALCULATOR =========
    class PerformanceCalculator:
        """Calculates performance metrics and mission statistics."""
        
        @staticmethod
        def calculate_mission_distance(
            climb_result: MinFuelSchedule,
            cruise_result: CruiseResults,
            descent_result: DescentResults,
            atmospheric_props: AtmosphericProperties
        ) -> float:
            """
            Calculate total mission distance using actual trajectory data.
            
            Distance integration accounts for climb and descent angles using
            horizontal distance components derived from velocity and time.
            
            Args:
                climb_result: Climb phase results
                cruise_result: Cruise phase results
                descent_result: Descent phase results
                atmospheric_props: Atmospheric properties calculator
                
            Returns:
                float: Total mission ground distance [km]
            """
            # Climb distance calculation
            climb_distance_km = 0.0
            if hasattr(climb_result, 'mach') and hasattr(climb_result, 'alt_m') and hasattr(climb_result, 'dt_s'):
                if len(climb_result.mach) > 0 and len(climb_result.alt_m) > 0 and len(climb_result.dt_s) > 0:
                    climb_distance_m = 0.0
                    for i in range(len(climb_result.dt_s)):
                        if i < len(climb_result.mach) and i < len(climb_result.alt_m):
                            a = atmospheric_props.a_from_altitude(climb_result.alt_m[i])
                            velocity_mps = climb_result.mach[i] * a
                            
                            # Horizontal distance accounting for climb angle
                            if i > 0 and i < len(climb_result.alt_m):
                                dh = climb_result.alt_m[i] - climb_result.alt_m[i-1]
                                ds_total = velocity_mps * climb_result.dt_s[i]
                                # Horizontal component: sqrt(ds² - dh²)
                                if ds_total**2 > dh**2:
                                    horizontal_distance = np.sqrt(ds_total**2 - dh**2)
                                else:
                                    horizontal_distance = 0.0
                                climb_distance_m += horizontal_distance
                            else:
                                climb_distance_m += velocity_mps * climb_result.dt_s[i]
                    climb_distance_km = climb_distance_m / 1000.0
            
            # Cruise distance from results
            cruise_distance_km = 0.0
            if hasattr(cruise_result, 'distance_km') and len(cruise_result.distance_km) > 0:
                cruise_distance_km = float(cruise_result.distance_km[-1])
            else:
                cruise_distance_km = CRUISE_DISTANCE_KM
            
            # Descent distance calculation
            descent_distance_km = 0.0
            if hasattr(descent_result, 'mach') and hasattr(descent_result, 'alt_m') and hasattr(descent_result, 'dt_s'):
                if len(descent_result.mach) > 0 and len(descent_result.alt_m) > 0 and len(descent_result.dt_s) > 0:
                    descent_distance_m = 0.0
                    for i in range(len(descent_result.dt_s)):
                        if i < len(descent_result.mach) and i < len(descent_result.alt_m):
                            a = atmospheric_props.a_from_altitude(descent_result.alt_m[i])
                            velocity_mps = descent_result.mach[i] * a
                            
                            # Horizontal distance accounting for descent angle
                            if i > 0 and i < len(descent_result.alt_m):
                                dh = descent_result.alt_m[i] - descent_result.alt_m[i-1]
                                ds_total = velocity_mps * descent_result.dt_s[i]
                                if ds_total**2 > dh**2:
                                    horizontal_distance = np.sqrt(ds_total**2 - dh**2)
                                else:
                                    horizontal_distance = 0.0
                                descent_distance_m += horizontal_distance
                            else:
                                descent_distance_m += velocity_mps * descent_result.dt_s[i]
                    descent_distance_km = descent_distance_m / 1000.0
            
            return climb_distance_km + cruise_distance_km + descent_distance_km
        
        @staticmethod
        def calculate_performance_metrics(
            climb_result: MinFuelSchedule,
            cruise_result: CruiseResults,
            descent_result: DescentResults,
            initial_mass_kg: float,
            aero: PyAerodynamicsWrapper,
            atmospheric_props: AtmosphericProperties
        ) -> Dict[str, float]:
            """
            Calculate comprehensive performance metrics for all mission phases.
            
            Computes aerodynamic efficiency (L/D ratios), engine utilization (lever positions),
            and energy management (specific energy) metrics across climb, cruise, and descent.
            
            Args:
                climb_result: Climb phase results
                cruise_result: Cruise phase results
                descent_result: Descent phase results
                initial_mass_kg: Initial aircraft mass [kg]
                aero: Aerodynamics wrapper
                atmospheric_props: Atmospheric properties calculator
                
            Returns:
                dict: Performance metrics for all phases
            """
            metrics = {}
            
            # ===== CLIMB PHASE METRICS =====
            avg_lift_climb_N = 0.0
            avg_drag_climb_N = 0.0
            avg_ld_climb = 0.0
            avg_lever_climb = 0.0
            avg_specific_energy_climb = 0.0
            
            if hasattr(climb_result, 'alt_m') and len(climb_result.alt_m) > 0:
                # Use actual cumulative fuel data for accurate weight trajectory
                if hasattr(climb_result, 'cumFuel_kg') and len(climb_result.cumFuel_kg) > 0:
                    cumulative_fuel = np.nan_to_num(climb_result.cumFuel_kg, nan=0.0)
                    weights = (initial_mass_kg - cumulative_fuel) * 9.81
                else:
                    climb_mass_end = initial_mass_kg - float(np.nan_to_num(climb_result.cumFuel_kg, nan=0.0)[-1])
                    weights = np.linspace(initial_mass_kg, climb_mass_end, len(climb_result.alt_m)) * 9.81
                
                avg_lift_climb_N = float(np.mean(weights))
                
                # Calculate detailed metrics
                drag_vals, ld_vals, lever_vals, se_vals = [], [], [], []
                
                for i in range(len(climb_result.alt_m)):
                    if hasattr(climb_result, 'mach') and hasattr(climb_result, 'alt_m'):
                        try:
                            _, _, rho = atmospheric_props.isa_properties(climb_result.alt_m[i])
                            a = atmospheric_props.a_from_altitude(climb_result.alt_m[i])
                            
                            weight_kg = weights[i] / 9.81
                            CD = aero.get_drag_coefficient(climb_result.mach[i], climb_result.alt_m[i], weight_kg)
                            drag = CD * 0.5 * rho * (climb_result.mach[i] * a)**2 * aero.params['S_REF_M2']
                            drag_vals.append(drag)
                            
                            if drag > 0:
                                ld_vals.append(weights[i] / drag)
                            
                            if hasattr(climb_result, 'lever'):
                                lever_vals.append(climb_result.lever[i])
                            
                            velocity = climb_result.mach[i] * a
                            pe = 9.81 * climb_result.alt_m[i]
                            ke = 0.5 * velocity**2
                            se_vals.append(pe + ke)
                        except:
                            pass
                
                if len(drag_vals) > 0:
                    avg_drag_climb_N = float(np.mean(drag_vals))
                if len(ld_vals) > 0:
                    avg_ld_climb = float(np.mean(ld_vals))
                if len(lever_vals) > 0:
                    avg_lever_climb = float(np.mean(lever_vals))
                if len(se_vals) > 0:
                    avg_specific_energy_climb = float(np.mean(se_vals))
            
            # ===== CRUISE PHASE METRICS =====
            avg_lift_cruise_N = 0.0
            avg_drag_cruise_N = 0.0
            avg_ld_cruise = 0.0
            avg_lever_cruise = 0.0
            avg_specific_energy_cruise = 0.0
            
            if hasattr(cruise_result, 'weight_kg') and len(cruise_result.weight_kg) > 0:
                avg_lift_cruise_N = float(np.mean(cruise_result.weight_kg) * 9.81)
            
            if hasattr(cruise_result, 'drag_N') and len(cruise_result.drag_N) > 0:
                avg_drag_cruise_N = float(np.mean(cruise_result.drag_N))
                if avg_drag_cruise_N > 0:
                    avg_ld_cruise = avg_lift_cruise_N / avg_drag_cruise_N
            
            if hasattr(cruise_result, 'lever_position') and len(cruise_result.lever_position) > 0:
                avg_lever_cruise = float(np.mean(cruise_result.lever_position))
            
            if hasattr(cruise_result, 'altitude_m') and hasattr(cruise_result, 'mach_number'):
                if len(cruise_result.altitude_m) > 0 and len(cruise_result.mach_number) > 0:
                    velocities = cruise_result.mach_number * np.array([atmospheric_props.a_from_altitude(h) 
                                                                       for h in cruise_result.altitude_m])
                    pe = 9.81 * np.array(cruise_result.altitude_m)
                    ke = 0.5 * velocities**2
                    avg_specific_energy_cruise = float(np.mean(pe + ke))
            
            # ===== DESCENT PHASE METRICS =====
            avg_lift_descent_N = 0.0
            avg_drag_descent_N = 0.0
            avg_ld_descent = 0.0
            avg_lever_descent = 0.0
            avg_specific_energy_descent = 0.0
            
            if hasattr(descent_result, 'weight_kg') and len(descent_result.weight_kg) > 0:
                avg_lift_descent_N = float(np.mean(descent_result.weight_kg) * 9.81)
                
                if hasattr(descent_result, 'mach') and hasattr(descent_result, 'alt_m'):
                    drag_vals, ld_vals, lever_vals, se_vals = [], [], [], []
                    
                    for i in range(len(descent_result.alt_m)):
                        try:
                            weight = descent_result.weight_kg[i] * 9.81
                            _, _, rho = atmospheric_props.isa_properties(descent_result.alt_m[i])
                            a = atmospheric_props.a_from_altitude(descent_result.alt_m[i])
                            
                            weight_kg = descent_result.weight_kg[i]
                            CD = aero.get_drag_coefficient(descent_result.mach[i], descent_result.alt_m[i], weight_kg)
                            drag = CD * 0.5 * rho * (descent_result.mach[i] * a)**2 * aero.params['S_REF_M2']
                            drag_vals.append(drag)
                            
                            if drag > 0:
                                ld_vals.append(weight / drag)
                            
                            if hasattr(descent_result, 'lever'):
                                lever_vals.append(descent_result.lever[i])
                            
                            velocity = descent_result.mach[i] * a
                            pe = 9.81 * descent_result.alt_m[i]
                            ke = 0.5 * velocity**2
                            se_vals.append(pe + ke)
                        except:
                            pass
                    
                    if len(drag_vals) > 0:
                        avg_drag_descent_N = float(np.mean(drag_vals))
                    if len(ld_vals) > 0:
                        avg_ld_descent = float(np.mean(ld_vals))
                    if len(lever_vals) > 0:
                        avg_lever_descent = float(np.mean(lever_vals))
                    if len(se_vals) > 0:
                        avg_specific_energy_descent = float(np.mean(se_vals))
            
            return {
                'avg_lift_climb_N': avg_lift_climb_N,
                'avg_drag_climb_N': avg_drag_climb_N,
                'avg_lift_cruise_N': avg_lift_cruise_N,
                'avg_drag_cruise_N': avg_drag_cruise_N,
                'avg_lift_descent_N': avg_lift_descent_N,
                'avg_drag_descent_N': avg_drag_descent_N,
                'avg_ld_climb': avg_ld_climb,
                'avg_ld_cruise': avg_ld_cruise,
                'avg_ld_descent': avg_ld_descent,
                'avg_lever_climb': avg_lever_climb,
                'avg_lever_cruise': avg_lever_cruise,
                'avg_lever_descent': avg_lever_descent,
                'avg_specific_energy_climb_J_kg': avg_specific_energy_climb,
                'avg_specific_energy_cruise_J_kg': avg_specific_energy_cruise,
                'avg_specific_energy_descent_J_kg': avg_specific_energy_descent
            }
    
    # ========= CONVERGENCE CONTROLLER =========
    class ConvergenceController:
        """Manages optimization loop with Aitken acceleration and convergence control."""
        
        @staticmethod
        def optimize_fuel_capacity(
            aero: PyAerodynamicsWrapper,
            eng: EngineWrapper,
            M_grid: np.ndarray,
            H_plot: np.ndarray,
            lever_samples: int = 50
        ) -> Tuple[MissionIterationResults, ConvergenceHistory]:
            """
            Main optimization loop to determine minimum required fuel capacity.
            
            Implements fixed-point iteration with Aitken's Δ² acceleration method
            for adaptive convergence enhancement.
            
            Process:
            1. Start with MAX_FUEL_KG as initial guess
            2. Run full mission and record fuel consumed
            3. Update fuel using Aitken acceleration (iter ≥ 3) or fixed damping (iter < 3)
            4. Repeat until convergence (fuel difference < 0.5%)
            5. Apply 5% safety buffer to final result
            
            Aitken Acceleration (Aitken, 1926):
            Adaptive relaxation method that computes optimal damping factor based on 
            convergence history. For linearly convergent sequences, achieves quadratic 
            convergence acceleration.
            
            Mathematical formulation:
                Δf_k = f_consumed_k - f_consumed_{k-1}
                Δf_{k-1} = f_consumed_{k-1} - f_consumed_{k-2}
                ω_k = ω_{k-1} × (1 - Δf_k / (Δf_k - Δf_{k-1}))
                f_next = ω_k × f_consumed + (1-ω_k) × f_current
            
            Args:
                aero: Aerodynamics wrapper instance
                eng: Engine wrapper instance  
                M_grid: Mach grid for optimization
                H_plot: Altitude grid for plotting
                lever_samples: Number of lever samples for DP optimization
                
            Returns:
                Tuple of (final optimized result, convergence history)
                
            References:
                - Aitken, A.C. (1926). "On Bernoulli's numerical solution of algebraic equations"
                - Burden & Faires, "Numerical Analysis" (Fixed-point iteration chapter)
            """
            print("\n" + "="*80)
            print("FUEL CAPACITY OPTIMIZATION WITH AITKEN ACCELERATION")
            print("="*80)
            print(f"Objective: Determine minimum required fuel for mission completion")
            print(f"Convergence criterion: {CONVERGENCE_TOLERANCE_PERCENT:.2f}% relative")
            print(f"Safety buffer: {SAFETY_BUFFER_PERCENT*100:.0f}%")
            print(f"Method: {'Aitken Δ² Acceleration' if USE_AITKEN_ACCELERATION else 'Fixed Damping'}")
            print(f"Initial damping factor: {DAMPING_FACTOR:.2f}")
            print("="*80)
            
            # Initialize with maximum fuel capacity
            initial_fuel_current_kg = MAX_FUEL_KG
            history = ConvergenceHistory()
            current_damping = DAMPING_FACTOR
            iteration_count = 0
            
            while iteration_count < MAX_ITERATIONS:
                iteration_count += 1
                
                # Calculate current total mass
                current_total_mass = W_OE_KG + W_PL_KG + initial_fuel_current_kg
                
                print(f"\n[ITERATION {iteration_count}] Initial fuel: {initial_fuel_current_kg:.1f} kg")
                print(f"[ITERATION {iteration_count}] Total mass: {current_total_mass:.1f} kg")
                
                # Run single mission iteration
                try:
                    iteration_result = FuelOptimizationCore.IterationExecutor.run_single_mission_iteration(
                        initial_mass_kg=current_total_mass,
                        aero=aero,
                        eng=eng,
                        M_grid=M_grid,
                        H_plot=H_plot,
                        lever_samples=lever_samples,
                        print_progress=True
                    )
                except RuntimeError as e:
                    print(f"\n[ERROR] Mission failed at iteration {iteration_count}: {str(e)}")
                    print(f"[ERROR] Initial fuel: {initial_fuel_current_kg:.1f} kg")
                    
                    if iteration_count == 1:
                        raise RuntimeError(
                            f"Mission infeasible even with MAX_FUEL_KG ({MAX_FUEL_KG:.1f} kg). "
                            f"Check TARGET_ALT_CLIMB_M, CRUISE_DISTANCE_KM, or other mission parameters."
                        )
                    elif len(history.iterations) > 0:
                        last_successful = history.iterations[-1]
                        print(f"\n[INFO] Fixed-point iteration reached numerical boundary")
                        print(f"[INFO] Last successful fuel: {last_successful.initial_fuel_kg:.1f} kg")
                        print(f"[INFO] Fuel consumed: {last_successful.fuel_consumed_kg:.1f} kg")
                        raise RuntimeError(
                            f"Fixed-point iteration failed. Last successful fuel: "
                            f"{last_successful.initial_fuel_kg:.1f} kg."
                        )
                    else:
                        raise
                
                # Store iteration results
                iteration_result.iteration = iteration_count
                if iteration_count > 1:
                    prev_result = history.iterations[-1]
                    delta_kg = iteration_result.fuel_consumed_kg - prev_result.fuel_consumed_kg
                    delta_percent = (delta_kg / prev_result.fuel_consumed_kg) * 100.0 if prev_result.fuel_consumed_kg > 0 else 0.0
                    iteration_result.convergence_delta_percent = delta_percent
                    
                    # Physical anomaly detection
                    mass_reduction_percent = ((prev_result.initial_mass_kg - current_total_mass) / 
                                            prev_result.initial_mass_kg * 100.0)
                    
                    if mass_reduction_percent > 0.5 and delta_percent > 5.0:
                        print(f"\n{'='*80}")
                        print(f"[ANOMALY WARNING] Physical inconsistency detected:")
                        print(f"{'='*80}")
                        print(f"  Aircraft mass reduced by {mass_reduction_percent:.2f}%")
                        print(f"  But fuel consumption increased by {delta_percent:.2f}%")
                        print(f"  Possible causes: DP grid discretization artifacts")
                        print(f"{'='*80}\n")
                else:
                    iteration_result.convergence_delta_percent = float('inf')
                
                # Add to history
                history.add_iteration(iteration_result)
                
                # Print iteration summary
                print(f"[ITERATION {iteration_count} SUMMARY] Mission fuel breakdown:")
                print(f"  Climb: {iteration_result.climb_fuel_kg:.1f} kg, "
                      f"Cruise: {iteration_result.cruise_fuel_kg:.1f} kg, "
                      f"Descent: {iteration_result.descent_fuel_kg:.1f} kg")
                print(f"  TOTAL (ALL THREE PHASES): {iteration_result.fuel_consumed_kg:.1f} kg")
                
                if iteration_count > 1:
                    print(f"[CONVERGENCE] Delta: {iteration_result.convergence_delta_percent:.3f}% "
                          f"(target: < {CONVERGENCE_TOLERANCE_PERCENT:.3f}%)")
                
                # Check convergence
                if history.is_converged():
                    print(f"\n[CONVERGENCE ACHIEVED] After {iteration_count} iterations")
                    print(f"[FINAL] COMPLETE MISSION FUEL CONSUMPTION:")
                    print(f"  Climb fuel:   {iteration_result.climb_fuel_kg:.1f} kg")
                    print(f"  Cruise fuel:  {iteration_result.cruise_fuel_kg:.1f} kg") 
                    print(f"  Descent fuel: {iteration_result.descent_fuel_kg:.1f} kg")
                    print(f"  TOTAL: {iteration_result.fuel_consumed_kg:.1f} kg")
                    
                    optimized_fuel = iteration_result.fuel_consumed_kg * (1.0 + SAFETY_BUFFER_PERCENT)
                    print(f"[SAFETY] Applied {SAFETY_BUFFER_PERCENT*100:.0f}% buffer: {optimized_fuel:.1f} kg")
                    break
                
                # ========= AITKEN ACCELERATION =========
                if USE_AITKEN_ACCELERATION and len(history.iterations) >= 3:
                    # Full Aitken acceleration
                    f_consumed_k = history.iterations[-1].fuel_consumed_kg
                    f_consumed_k_minus_1 = history.iterations[-2].fuel_consumed_kg
                    f_consumed_k_minus_2 = history.iterations[-3].fuel_consumed_kg
                    
                    delta_f_k = f_consumed_k - f_consumed_k_minus_1
                    delta_f_k_minus_1 = f_consumed_k_minus_1 - f_consumed_k_minus_2
                    denominator = delta_f_k - delta_f_k_minus_1
                    
                    if abs(denominator) > 1e-6:
                        aitken_factor = 1.0 - (delta_f_k / denominator)
                        new_damping = current_damping * aitken_factor
                        new_damping = np.clip(new_damping, AITKEN_MIN_DAMPING, AITKEN_MAX_DAMPING)
                        
                        print(f"[AITKEN] Computed adaptive damping: {new_damping:.4f} "
                              f"(previous: {current_damping:.4f})")
                        print(f"[AITKEN] Δf_k = {delta_f_k:.2f} kg, Δf_k-1 = {delta_f_k_minus_1:.2f} kg")
                        
                        current_damping = new_damping
                    else:
                        print(f"[AITKEN] Denominator too small, using previous damping: {current_damping:.4f}")
                
                elif USE_AITKEN_ACCELERATION and len(history.iterations) == 2:
                    # Simplified adaptive damping for iteration 2
                    f_consumed_curr = history.iterations[-1].fuel_consumed_kg
                    f_consumed_prev = history.iterations[-2].fuel_consumed_kg
                    delta_f = f_consumed_curr - f_consumed_prev
                    delta_percent = abs(delta_f / f_consumed_prev) * 100.0 if f_consumed_prev > 0 else 0.0
                    
                    if delta_percent < 1.0:
                        new_damping = min(0.7, current_damping * 1.5)
                    elif delta_percent < 5.0:
                        new_damping = min(0.6, current_damping * 1.2)
                    elif delta_percent < 15.0:
                        new_damping = current_damping
                    else:
                        new_damping = max(0.2, current_damping * 0.7)
                    
                    new_damping = np.clip(new_damping, AITKEN_MIN_DAMPING, AITKEN_MAX_DAMPING)
                    
                    print(f"[ADAPTIVE-DAMP] Iteration 2: delta = {delta_percent:.2f}%")
                    print(f"[ADAPTIVE-DAMP] Adjusted damping: {current_damping:.4f} → {new_damping:.4f}")
                    
                    current_damping = new_damping
                else:
                    if USE_AITKEN_ACCELERATION:
                        print(f"[AITKEN] Insufficient history (iteration {len(history.iterations)}), "
                              f"using fixed damping: {current_damping:.4f}")
                
                # Apply relaxation update
                fuel_update = (current_damping * iteration_result.fuel_consumed_kg +
                              (1.0 - current_damping) * initial_fuel_current_kg)
                
                print(f"[UPDATE] Fuel for next iteration: {fuel_update:.1f} kg (damping: {current_damping:.4f})")
                print(f"[UPDATE] Change: {fuel_update - initial_fuel_current_kg:+.1f} kg "
                      f"({(fuel_update - initial_fuel_current_kg)/initial_fuel_current_kg*100:+.2f}%)")
                
                initial_fuel_current_kg = fuel_update
            
            # Check convergence status
            if iteration_count >= MAX_ITERATIONS and not history.is_converged():
                last_delta = history.iterations[-1].convergence_delta_percent if len(history.iterations) > 0 else float('inf')
                print(f"\n{'='*80}")
                print(f"[WARNING] Reached MAX_ITERATIONS ({MAX_ITERATIONS}) without full convergence")
                print(f"{'='*80}")
                print(f"Last convergence delta: {last_delta:.3f}% (tolerance: {CONVERGENCE_TOLERANCE_PERCENT:.3f}%)")
                print(f"Proceeding with best available result from iteration {iteration_count}")
                print(f"{'='*80}\n")
            
            # Return final result
            if len(history.iterations) == 0:
                raise RuntimeError("No successful iterations completed! Check mission configuration.")
            
            # Select iteration with minimum fuel consumption
            min_fuel_iteration = min(history.iterations, key=lambda x: x.fuel_consumed_kg)
            final_result = min_fuel_iteration
            
            if final_result.iteration != history.iterations[-1].iteration:
                print(f"\n{'='*80}")
                print(f"[OPTIMIZATION] Best result from iteration {final_result.iteration} "
                      f"(not last iteration {history.iterations[-1].iteration})")
                print(f"{'='*80}")
                print(f"Fuel savings by selecting best: "
                      f"{history.iterations[-1].fuel_consumed_kg - final_result.fuel_consumed_kg:.1f} kg")
                print(f"{'='*80}\n")
            
            print("\n" + "="*80)
            converged_status = "CONVERGED" if history.is_converged() else "PARTIALLY CONVERGED (Best Available)"
            print(f"OPTIMIZATION COMPLETE - {converged_status}")
            print("="*80)
            print(f"Total iterations: {iteration_count}")
            if not history.is_converged():
                print(f"Final convergence delta: {final_result.convergence_delta_percent:.3f}% "
                      f"(target: {CONVERGENCE_TOLERANCE_PERCENT:.3f}%)")
            print(f"Optimized fuel capacity: {final_result.fuel_consumed_kg * (1.0 + SAFETY_BUFFER_PERCENT):.1f} kg")
            print(f"Mission fuel consumption: {final_result.fuel_consumed_kg:.1f} kg")
            print(f"Safety buffer: {SAFETY_BUFFER_PERCENT*100:.0f}%")
            print("="*80 + "\n")
            
            return final_result, history
    
    # ========= CONFIGURATION MANAGER =========
    class ConfigurationManager:
        """Manages aircraft configuration updates with optimized fuel values."""
        
        @staticmethod
        def apply_optimized_fuel_to_configuration(optimized_fuel_kg: float) -> None:
            """
            Update the aircraft configuration with the optimized fuel capacity.
            
            This function modifies MAX_FUEL_KG in aircraft_config.py to reflect
            the optimized fuel capacity determined by the optimization process.
            
            Args:
                optimized_fuel_kg: The optimized fuel capacity in kg
                
            Notes:
                - Updates class attribute in SystemConfiguration
                - Preserves file structure and comments
                - Validates successful update before completing
            """
            import re
            
            print(f"\n[CONFIG UPDATE] Optimized fuel capacity: {optimized_fuel_kg:.1f} kg")
            print(f"[CONFIG UPDATE] Original MAX_FUEL_KG: {MAX_FUEL_KG:.1f} kg")
            print(f"[CONFIG UPDATE] Fuel savings: {MAX_FUEL_KG - optimized_fuel_kg:.1f} kg "
                  f"({(MAX_FUEL_KG - optimized_fuel_kg) / MAX_FUEL_KG * 100:.1f}%)")
            
            # Read aircraft_config.py
            config_file = "aircraft_config.py"
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find and replace MAX_FUEL_KG value
                pattern = r'(\s*MAX_FUEL_KG\s*=\s*)[\d]+\.?[\d]*(\s*#.*)'
                replacement = f'\\g<1>{optimized_fuel_kg:.1f}\\2'
                
                new_content = re.sub(pattern, replacement, content)
                
                if new_content == content:
                    raise RuntimeError("Failed to update MAX_FUEL_KG - pattern did not match")
                
                if f'{optimized_fuel_kg:.1f}' not in new_content:
                    raise RuntimeError(f"Updated value {optimized_fuel_kg:.1f} not found in new content")
                
                # Write back to file
                import os
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                    f.flush()
                    os.fsync(f.fileno())
                
                print(f"[CONFIG UPDATE] Successfully updated {config_file}")
                print(f"[CONFIG UPDATE] MAX_FUEL_KG is now set to {optimized_fuel_kg:.1f} kg")
                
            except Exception as e:
                print(f"[CONFIG UPDATE ERROR] Failed to update {config_file}: {e}")
                raise


# Create global fuel optimization core instance
_fuel_optimization_core = FuelOptimizationCore()


# =========  5 - BACKWARD COMPATIBILITY WRAPPERS =================
# Mission iteration function
def run_single_mission_iteration(
    initial_mass_kg: float,
    aero: PyAerodynamicsWrapper,
    eng: EngineWrapper,
    M_grid: np.ndarray,
    H_plot: np.ndarray,
    lever_samples: int,
    print_progress: bool = True
) -> MissionIterationResults:
    """Backward compatibility wrapper for FuelOptimizationCore.IterationExecutor.run_single_mission_iteration"""
    return FuelOptimizationCore.IterationExecutor.run_single_mission_iteration(
        initial_mass_kg, aero, eng, M_grid, H_plot, lever_samples, print_progress
    )

# Optimization function
def optimize_fuel_capacity(
    aero: PyAerodynamicsWrapper,
    eng: EngineWrapper,
    M_grid: np.ndarray,
    H_plot: np.ndarray,
    lever_samples: int = 50
) -> Tuple[MissionIterationResults, ConvergenceHistory]:
    """Backward compatibility wrapper for FuelOptimizationCore.ConvergenceController.optimize_fuel_capacity"""
    return FuelOptimizationCore.ConvergenceController.optimize_fuel_capacity(
        aero, eng, M_grid, H_plot, lever_samples
    )

# Configuration update function
def apply_optimized_fuel_to_configuration(optimized_fuel_kg: float) -> None:
    """Backward compatibility wrapper for FuelOptimizationCore.ConfigurationManager.apply_optimized_fuel_to_configuration"""
    return FuelOptimizationCore.ConfigurationManager.apply_optimized_fuel_to_configuration(optimized_fuel_kg)
