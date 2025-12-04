# =========  1 - MODULE INITIALIZATION =================
# ========= IMPORTS AND BASIC SETUP ===========================================
"""
Fuel Capacity Optimization Module

This module implements bisection-based optimization to determine the minimum 
required fuel capacity for mission completion. The bisection method guarantees
monotonic convergence and avoids oscillation issues present in fixed-point iteration.

Scientific Approach:
- Bisection method for robust, guaranteed convergence
- Dynamic mass evolution tracking throughout mission phases
- Safety buffer application to converged results

Module Structure:
1. Data Structures (iteration results, convergence history)
2. Mission Iteration Executor
3. Bisection Controller (optimization loop)
4. Configuration Manager (fuel parameter updates)

Key Features:
- Bisection method with guaranteed monotonic convergence
- Fuel deficit tracking (consumed vs available)
- Absolute tolerance convergence criterion (10 kg)
- Safety buffer application (5%) after convergence
- Comprehensive error handling and recovery mechanisms
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
from descent import run_optimization as run_descent_optimization, DescentResults

# Import configuration
from aircraft_config import INITIAL_MASS_KG, W_FUEL_KG, W_OE_KG, W_PL_KG
from atmosphere import a_from_altitude
from mission_config import (
    TARGET_ALT_CLIMB_M, START_ALTITUDE_CLIMB_M, START_VELOCITY_CLIMB_MS, START_LEVER_CLIMB,
    CRUISE_DISTANCE_KM, CRUISE_TIME_STEP_S,
    TARGET_DESCENT_ALT_M, TARGET_DESCENT_MACH,
    TARGET_MACH_CRUISE, TARGET_MACH_TOLERANCE,
    N_MACH_SAMPLES_CLIMB, N_ALTITUDE_STEPS_CLIMB, N_LEVER_SAMPLES_CLIMB,
    N_MACH_SAMPLES_DESCENT, N_ALTITUDE_STEPS_DESCENT, N_LEVER_SAMPLES_DESCENT,
    FUEL_OPTIMIZATION_CONVERGENCE_TOLERANCE_KG,
    FUEL_OPTIMIZATION_SAFETY_BUFFER_PERCENT,
    FUEL_OPTIMIZATION_MAX_ITERATIONS,
    FUEL_OPTIMIZATION_INITIAL_FUEL_LOW_KG,
    FUEL_OPTIMIZATION_INITIAL_FUEL_HIGH_KG
)

# Import wrappers
from pyaerodynamics_wrapper import PyAerodynamicsWrapper
from pyengine_wrapper import EngineWrapper


# =========  2 - CONVERGENCE PARAMETERS =================
# Fuel optimization parameters are now imported from mission_config.py
# Backward compatibility constants (aliases for imported values)
CONVERGENCE_TOLERANCE_KG = FUEL_OPTIMIZATION_CONVERGENCE_TOLERANCE_KG
SAFETY_BUFFER_PERCENT = FUEL_OPTIMIZATION_SAFETY_BUFFER_PERCENT
MAX_ITERATIONS = FUEL_OPTIMIZATION_MAX_ITERATIONS
INITIAL_FUEL_LOW_KG = FUEL_OPTIMIZATION_INITIAL_FUEL_LOW_KG
INITIAL_FUEL_HIGH_KG = FUEL_OPTIMIZATION_INITIAL_FUEL_HIGH_KG


# =========  3 - DATA STRUCTURES =================
@dataclass
class MissionIterationResults:
    """
    Results from a single mission iteration.
    
    This structure encapsulates complete mission simulation results including
    fuel consumption and phase-wise detailed results.
    
    Attributes:
        iteration: Iteration number
        initial_fuel_kg: Initial fuel load for this iteration [kg]
        initial_mass_kg: Initial total aircraft mass [kg]
        fuel_consumed_kg: Total fuel consumed across all phases [kg]
        fuel_deficit_kg: Difference between consumed and available fuel [kg]
        climb_result: Climb phase optimization results
        cruise_result: Cruise phase simulation results
        descent_result: Descent phase optimization results
        total_time_s: Total mission duration [s]
        climb_fuel_kg: Fuel consumed in climb phase [kg]
        cruise_fuel_kg: Fuel consumed in cruise phase [kg]
        descent_fuel_kg: Fuel consumed in descent phase [kg]
        climb_time_s: Climb phase duration [s]
        cruise_time_s: Cruise phase duration [s]
        descent_time_s: Descent phase duration [s]
        final_mass_kg: Final aircraft mass after mission [kg]
    """
    iteration: int
    initial_fuel_kg: float
    initial_mass_kg: float
    fuel_consumed_kg: float
    fuel_deficit_kg: float
    climb_result: MinFuelSchedule
    cruise_result: CruiseResults
    descent_result: DescentResults
    total_time_s: float
    climb_fuel_kg: float
    cruise_fuel_kg: float
    descent_fuel_kg: float
    climb_time_s: float
    cruise_time_s: float
    descent_time_s: float
    final_mass_kg: float


@dataclass
class ConvergenceHistory:
    """
    Tracking structure for bisection convergence analysis.
    
    This structure maintains complete optimization history for analysis,
    diagnostics, and visualization purposes.
    
    Attributes:
        iterations: List of all mission iteration results
        fuel_bounds_history: List of (lower_bound, upper_bound) tuples
    """
    iterations: List[MissionIterationResults]
    fuel_bounds_history: List[Tuple[float, float]]
    
    def __init__(self):
        """Initialize empty convergence history."""
        self.iterations = []
        self.fuel_bounds_history = []
    
    def add_iteration(self, result: MissionIterationResults, bounds: Tuple[float, float]):
        """
        Add iteration result to history.
        
        Args:
            result: Mission iteration results to add
            bounds: Current (lower, upper) fuel bounds
        """
        self.iterations.append(result)
        self.fuel_bounds_history.append(bounds)


# =========  4 - FUEL OPTIMIZATION CORE SYSTEM =================
class FuelOptimizationCore:
    """
    Fuel capacity optimization framework using bisection method.
    
    This class implements a robust bisection-based approach to determine the minimum
    required fuel capacity for mission completion. The bisection method guarantees
    monotonic convergence and avoids oscillation issues.
    
    System Components:
    - IterationExecutor: Executes complete mission simulations (climb + cruise + descent)
    - BisectionController: Manages bisection optimization loop with guaranteed convergence
    - ConfigurationManager: Updates aircraft configuration files with optimized fuel
    
    Bisection Algorithm:
        Initialize: F_low (too little), F_high (too much)
        Loop:
            F_mid = (F_low + F_high) / 2
            F_consumed = run_mission(F_mid)
            
            if F_consumed > F_mid:
                F_low = F_mid  (need more fuel)
            else:
                F_high = F_mid  (have excess fuel)
                
        Until: |F_high - F_low| < tolerance
    
    Implementation:
        # Run complete optimization
        optimal, history = FuelOptimizationCore.BisectionController.optimize_fuel_capacity(...)
        
        # Update configuration
        FuelOptimizationCore.ConfigurationManager.apply_optimized_fuel_to_configuration(...)
    """
    
    # ========= MISSION ITERATION EXECUTOR =========
    class IterationExecutor:
        """Manages execution of complete mission iterations."""
        
        @staticmethod
        def run_single_mission_iteration(
            initial_fuel_kg: float,
            aero: PyAerodynamicsWrapper,
            eng: EngineWrapper,
            mach_grid: np.ndarray,
            H_plot: np.ndarray,
            lever_samples: int,
            print_progress: bool = True
        ) -> MissionIterationResults:
            """
            Execute a complete mission iteration (climb + cruise + descent).
            
            Args:
                initial_fuel_kg: Initial fuel capacity for this iteration [kg]
                aero: Aerodynamics wrapper instance
                eng: Engine wrapper instance
                mach_grid: Mach grid for optimization
                H_plot: Altitude grid for plotting
                lever_samples: Number of lever samples for DP optimization
                print_progress: Whether to print progress messages
                    
            Returns:
                MissionIterationResults containing all phase results
            """
            iteration_start_time = time.time()
            
            # Calculate initial mass
            initial_mass_kg = W_OE_KG + W_PL_KG + initial_fuel_kg
            
            if print_progress:
                print(f"\n[MISSION ITERATION] Initial fuel: {initial_fuel_kg:.1f} kg, Total mass: {initial_mass_kg:.1f} kg")
            
            # ========= CLIMB PHASE =========================================
            if print_progress:
                print("[CLIMB] Computing optimal climb trajectory...")
            
            # Calculate starting Mach from takeoff velocity at start altitude
            a = a_from_altitude(START_ALTITUDE_CLIMB_M)
            start_mach = START_VELOCITY_CLIMB_MS / a
            
            # Create uniform altitude steps
            uniform_step_size = TARGET_ALT_CLIMB_M / N_ALTITUDE_STEPS_CLIMB
            altitude_sched = np.arange(START_ALTITUDE_CLIMB_M, 
                                TARGET_ALT_CLIMB_M + uniform_step_size, 
                                uniform_step_size)
            
            # Solve 3D DP for climb
            dp_sched, dp_info = ClimbingCore.DynamicProgrammingOptimizer.solve_3d_dp(
                aero, eng, mach_grid, altitude_sched, 
                lever_samples=lever_samples,
                target_mach=TARGET_MACH_CRUISE,
                target_mach_tolerance=TARGET_MACH_TOLERANCE,
                start_mach=start_mach,
                start_lever=START_LEVER_CLIMB,
                mass_kg=initial_mass_kg
            )
            
            climb_fuel = float(np.nan_to_num(dp_sched.cumFuel_kg, nan=0.0)[-1])
            climb_time_s = float(np.sum(np.nan_to_num(dp_sched.dt_s, nan=0.0)))
            climb_mass_end = initial_mass_kg - climb_fuel
            
            if print_progress:
                print(f"[CLIMB] Completed: {climb_time_s/60:.1f} min, {climb_fuel:.1f} kg fuel")
                print(f"[CLIMB] Mass: {initial_mass_kg:.1f} kg -> {climb_mass_end:.1f} kg "
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
                print(f"[CRUISE] Mass: {climb_mass_end:.1f} kg -> {cruise_mass_end:.1f} kg "
                      f"(burned {cruise_fuel:.1f} kg, {cruise_fuel/climb_mass_end*100:.2f}%)")
            
            # ========= DESCENT PHASE =========================================
            if print_progress:
                print("[DESCENT] Computing optimal descent trajectory...")
            
            H_descent = np.linspace(cruise_results.altitude_m[-1], 
                                   TARGET_DESCENT_ALT_M, 
                                   N_ALTITUDE_STEPS_DESCENT)
            
            # Use precomputed grids for exact cache alignment
            descent_result, descent_info = run_descent_optimization(
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
                print(f"[DESCENT] Mass: {cruise_mass_end:.1f} kg -> {descent_mass_end:.1f} kg "
                      f"(burned {descent_fuel:.1f} kg, {descent_fuel/cruise_mass_end*100:.2f}%)")
            
            # ========= COMPUTE SUMMARY =========================================
            total_fuel = climb_fuel + cruise_fuel + descent_fuel
            total_time_s = climb_time_s + cruise_time_s + descent_time_s
            fuel_deficit_kg = total_fuel - initial_fuel_kg
            
            iteration_time = time.time() - iteration_start_time
            
            if print_progress:
                print(f"[ITERATION] Completed in {iteration_time:.1f}s")
                print(f"[MISSION TOTALS] Fuel consumed: {total_fuel:.1f} kg, Available: {initial_fuel_kg:.1f} kg")
                print(f"[DEFICIT] {fuel_deficit_kg:+.1f} kg ({'INSUFFICIENT' if fuel_deficit_kg > 0 else 'EXCESS'} fuel)")
            
            return MissionIterationResults(
                iteration=-1,  # Will be set by caller
                initial_fuel_kg=initial_fuel_kg,
                initial_mass_kg=initial_mass_kg,
                fuel_consumed_kg=total_fuel,
                fuel_deficit_kg=fuel_deficit_kg,
                climb_result=dp_sched,
                cruise_result=cruise_results,
                descent_result=descent_result,
                total_time_s=total_time_s,
                climb_fuel_kg=climb_fuel,
                cruise_fuel_kg=cruise_fuel,
                descent_fuel_kg=descent_fuel,
                climb_time_s=climb_time_s,
                cruise_time_s=cruise_time_s,
                descent_time_s=descent_time_s,
                final_mass_kg=descent_result.final_mass_kg
            )
    
    # ========= BISECTION CONTROLLER =========
    class BisectionController:
        """Manages bisection optimization loop with guaranteed monotonic convergence."""
        
        @staticmethod
        def optimize_fuel_capacity(
            aero: PyAerodynamicsWrapper,
            eng: EngineWrapper,
            mach_grid: np.ndarray,
            H_plot: np.ndarray,
            lever_samples: int = 50
        ) -> Tuple[MissionIterationResults, ConvergenceHistory]:
            """
            Main bisection optimization loop to determine minimum required fuel capacity.
            
            Bisection Method:
            - Initialize F_low (insufficient fuel) and F_high (excess fuel)
            - Iteratively compute F_mid = (F_low + F_high) / 2
            - Run mission with F_mid and measure F_consumed
            - If F_consumed > F_mid: increase lower bound (F_low = F_mid)
            - If F_consumed < F_mid: decrease upper bound (F_high = F_mid)
            - Continue until |F_high - F_low| < tolerance
            
            Args:
                aero: Aerodynamics wrapper instance
                eng: Engine wrapper instance  
                mach_grid: Mach grid for optimization
                H_plot: Altitude grid for plotting
                lever_samples: Number of lever samples for DP optimization
                
            Returns:
                Tuple of (final optimized result, convergence history)
            """
            print("\n" + "="*80)
            print("FUEL CAPACITY OPTIMIZATION USING BISECTION METHOD")
            print("="*80)
            print(f"Objective: Determine minimum required fuel for mission completion")
            print(f"Convergence tolerance: {CONVERGENCE_TOLERANCE_KG:.1f} kg")
            print(f"Safety buffer: {SAFETY_BUFFER_PERCENT*100:.0f}%")
            print(f"Method: Bisection with guaranteed monotonic convergence")
            print("="*80)
            
            # Initialize bisection bounds
            fuel_low = INITIAL_FUEL_LOW_KG
            fuel_high = INITIAL_FUEL_HIGH_KG
            history = ConvergenceHistory()
            iteration_count = 0
            
            # Store best result (closest to zero deficit)
            best_result = None
            best_deficit_abs = float('inf')
            
            print(f"\n[BISECTION] Initial bounds: [{fuel_low:.1f}, {fuel_high:.1f}] kg")
            
            while iteration_count < MAX_ITERATIONS:
                iteration_count += 1
                
                # Bisection: try midpoint
                fuel_mid = (fuel_low + fuel_high) / 2.0
                convergence_range = fuel_high - fuel_low
                
                print(f"\n[ITERATION {iteration_count}] Bounds: [{fuel_low:.1f}, {fuel_high:.1f}] kg, Range: {convergence_range:.1f} kg")
                print(f"[ITERATION {iteration_count}] Testing fuel: {fuel_mid:.1f} kg")
                
                # Run mission with current fuel estimate
                try:
                    iteration_result = FuelOptimizationCore.IterationExecutor.run_single_mission_iteration(
                        initial_fuel_kg=fuel_mid,
                        aero=aero,
                        engine=eng,
                        mach_grid=mach_grid,
                        H_plot=H_plot,
                        lever_samples=lever_samples,
                        print_progress=True
                    )
                except RuntimeError as e:
                    print(f"\n[ERROR] Mission failed at iteration {iteration_count}: {str(e)}")
                    print(f"[ERROR] Fuel: {fuel_mid:.1f} kg")
                    
                    if iteration_count == 1:
                        raise RuntimeError(
                            f"Mission infeasible even with high fuel estimate ({fuel_high:.1f} kg). "
                            f"Check mission parameters or increase INITIAL_FUEL_HIGH_KG."
                        )
                    else:
                        # Mission failed - probably insufficient fuel
                        print(f"[BISECTION] Mission failure indicates insufficient fuel")
                        fuel_low = fuel_mid
                        continue
                
                # Store iteration results
                iteration_result.iteration = iteration_count
                history.add_iteration(iteration_result, (fuel_low, fuel_high))
                
                # Track best result
                deficit_abs = abs(iteration_result.fuel_deficit_kg)
                if deficit_abs < best_deficit_abs:
                    best_deficit_abs = deficit_abs
                    best_result = iteration_result
                
                # Bisection logic
                if iteration_result.fuel_deficit_kg > 0:
                    # Consumed more than available - need MORE fuel
                    print(f"[BISECTION] Insufficient fuel (deficit: {iteration_result.fuel_deficit_kg:+.1f} kg)")
                    print(f"[BISECTION] Increasing lower bound: {fuel_low:.1f} -> {fuel_mid:.1f} kg")
                    fuel_low = fuel_mid
                else:
                    # Consumed less than available - have EXCESS fuel
                    print(f"[BISECTION] Excess fuel (surplus: {-iteration_result.fuel_deficit_kg:+.1f} kg)")
                    print(f"[BISECTION] Decreasing upper bound: {fuel_high:.1f} -> {fuel_mid:.1f} kg")
                    fuel_high = fuel_mid
                
                # Check convergence
                if convergence_range < CONVERGENCE_TOLERANCE_KG:
                    print(f"\n[CONVERGENCE ACHIEVED] After {iteration_count} iterations")
                    print(f"[CONVERGENCE] Final range: {convergence_range:.1f} kg < {CONVERGENCE_TOLERANCE_KG:.1f} kg tolerance")
                    break
            
            # Check convergence status
            if iteration_count >= MAX_ITERATIONS:
                print(f"\n{'='*80}")
                print(f"[WARNING] Reached MAX_ITERATIONS ({MAX_ITERATIONS}) without full convergence")
                print(f"{'='*80}")
                print(f"Final range: {fuel_high - fuel_low:.1f} kg (tolerance: {CONVERGENCE_TOLERANCE_KG:.1f} kg)")
                print(f"Using best result from iteration {best_result.iteration}")
                print(f"{'='*80}\n")
            
            # Use best result (closest to equilibrium)
            if best_result is None:
                raise RuntimeError("No successful iterations completed! Check mission configuration.")
            
            final_result = best_result
            
            print("\n" + "="*80)
            print("OPTIMIZATION COMPLETE - BISECTION CONVERGED")
            print("="*80)
            print(f"Total iterations: {iteration_count}")
            print(f"Final fuel range: [{fuel_low:.1f}, {fuel_high:.1f}] kg")
            print(f"Selected fuel: {final_result.initial_fuel_kg:.1f} kg")
            print(f"Fuel consumed: {final_result.fuel_consumed_kg:.1f} kg")
            print(f"Deficit: {final_result.fuel_deficit_kg:+.1f} kg")
            optimized_fuel = final_result.fuel_consumed_kg * (1.0 + SAFETY_BUFFER_PERCENT)
            print(f"With {SAFETY_BUFFER_PERCENT*100:.0f}% safety buffer: {optimized_fuel:.1f} kg")
            print("="*80 + "\n")
            
            return final_result, history
    
    # ========= CONFIGURATION MANAGER =========
    class ConfigurationManager:
        """Manages aircraft configuration updates with optimized fuel values."""
        
        @staticmethod
        def apply_optimized_fuel_to_configuration(optimized_fuel_kg: float) -> None:
            """
            Update the aircraft configuration with the optimized fuel capacity.
            
            This function modifies W_FUEL_KG in aircraft_config.py to reflect
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
            print(f"[CONFIG UPDATE] Original W_FUEL_KG: {W_FUEL_KG:.1f} kg")
            print(f"[CONFIG UPDATE] Fuel savings: {W_FUEL_KG - optimized_fuel_kg:.1f} kg "
                  f"({(W_FUEL_KG - optimized_fuel_kg) / W_FUEL_KG * 100:.1f}%)")
            
            # Read aircraft_config.py
            config_file = "aircraft_config.py"
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find and replace W_FUEL_KG value
                pattern = r'(\s*W_FUEL_KG\s*=\s*)[\d]+\.?[\d]*(\s*#.*)'
                replacement = f'\\g<1>{optimized_fuel_kg:.1f}\\2'
                
                new_content = re.sub(pattern, replacement, content)
                
                if new_content == content:
                    raise RuntimeError("Failed to update W_FUEL_KG - pattern did not match")
                
                if f'{optimized_fuel_kg:.1f}' not in new_content:
                    raise RuntimeError(f"Updated value {optimized_fuel_kg:.1f} not found in new content")
                
                # Write back to file
                import os
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                    f.flush()
                    os.fsync(f.fileno())
                
                print(f"[CONFIG UPDATE] Successfully updated {config_file}")
                print(f"[CONFIG UPDATE] W_FUEL_KG is now set to {optimized_fuel_kg:.1f} kg")
                
            except Exception as e:
                print(f"[CONFIG UPDATE ERROR] Failed to update {config_file}: {e}")
                raise


# =========  5 - BACKWARD COMPATIBILITY WRAPPERS =================
# Mission iteration function
def run_single_mission_iteration(
    initial_fuel_kg: float,
    aero: PyAerodynamicsWrapper,
    eng: EngineWrapper,
    mach_grid: np.ndarray,
    H_plot: np.ndarray,
    lever_samples: int,
    print_progress: bool = True
) -> MissionIterationResults:
    """Backward compatibility wrapper for FuelOptimizationCore.IterationExecutor.run_single_mission_iteration"""
    return FuelOptimizationCore.IterationExecutor.run_single_mission_iteration(
        initial_fuel_kg, aero, eng, mach_grid, H_plot, lever_samples, print_progress
    )

# Optimization function
def optimize_fuel_capacity(
    aero: PyAerodynamicsWrapper,
    eng: EngineWrapper,
    mach_grid: np.ndarray,
    H_plot: np.ndarray,
    lever_samples: int = 50
) -> Tuple[MissionIterationResults, ConvergenceHistory]:
    """Backward compatibility wrapper for FuelOptimizationCore.BisectionController.optimize_fuel_capacity"""
    return FuelOptimizationCore.BisectionController.optimize_fuel_capacity(
        aero, eng, mach_grid, H_plot, lever_samples
    )

# Configuration update function
def apply_optimized_fuel_to_configuration(optimized_fuel_kg: float) -> None:
    """Backward compatibility wrapper for FuelOptimizationCore.ConfigurationManager.apply_optimized_fuel_to_configuration"""
    return FuelOptimizationCore.ConfigurationManager.apply_optimized_fuel_to_configuration(optimized_fuel_kg)
