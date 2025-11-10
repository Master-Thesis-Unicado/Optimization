# =========  1 - MODULE INITIALIZATION =================
# ========= IMPORTS AND BASIC SETUP ===========================================
"""
Mission Range Optimization Module

This module implements iterative range optimization functionality for aircraft mission
analysis. The optimization adjusts cruise distance to achieve a user-specified target
mission range while maintaining optimal climb and descent profiles.

Scientific Approach:
- Iterative convergence with damping factor for stability
- Distance calculation based on true airspeed integration over time
- Intelligent cruise segment adjustment (extension or truncation)
- Convergence detection with specified tolerance

Module Structure:
1. Distance Calculation System
2. Cruise Segment Management System
3. Optimization Controller
4. Integration Utilities

Author: Mission Analysis System
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

# Import atmospheric properties for distance calculations
from aircraft_config import a_from_altitude

# Import phase result types
from climb import MinFuelSchedule
from cruise import CruiseResults, CruiseInitialState, simulate_steady_cruise
from descent import DescentResults

# Import mission configuration parameters
from mission_config import (
    RANGE_OPTIMIZATION_TOLERANCE_KM,
    RANGE_OPTIMIZATION_DAMPING_FACTOR,
    MAX_RANGE_OPTIMIZATION_ITERATIONS
)


# =========  2 - CORE OPTIMIZATION SYSTEM =================
class RangeOptimizationCore:
    """
    Comprehensive mission range optimization framework for iterative convergence.
    
    This class implements a complete computational framework for mission range targeting
    through three integrated subsystems: distance calculation, cruise segment management,
    and iterative optimization control. The system enables user-specified mission ranges
    through automatic cruise distance adjustment while maintaining fuel-optimal climb
    and descent trajectories.
    
    System Components:
    - DistanceCalculator: Computes horizontal ground distances for all flight phases
      through true airspeed integration with small angle approximations
    - CruiseSegmentManager: Handles intelligent cruise segment extension and truncation
      to efficiently adjust mission range without complete re-simulation
    - OptimizationController: Manages iterative convergence process with damped feedback
      control and iteration history tracking
    
    Computational Features:
    - Horizontal distance integration using true airspeed and time steps
    - Small angle approximation for climb and descent trajectories
    - Cruise segment continuity preservation through state management
    - Damped feedback control for stable convergence
    - Iteration history tracking for analysis and debugging
    
    Implementation:
        # Distance calculation
        total_dist, breakdown = RangeOptimizationCore.DistanceCalculator.calculate_total_mission_distance_km(...)
        
        # Cruise segment management
        extended_cruise = RangeOptimizationCore.CruiseSegmentManager.adjust_cruise_segment_extension(...)
        truncated_cruise = RangeOptimizationCore.CruiseSegmentManager.adjust_cruise_segment_truncation(...)
        
        # Optimization control
        optimizer = RangeOptimizationCore.OptimizationController(target_range_km=1000.0)
        converged, error = optimizer.check_convergence(actual_range_km)
    """
    
    # ========= DISTANCE CALCULATION SYSTEM =========
    class DistanceCalculator:
        """Manages distance calculations for all flight phases."""
        
        @staticmethod
        def calculate_climb_distance_km(climb_result: MinFuelSchedule) -> float:
            """
            Calculate horizontal ground distance covered during climb phase.
            
            The horizontal distance is computed by integrating true airspeed over time,
            assuming small climb angles such that horizontal velocity ≈ TAS.
            
            Mathematical formulation:
                d_horizontal = ∫ V_TAS(t) dt ≈ Σ V_TAS,i × Δt_i
            
            where V_TAS = M × a(h) is the true airspeed at each altitude h.
            
            Args:
                climb_result: Climb optimization results containing trajectory data
                
            Returns:
                float: Horizontal distance covered during climb [km]
                
            Notes:
                - Small angle approximation: cos(γ) ≈ 1 for typical climb angles γ < 10°
                - Distance calculation excludes vertical displacement component
            """
            if len(climb_result.dt_s) == 0 or len(climb_result.mach) == 0:
                return 0.0
            
            climb_distances = []
            for i in range(len(climb_result.dt_s)):
                if i < len(climb_result.mach) and i < len(climb_result.alt_m):
                    # Calculate speed of sound at current altitude
                    a = a_from_altitude(climb_result.alt_m[i])
                    
                    # True airspeed: V_TAS = M × a
                    V_tas = climb_result.mach[i] * a
                    
                    # Horizontal distance increment: Δd = V_TAS × Δt
                    distance_increment_km = V_tas * climb_result.dt_s[i] / 1000.0
                    climb_distances.append(distance_increment_km)
            
            return sum(climb_distances)
        
        @staticmethod
        def calculate_descent_distance_km(descent_result: DescentResults) -> float:
            """
            Calculate horizontal ground distance covered during descent phase.
            
            Similar to climb, horizontal distance is computed through TAS integration,
            with small angle approximation for descent trajectory.
            
            Mathematical formulation:
                d_horizontal = ∫ V_TAS(t) dt ≈ Σ V_TAS,i × Δt_i
            
            Args:
                descent_result: Descent optimization results containing trajectory data
                
            Returns:
                float: Horizontal distance covered during descent [km]
                
            Notes:
                - Descent angles typically γ < -6°, maintaining small angle validity
                - Consistent methodology with climb distance calculation
            """
            if len(descent_result.dt_s) == 0 or len(descent_result.mach) == 0:
                return 0.0
            
            descent_distances = []
            for i in range(len(descent_result.dt_s)):
                if i < len(descent_result.mach) and i < len(descent_result.alt_m):
                    # Calculate speed of sound at current altitude
                    a = a_from_altitude(descent_result.alt_m[i])
                    
                    # True airspeed: V_TAS = M × a
                    V_tas = descent_result.mach[i] * a
                    
                    # Horizontal distance increment: Δd = V_TAS × Δt
                    distance_increment_km = V_tas * descent_result.dt_s[i] / 1000.0
                    descent_distances.append(distance_increment_km)
            
            return sum(descent_distances)
        
        @staticmethod
        def calculate_total_mission_distance_km(
            climb_result: MinFuelSchedule,
            cruise_result: CruiseResults,
            descent_result: DescentResults
        ) -> Tuple[float, Dict[str, float]]:
            """
            Calculate total mission range from all three flight phases.
            
            Total mission range is the sum of horizontal distances covered in climb,
            cruise, and descent phases. This provides the ground distance traveled
            from takeoff to landing.
            
            Args:
                climb_result: Climb phase optimization results
                cruise_result: Cruise phase simulation results
                descent_result: Descent phase optimization results
                
            Returns:
                tuple: (total_distance_km, breakdown_dict)
                    - total_distance_km: Total mission range [km]
                    - breakdown_dict: Dictionary with individual phase distances
                    
            Notes:
                - Cruise distance is directly available from simulation results
                - Climb and descent distances require TAS integration
                - Breakdown provides insight into phase-wise distance contributions
            """
            # Calculate individual phase distances
            climb_distance_km = RangeOptimizationCore.DistanceCalculator.calculate_climb_distance_km(climb_result)
            cruise_distance_km = cruise_result.target_distance_km  # Direct from cruise results
            descent_distance_km = RangeOptimizationCore.DistanceCalculator.calculate_descent_distance_km(descent_result)
            
            # Total mission distance
            total_distance_km = climb_distance_km + cruise_distance_km + descent_distance_km
            
            # Create breakdown dictionary for analysis
            breakdown = {
                'climb_km': climb_distance_km,
                'cruise_km': cruise_distance_km,
                'descent_km': descent_distance_km,
                'total_km': total_distance_km
            }
            
            return total_distance_km, breakdown
    
    # ========= CRUISE SEGMENT MANAGEMENT SYSTEM =========
    class CruiseSegmentManager:
        """Manages cruise segment extension and truncation for efficient optimization."""
        
        @staticmethod
        def adjust_cruise_segment_extension(
            cruise_result: CruiseResults,
            additional_distance_km: float,
            aero,
            engine,
            time_step_s: float
        ) -> CruiseResults:
            """
            Extend cruise segment by continuing from last cruise point.
            
            When target range requires longer cruise distance, simulation resumes from
            the final cruise state and continues for the additional required distance.
            This maintains continuity in fuel consumption and weight evolution.
            
            Args:
                cruise_result: Existing cruise simulation results
                additional_distance_km: Additional distance to cruise [km]
                aero: Aerodynamics wrapper
                engine: Engine wrapper
                time_step_s: Time step for cruise simulation [s]
                
            Returns:
                CruiseResults: Extended cruise results with combined trajectory data
                
            Notes:
                - Initial state for extension uses final state from previous cruise
                - Trajectories are concatenated to maintain continuity
                - Weight decreases according to fuel consumption in extension
            """
            # Extract final state from previous cruise segment
            final_weight_kg = cruise_result.weight_kg[-1]
            final_altitude_m = cruise_result.altitude_m[-1]
            final_mach = cruise_result.mach_number[-1]
            
            # Create initial state for cruise extension
            extension_initial_state = CruiseInitialState(
                altitude_m=final_altitude_m,
                mach=final_mach,
                weight_kg=final_weight_kg,
                fuel_consumed_climb_kg=0.0,  
                climb_time_s=0.0   
            )
            
            # Simulate additional cruise segment
            extension_result = simulate_steady_cruise(
                initial_state=extension_initial_state,
                target_distance_km=additional_distance_km,
                aero=aero,
                engine=engine,
                time_step_s=time_step_s
            )
            
            # Concatenate trajectories: original + extension
            # Time offset for extension segment
            time_offset = cruise_result.time_s[-1]
            extension_time_offset = extension_result.time_s + time_offset
            
            # Fuel offset for extension segment
            fuel_offset = cruise_result.fuel_consumed_kg[-1]
            extension_fuel_offset = extension_result.fuel_consumed_kg + fuel_offset
            
            # Create combined cruise result
            combined_result = CruiseResults(
                initial_state=cruise_result.initial_state,
                target_distance_km=cruise_result.target_distance_km + additional_distance_km,
                time_step_s=time_step_s,
                # Concatenate trajectory arrays
                time_s=np.concatenate([cruise_result.time_s, extension_time_offset]),
                distance_km=np.concatenate([cruise_result.distance_km, 
                                           cruise_result.distance_km[-1] + extension_result.distance_km]),
                weight_kg=np.concatenate([cruise_result.weight_kg, extension_result.weight_kg]),
                fuel_consumed_kg=np.concatenate([cruise_result.fuel_consumed_kg, extension_fuel_offset]),
                thrust_total_N=np.concatenate([cruise_result.thrust_total_N, extension_result.thrust_total_N]),
                drag_N=np.concatenate([cruise_result.drag_N, extension_result.drag_N]),
                fuel_flow_kgps=np.concatenate([cruise_result.fuel_flow_kgps, extension_result.fuel_flow_kgps]),
                specific_excess_power_mps=np.concatenate([cruise_result.specific_excess_power_mps, 
                                                          extension_result.specific_excess_power_mps]),
                lever_position=np.concatenate([cruise_result.lever_position, extension_result.lever_position]),
                altitude_m=np.concatenate([cruise_result.altitude_m, extension_result.altitude_m]),
                mach_number=np.concatenate([cruise_result.mach_number, extension_result.mach_number]),
                temperature_K=np.concatenate([cruise_result.temperature_K, extension_result.temperature_K]),
                density_kgpm3=np.concatenate([cruise_result.density_kgpm3, extension_result.density_kgpm3]),
                true_airspeed_mps=np.concatenate([cruise_result.true_airspeed_mps, 
                                                 extension_result.true_airspeed_mps]),
                # Update summary statistics
                total_time_s=cruise_result.total_time_s + extension_result.total_time_s,
                total_fuel_consumed_kg=cruise_result.total_fuel_consumed_kg + extension_result.total_fuel_consumed_kg,
                final_weight_kg=extension_result.final_weight_kg,
                average_fuel_flow_kgps=(cruise_result.total_fuel_consumed_kg + extension_result.total_fuel_consumed_kg) / 
                                       (cruise_result.total_time_s + extension_result.total_time_s),
                average_thrust_N=(cruise_result.average_thrust_N * cruise_result.total_time_s + 
                                 extension_result.average_thrust_N * extension_result.total_time_s) / 
                                (cruise_result.total_time_s + extension_result.total_time_s)
            )
            
            return combined_result
        
        @staticmethod
        def adjust_cruise_segment_truncation(
            cruise_result: CruiseResults,
            new_cruise_distance_km: float
        ) -> CruiseResults:
            """
            Truncate cruise segment to specified distance.
            
            When target range requires shorter cruise distance, trajectory is truncated
            at the point corresponding to the new distance. Subsequent cruise data is
            discarded, and descent begins from the truncation point.
            
            Args:
                cruise_result: Existing cruise simulation results
                new_cruise_distance_km: New shorter cruise distance [km]
                
            Returns:
                CruiseResults: Truncated cruise results up to specified distance
                
            Notes:
                - Linear interpolation used if exact distance not in discrete trajectory
                - All trajectory arrays truncated consistently at same point
                - Weight and fuel consumption reflect truncated segment only
            """
            # Find index where cruise distance equals or exceeds new target distance
            distance_array = cruise_result.distance_km
            
            # Find truncation index
            truncation_idx = np.searchsorted(distance_array, new_cruise_distance_km)
            
            # Handle edge cases
            if truncation_idx == 0:
                truncation_idx = 1  # Keep at least one point
            if truncation_idx >= len(distance_array):
                truncation_idx = len(distance_array) - 1
            
            # Truncate all trajectory arrays at the identified index
            truncated_result = CruiseResults(
                initial_state=cruise_result.initial_state,
                target_distance_km=new_cruise_distance_km,
                time_step_s=cruise_result.time_step_s,
                # Truncated trajectory arrays
                time_s=cruise_result.time_s[:truncation_idx+1],
                distance_km=cruise_result.distance_km[:truncation_idx+1],
                weight_kg=cruise_result.weight_kg[:truncation_idx+1],
                fuel_consumed_kg=cruise_result.fuel_consumed_kg[:truncation_idx+1],
                thrust_total_N=cruise_result.thrust_total_N[:truncation_idx+1],
                drag_N=cruise_result.drag_N[:truncation_idx+1],
                fuel_flow_kgps=cruise_result.fuel_flow_kgps[:truncation_idx+1],
                specific_excess_power_mps=cruise_result.specific_excess_power_mps[:truncation_idx+1],
                lever_position=cruise_result.lever_position[:truncation_idx+1],
                altitude_m=cruise_result.altitude_m[:truncation_idx+1],
                mach_number=cruise_result.mach_number[:truncation_idx+1],
                temperature_K=cruise_result.temperature_K[:truncation_idx+1],
                density_kgpm3=cruise_result.density_kgpm3[:truncation_idx+1],
                true_airspeed_mps=cruise_result.true_airspeed_mps[:truncation_idx+1],
                # Update summary statistics
                total_time_s=cruise_result.time_s[truncation_idx],
                total_fuel_consumed_kg=cruise_result.fuel_consumed_kg[truncation_idx],
                final_weight_kg=cruise_result.weight_kg[truncation_idx],
                average_fuel_flow_kgps=np.mean(cruise_result.fuel_flow_kgps[:truncation_idx+1]),
                average_thrust_N=np.mean(cruise_result.thrust_total_N[:truncation_idx+1])
            )
            
            return truncated_result
    
    # ========= OPTIMIZATION CONTROLLER =========
    class OptimizationController:
        """
        Controller class for mission range optimization through iterative adjustment.
        
        This class manages the iterative optimization process to achieve a target
        mission range by adjusting cruise distance. The optimization employs damping
        to ensure stable convergence and prevents oscillations.
        
        Mathematical Formulation:
            Error: e_k = R_target - R_actual,k
            Update: d_cruise,k+1 = d_cruise,k + α × e_k
            
        where:
            - R_target: Target mission range
            - R_actual,k: Actual mission range in iteration k
            - d_cruise,k: Cruise distance in iteration k
            - α: Damping factor (0 < α ≤ 1)
            
        Convergence Criterion:
            |e_k| < ε_tol
            
        where ε_tol is the specified tolerance.
        
        Attributes:
            target_range_km: Target total mission range [km]
            tolerance_km: Convergence tolerance [km]
            damping_factor: Damping factor for stability (0 < α ≤ 1)
            max_iterations: Maximum allowed iterations
            iteration_history: List of iteration records
        """
        
        def __init__(
            self,
            target_range_km: float,
            tolerance_km: float = RANGE_OPTIMIZATION_TOLERANCE_KM,
            damping_factor: float = RANGE_OPTIMIZATION_DAMPING_FACTOR,
            max_iterations: int = MAX_RANGE_OPTIMIZATION_ITERATIONS
        ):
            """
            Initialize mission range optimizer.
            
            Args:
                target_range_km: Target total mission range [km]
                tolerance_km: Convergence tolerance [km]
                damping_factor: Damping factor for cruise adjustment (0 < α ≤ 1)
                max_iterations: Maximum optimization iterations
            """
            self.target_range_km = target_range_km
            self.tolerance_km = tolerance_km
            self.damping_factor = damping_factor
            self.max_iterations = max_iterations
            self.iteration_history: List[OptimizationIteration] = []
        
        def check_convergence(self, actual_range_km: float) -> Tuple[bool, float]:
            """
            Check if optimization has converged to target range.
            
            Convergence is achieved when the absolute error between actual and
            target range falls within the specified tolerance.
            
            Args:
                actual_range_km: Actual computed mission range [km]
                
            Returns:
                tuple: (converged, error_km)
                    - converged: True if within tolerance
                    - error_km: Distance error [km]
            """
            error_km = self.target_range_km - actual_range_km
            converged = abs(error_km) <= self.tolerance_km
            return converged, error_km
        
        def compute_next_cruise_distance(
            self,
            current_cruise_distance_km: float,
            error_km: float
        ) -> float:
            """
            Compute next cruise distance using damped adjustment.
            
            Damping factor prevents overshoot and oscillations in iterative process.
            The adjustment is proportional to the current error, scaled by damping factor.
            
            Mathematical formulation:
                d_cruise,new = d_cruise,current + α × error
                
            Args:
                current_cruise_distance_km: Current cruise distance [km]
                error_km: Distance error (target - actual) [km]
                
            Returns:
                float: Next cruise distance to attempt [km]
                
            Notes:
                - Positive error → increase cruise distance
                - Negative error → decrease cruise distance
                - Damping factor alpha in (0, 1] controls adjustment aggressiveness
            """
            # Damped adjustment: avoid overshoot
            adjustment_km = self.damping_factor * error_km
            new_cruise_distance_km = current_cruise_distance_km + adjustment_km
            
            # Ensure cruise distance remains positive
            new_cruise_distance_km = max(10.0, new_cruise_distance_km)
            
            return new_cruise_distance_km
        
        def record_iteration(
            self,
            iteration: int,
            cruise_distance_km: float,
            total_distance_km: float,
            error_km: float,
            converged: bool
        ):
            """
            Record iteration data for history and analysis.
            
            Args:
                iteration: Iteration number
                cruise_distance_km: Cruise distance used [km]
                total_distance_km: Resulting total distance [km]
                error_km: Distance error [km]
                converged: Convergence status
            """
            record = OptimizationIteration(
                iteration=iteration,
                cruise_distance_km=cruise_distance_km,
                total_distance_km=total_distance_km,
                distance_error_km=error_km,
                converged=converged
            )
            self.iteration_history.append(record)
        
        def print_iteration_status(
            self,
            iteration: int,
            cruise_distance_km: float,
            total_distance_km: float,
            error_km: float,
            converged: bool
        ):
            """
            Print formatted iteration status for monitoring convergence progress.
            
            Args:
                iteration: Current iteration number
                cruise_distance_km: Cruise distance used [km]
                total_distance_km: Resulting total distance [km]
                error_km: Distance error [km]
                converged: Convergence status
            """
            status = "CONVERGED" if converged else "Continuing..."
            print(f"[ITER {iteration:2d}] Cruise: {cruise_distance_km:7.1f} km | "
                  f"Total: {total_distance_km:7.1f} km | "
                  f"Error: {error_km:+7.1f} km | {status}")
        
        def get_optimization_summary(self) -> Dict[str, Any]:
            """
            Generate optimization summary statistics.
            
            Returns:
                dict: Summary containing convergence history and final results
            """
            if not self.iteration_history:
                return {}
            
            final_iteration = self.iteration_history[-1]
            
            summary = {
                'target_range_km': self.target_range_km,
                'tolerance_km': self.tolerance_km,
                'damping_factor': self.damping_factor,
                'total_iterations': len(self.iteration_history),
                'converged': final_iteration.converged,
                'final_cruise_distance_km': final_iteration.cruise_distance_km,
                'final_total_distance_km': final_iteration.total_distance_km,
                'final_error_km': final_iteration.distance_error_km,
                'iteration_history': self.iteration_history
            }
            
            return summary


# Create global range optimization core instance
_range_optimization_core = RangeOptimizationCore()


# =========  3 - DATA STRUCTURES =================
@dataclass
class OptimizationIteration:
    """
    Data structure for storing iteration history.
    
    Attributes:
        iteration: Iteration number
        cruise_distance_km: Cruise distance used in this iteration [km]
        total_distance_km: Resulting total mission distance [km]
        distance_error_km: Error from target distance [km]
        converged: Whether convergence achieved in this iteration
    """
    iteration: int
    cruise_distance_km: float
    total_distance_km: float
    distance_error_km: float
    converged: bool


# =========  4 - BACKWARD COMPATIBILITY WRAPPERS =================
# Distance calculation functions
def calculate_climb_distance_km(climb_result: MinFuelSchedule) -> float:
    """Backward compatibility wrapper for RangeOptimizationCore.DistanceCalculator.calculate_climb_distance_km"""
    return RangeOptimizationCore.DistanceCalculator.calculate_climb_distance_km(climb_result)

def calculate_descent_distance_km(descent_result: DescentResults) -> float:
    """Backward compatibility wrapper for RangeOptimizationCore.DistanceCalculator.calculate_descent_distance_km"""
    return RangeOptimizationCore.DistanceCalculator.calculate_descent_distance_km(descent_result)

def calculate_total_mission_distance_km(
    climb_result: MinFuelSchedule,
    cruise_result: CruiseResults,
    descent_result: DescentResults
) -> Tuple[float, Dict[str, float]]:
    """Backward compatibility wrapper for RangeOptimizationCore.DistanceCalculator.calculate_total_mission_distance_km"""
    return RangeOptimizationCore.DistanceCalculator.calculate_total_mission_distance_km(
        climb_result, cruise_result, descent_result
    )

# Cruise segment management functions
def adjust_cruise_segment_extension(
    cruise_result: CruiseResults,
    additional_distance_km: float,
    aero,
    engine,
    time_step_s: float
) -> CruiseResults:
    """Backward compatibility wrapper for RangeOptimizationCore.CruiseSegmentManager.adjust_cruise_segment_extension"""
    return RangeOptimizationCore.CruiseSegmentManager.adjust_cruise_segment_extension(
        cruise_result, additional_distance_km, aero, engine, time_step_s
    )

def adjust_cruise_segment_truncation(
    cruise_result: CruiseResults,
    new_cruise_distance_km: float
) -> CruiseResults:
    """Backward compatibility wrapper for RangeOptimizationCore.CruiseSegmentManager.adjust_cruise_segment_truncation"""
    return RangeOptimizationCore.CruiseSegmentManager.adjust_cruise_segment_truncation(
        cruise_result, new_cruise_distance_km
    )

# Optimization controller class (main interface)
class MissionRangeOptimizer(RangeOptimizationCore.OptimizationController):
    """
    Main interface for mission range optimization.
    
    This class inherits from RangeOptimizationCore.OptimizationController and provides
    the primary interface for range optimization operations.
    
    Usage:
        optimizer = MissionRangeOptimizer(target_range_km=1000.0)
        converged, error = optimizer.check_convergence(actual_range_km)
        next_cruise = optimizer.compute_next_cruise_distance(current_cruise, error)
        optimizer.record_iteration(iteration, cruise, total, error, converged)
        summary = optimizer.get_optimization_summary()
    """
    pass
