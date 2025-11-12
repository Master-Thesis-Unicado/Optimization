# =========  1 - MODULE INITIALIZATION =================
# ========= IMPORTS AND BASIC SETUP ===========================================
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
    Mission range optimization framework for iterative convergence to target range.
    
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
    - Iteration history tracking for analysis
    
    Implementation:
        total_dist, breakdown = RangeOptimizationCore.DistanceCalculator.calculate_total_mission_distance_km(...)
        extended_cruise = RangeOptimizationCore.CruiseSegmentManager.adjust_cruise_segment_extension(...)
        optimizer = RangeOptimizationCore.OptimizationController(target_range_km=1000.0)
    """
    
    # ========= DISTANCE CALCULATION SYSTEM =========
    class DistanceCalculator:
        """Distance calculation for all flight phases using TAS integration."""
        
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
            """
            if len(climb_result.dt_s) == 0 or len(climb_result.mach) == 0:
                return 0.0
            
            climb_distances = []
            for i in range(len(climb_result.dt_s)):
                if i < len(climb_result.mach) and i < len(climb_result.alt_m):
                    a = a_from_altitude(climb_result.alt_m[i])
                    V_tas = climb_result.mach[i] * a
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
            """
            if len(descent_result.dt_s) == 0 or len(descent_result.mach) == 0:
                return 0.0
            
            descent_distances = []
            for i in range(len(descent_result.dt_s)):
                if i < len(descent_result.mach) and i < len(descent_result.alt_m):
                    a = a_from_altitude(descent_result.alt_m[i])
                    V_tas = descent_result.mach[i] * a
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
            """
            climb_distance_km = RangeOptimizationCore.DistanceCalculator.calculate_climb_distance_km(climb_result)
            cruise_distance_km = cruise_result.target_distance_km
            descent_distance_km = RangeOptimizationCore.DistanceCalculator.calculate_descent_distance_km(descent_result)
            
            total_distance_km = climb_distance_km + cruise_distance_km + descent_distance_km
            
            breakdown = {
                'climb_km': climb_distance_km,
                'cruise_km': cruise_distance_km,
                'descent_km': descent_distance_km,
                'total_km': total_distance_km
            }
            
            return total_distance_km, breakdown
    
    # ========= CRUISE SEGMENT MANAGEMENT SYSTEM =========
    class CruiseSegmentManager:
        """Cruise segment extension and truncation for efficient optimization."""
        
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
            """
            final_weight_kg = cruise_result.weight_kg[-1]
            final_altitude_m = cruise_result.altitude_m[-1]
            final_mach = cruise_result.mach_number[-1]
            previous_distance = cruise_result.target_distance_km
            previous_fuel = cruise_result.total_fuel_consumed_kg
            
            print(f"      [EXTENSION] Starting extension from previous cruise endpoint:")
            print(f"        Previous cruise distance: {previous_distance:.2f} km")
            print(f"        Mass at endpoint: {final_weight_kg:.1f} kg")
            print(f"        Fuel consumed in previous cruise: {previous_fuel:.1f} kg")
            print(f"        Extension distance: {additional_distance_km:.2f} km")
            
            extension_initial_state = CruiseInitialState(
                altitude_m=final_altitude_m,
                mach=final_mach,
                weight_kg=final_weight_kg,
                fuel_consumed_climb_kg=0.0,  
                climb_time_s=0.0   
            )
            
            extension_result = simulate_steady_cruise(
                initial_state=extension_initial_state,
                target_distance_km=additional_distance_km,
                aero=aero,
                engine=engine,
                time_step_s=time_step_s
            )
            
            # Time and fuel offsets for extension segment
            time_offset = cruise_result.time_s[-1]
            extension_time_offset = extension_result.time_s + time_offset
            
            fuel_offset = cruise_result.fuel_consumed_kg[-1]
            extension_fuel_offset = extension_result.fuel_consumed_kg + fuel_offset
            
            # Mass continuity verification
            extension_fuel = extension_result.total_fuel_consumed_kg
            extension_final_weight = extension_result.weight_kg[-1]
            
            previous_final_weight = cruise_result.weight_kg[-1]
            extension_start_weight = extension_result.weight_kg[0]
            weight_continuity_error = abs(previous_final_weight - extension_start_weight)
            
            print(f"      [EXTENSION] Extension segment results:")
            print(f"        Fuel consumed in extension: {extension_fuel:.1f} kg")
            print(f"        Final mass after extension: {extension_final_weight:.1f} kg")
            print(f"        Total combined cruise distance: {cruise_result.target_distance_km + additional_distance_km:.2f} km")
            print(f"        Weight continuity at junction: {weight_continuity_error:.3f} kg error (✓ OK)" if weight_continuity_error < 0.1 else f"        Weight continuity at junction: {weight_continuity_error:.3f} kg error (⚠ WARNING)")
            
            # Create combined cruise result (remove last point from previous cruise, keep all extension points)
            combined_result = CruiseResults(
                initial_state=cruise_result.initial_state,
                target_distance_km=cruise_result.target_distance_km + additional_distance_km,
                time_step_s=time_step_s,
                time_s=np.concatenate([cruise_result.time_s[:-1], extension_time_offset]),
                distance_km=np.concatenate([cruise_result.distance_km[:-1], 
                                           cruise_result.distance_km[-1] + extension_result.distance_km]),
                weight_kg=np.concatenate([cruise_result.weight_kg[:-1], extension_result.weight_kg]),
                fuel_consumed_kg=np.concatenate([cruise_result.fuel_consumed_kg[:-1], extension_fuel_offset]),
                thrust_total_N=np.concatenate([cruise_result.thrust_total_N[:-1], extension_result.thrust_total_N]),
                drag_N=np.concatenate([cruise_result.drag_N[:-1], extension_result.drag_N]),
                fuel_flow_kgps=np.concatenate([cruise_result.fuel_flow_kgps[:-1], extension_result.fuel_flow_kgps]),
                specific_excess_power_mps=np.concatenate([cruise_result.specific_excess_power_mps[:-1], 
                                                          extension_result.specific_excess_power_mps]),
                lever_position=np.concatenate([cruise_result.lever_position[:-1], extension_result.lever_position]),
                altitude_m=np.concatenate([cruise_result.altitude_m[:-1], extension_result.altitude_m]),
                mach_number=np.concatenate([cruise_result.mach_number[:-1], extension_result.mach_number]),
                temperature_K=np.concatenate([cruise_result.temperature_K[:-1], extension_result.temperature_K]),
                density_kgpm3=np.concatenate([cruise_result.density_kgpm3[:-1], extension_result.density_kgpm3]),
                true_airspeed_mps=np.concatenate([cruise_result.true_airspeed_mps[:-1], 
                                                 extension_result.true_airspeed_mps]),
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
            """
            distance_array = cruise_result.distance_km
            
            truncation_idx = np.searchsorted(distance_array, new_cruise_distance_km)
            
            if truncation_idx == 0:
                truncation_idx = 1
            if truncation_idx >= len(distance_array):
                truncation_idx = len(distance_array) - 1
            
            distance_at_idx = distance_array[truncation_idx]
            distance_error_km = abs(distance_at_idx - new_cruise_distance_km)
            
            # Check if interpolation is needed
            if distance_error_km > 0.01 and truncation_idx > 0:
                idx_before = truncation_idx - 1
                idx_after = truncation_idx
                
                d_before = distance_array[idx_before]
                d_after = distance_array[idx_after]
                
                alpha = (new_cruise_distance_km - d_before) / (d_after - d_before) if (d_after - d_before) > 0 else 0.0
                alpha = np.clip(alpha, 0.0, 1.0)
                
                mass_before = cruise_result.weight_kg[idx_before]
                mass_after = cruise_result.weight_kg[idx_after]
                interpolated_mass = mass_before + alpha * (mass_after - mass_before)
                
                fuel_before = cruise_result.fuel_consumed_kg[idx_before]
                fuel_after = cruise_result.fuel_consumed_kg[idx_after]
                interpolated_fuel = fuel_before + alpha * (fuel_after - fuel_before)
                
                time_before = cruise_result.time_s[idx_before]
                time_after = cruise_result.time_s[idx_after]
                interpolated_time = time_before + alpha * (time_after - time_before)
                
                print(f"      [TRUNCATION] Interpolation required:")
                print(f"        Target distance: {new_cruise_distance_km:.2f} km")
                print(f"        Bracketing points: [{idx_before}] {d_before:.2f} km → [{idx_after}] {d_after:.2f} km")
                print(f"        Interpolation weight α: {alpha:.4f}")
                print(f"        Interpolated mass: {interpolated_mass:.1f} kg (between {mass_before:.1f} and {mass_after:.1f})")
                print(f"        Interpolated fuel: {interpolated_fuel:.1f} kg")
                
                truncation_idx = idx_before
                requires_interpolation = True
            else:
                interpolated_mass = None
                interpolated_fuel = None
                interpolated_time = None
                requires_interpolation = False
                
                print(f"      [TRUNCATION] Using discrete cruise point:")
                print(f"        Index: {truncation_idx} of {len(distance_array)}")
                print(f"        Distance: {distance_at_idx:.2f} km (target: {new_cruise_distance_km:.2f} km, error: {distance_error_km*1000:.1f} m)")
                print(f"        Mass at this point: {cruise_result.weight_kg[truncation_idx]:.1f} kg")
                print(f"        Fuel consumed to this point: {cruise_result.fuel_consumed_kg[truncation_idx]:.1f} kg")
            
            if requires_interpolation:
                def interpolate_value(array, idx_before, alpha):
                    val_before = array[idx_before]
                    val_after = array[idx_before + 1]
                    return val_before + alpha * (val_after - val_before)
                
                truncated_result = CruiseResults(
                    initial_state=cruise_result.initial_state,
                    target_distance_km=new_cruise_distance_km,
                    time_step_s=cruise_result.time_step_s,
                    time_s=np.append(cruise_result.time_s[:truncation_idx+1], interpolated_time),
                    distance_km=np.append(cruise_result.distance_km[:truncation_idx+1], new_cruise_distance_km),
                    weight_kg=np.append(cruise_result.weight_kg[:truncation_idx+1], interpolated_mass),
                    fuel_consumed_kg=np.append(cruise_result.fuel_consumed_kg[:truncation_idx+1], interpolated_fuel),
                    thrust_total_N=np.append(cruise_result.thrust_total_N[:truncation_idx+1], 
                                            interpolate_value(cruise_result.thrust_total_N, truncation_idx, alpha)),
                    drag_N=np.append(cruise_result.drag_N[:truncation_idx+1],
                                    interpolate_value(cruise_result.drag_N, truncation_idx, alpha)),
                    fuel_flow_kgps=np.append(cruise_result.fuel_flow_kgps[:truncation_idx+1],
                                            interpolate_value(cruise_result.fuel_flow_kgps, truncation_idx, alpha)),
                    specific_excess_power_mps=np.append(cruise_result.specific_excess_power_mps[:truncation_idx+1],
                                                        interpolate_value(cruise_result.specific_excess_power_mps, truncation_idx, alpha)),
                    lever_position=np.append(cruise_result.lever_position[:truncation_idx+1],
                                            interpolate_value(cruise_result.lever_position, truncation_idx, alpha)),
                    altitude_m=np.append(cruise_result.altitude_m[:truncation_idx+1],
                                        interpolate_value(cruise_result.altitude_m, truncation_idx, alpha)),
                    mach_number=np.append(cruise_result.mach_number[:truncation_idx+1],
                                         interpolate_value(cruise_result.mach_number, truncation_idx, alpha)),
                    temperature_K=np.append(cruise_result.temperature_K[:truncation_idx+1],
                                           interpolate_value(cruise_result.temperature_K, truncation_idx, alpha)),
                    density_kgpm3=np.append(cruise_result.density_kgpm3[:truncation_idx+1],
                                           interpolate_value(cruise_result.density_kgpm3, truncation_idx, alpha)),
                    true_airspeed_mps=np.append(cruise_result.true_airspeed_mps[:truncation_idx+1],
                                                interpolate_value(cruise_result.true_airspeed_mps, truncation_idx, alpha)),
                    total_time_s=interpolated_time,
                    total_fuel_consumed_kg=interpolated_fuel,
                    final_weight_kg=interpolated_mass,
                    average_fuel_flow_kgps=interpolated_fuel / interpolated_time if interpolated_time > 0 else 0.0,
                    average_thrust_N=interpolate_value(cruise_result.thrust_total_N, truncation_idx, alpha)
                )
            else:
                truncated_result = CruiseResults(
                    initial_state=cruise_result.initial_state,
                    target_distance_km=new_cruise_distance_km,
                    time_step_s=cruise_result.time_step_s,
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
        Controller for mission range optimization through iterative adjustment.
        
        This class manages the iterative optimization process to achieve a target
        mission range by adjusting cruise distance. The optimization employs damping
        to ensure stable convergence and prevents oscillations.
        
        Mathematical Formulation:
            Error: e_k = R_target - R_actual,k
            Update: d_cruise,k+1 = d_cruise,k + α × e_k
            
        where α is the damping factor (0 < α ≤ 1).
        
        Convergence Criterion:
            |e_k| < ε_tol
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
            
            Args:
                actual_range_km: Actual computed mission range [km]
                
            Returns:
                tuple: (converged, error_km)
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
            
            Mathematical formulation:
                d_cruise,new = d_cruise,current + α × error
                
            Args:
                current_cruise_distance_km: Current cruise distance [km]
                error_km: Distance error (target - actual) [km]
                
            Returns:
                float: Next cruise distance to attempt [km]
            """
            adjustment_km = self.damping_factor * error_km
            new_cruise_distance_km = current_cruise_distance_km + adjustment_km
            new_cruise_distance_km = max(10.0, new_cruise_distance_km)
            
            return new_cruise_distance_km
        
        def record_iteration(
            self,
            iteration: int,
            cruise_distance_km: float,
            total_distance_km: float,
            error_km: float,
            converged: bool,
            cruise_final_mass_kg: float = 0.0,
            descent_initial_mass_kg: float = 0.0,
            descent_final_mass_kg: float = 0.0
        ):
            """
            Record iteration data for history and analysis.
            
            Args:
                iteration: Iteration number
                cruise_distance_km: Cruise distance used [km]
                total_distance_km: Resulting total distance [km]
                error_km: Distance error [km]
                converged: Convergence status
                cruise_final_mass_kg: Mass at end of cruise [kg]
                descent_initial_mass_kg: Mass at start of descent [kg]
                descent_final_mass_kg: Mass at end of descent [kg]
            """
            record = OptimizationIteration(
                iteration=iteration,
                cruise_distance_km=cruise_distance_km,
                total_distance_km=total_distance_km,
                distance_error_km=error_km,
                converged=converged,
                cruise_final_mass_kg=cruise_final_mass_kg,
                descent_initial_mass_kg=descent_initial_mass_kg,
                descent_final_mass_kg=descent_final_mass_kg
            )
            self.iteration_history.append(record)
        
        def print_iteration_status(
            self,
            iteration: int,
            cruise_distance_km: float,
            total_distance_km: float,
            error_km: float,
            converged: bool,
            cruise_final_mass_kg: float = None,
            descent_final_mass_kg: float = None
        ):
            """
            Print formatted iteration status for monitoring convergence progress.
            """
            status = "CONVERGED" if converged else "Continuing..."
            base_msg = (f"[ITER {iteration:2d}] Cruise: {cruise_distance_km:7.1f} km | "
                       f"Total: {total_distance_km:7.1f} km | "
                       f"Error: {error_km:+7.1f} km")
            
            if cruise_final_mass_kg is not None and descent_final_mass_kg is not None:
                mass_msg = f" | Mass: {cruise_final_mass_kg:7.1f}→{descent_final_mass_kg:7.1f} kg"
                print(base_msg + mass_msg + f" | {status}")
            else:
                print(base_msg + f" | {status}")
        
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
        
        def print_mass_evolution_summary(self):
            """
            Print detailed summary of mass evolution across iterations.
            
            This method displays how aircraft mass changed throughout the optimization
            process, demonstrating proper mass continuity between phases.
            """
            if not self.iteration_history:
                print("[MASS] No iteration history available")
                return
            
            print(f"\n{'='*80}")
            print("MASS EVOLUTION ACROSS ITERATIONS")
            print(f"{'='*80}")
            print(f"{'Iter':<6} {'Cruise Dist':>12} {'Cruise Final':>14} {'Descent Init':>14} "
                  f"{'Descent Final':>14} {'Continuity':>12}")
            print(f"{'':^6} {'[km]':>12} {'Mass [kg]':>14} {'Mass [kg]':>14} "
                  f"{'Mass [kg]':>14} {'Check':>12}")
            print("-"*80)
            
            for record in self.iteration_history:
                if record.cruise_final_mass_kg > 0 and record.descent_initial_mass_kg > 0:
                    mass_error = abs(record.cruise_final_mass_kg - record.descent_initial_mass_kg)
                    continuity_status = "✓ OK" if mass_error < 0.1 else f"⚠ {mass_error:.1f}kg"
                else:
                    continuity_status = "N/A"
                
                print(f"{record.iteration:<6} {record.cruise_distance_km:>12.1f} "
                      f"{record.cruise_final_mass_kg:>14.1f} {record.descent_initial_mass_kg:>14.1f} "
                      f"{record.descent_final_mass_kg:>14.1f} {continuity_status:>12}")
            
            print("="*80)
            
            if len(self.iteration_history) > 1:
                first_iter = self.iteration_history[0]
                last_iter = self.iteration_history[-1]
                
                cruise_mass_change = last_iter.cruise_final_mass_kg - first_iter.cruise_final_mass_kg
                descent_mass_change = last_iter.descent_final_mass_kg - first_iter.descent_final_mass_kg
                
                print(f"\nMass Evolution Summary:")
                print(f"  First iteration:")
                print(f"    Cruise final mass:  {first_iter.cruise_final_mass_kg:.1f} kg")
                print(f"    Descent final mass: {first_iter.descent_final_mass_kg:.1f} kg")
                print(f"  Final iteration:")
                print(f"    Cruise final mass:  {last_iter.cruise_final_mass_kg:.1f} kg")
                print(f"    Descent final mass: {last_iter.descent_final_mass_kg:.1f} kg")
                print(f"  Changes due to range optimization:")
                print(f"    Cruise final mass change:  {cruise_mass_change:+.1f} kg")
                print(f"    Descent final mass change: {descent_mass_change:+.1f} kg")
            
            print("="*80)


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
        cruise_final_mass_kg: Mass at end of cruise phase [kg]
        descent_initial_mass_kg: Mass at start of descent phase [kg]
        descent_final_mass_kg: Mass at end of descent phase [kg]
    """
    iteration: int
    cruise_distance_km: float
    total_distance_km: float
    distance_error_km: float
    converged: bool
    cruise_final_mass_kg: float = 0.0
    descent_initial_mass_kg: float = 0.0
    descent_final_mass_kg: float = 0.0


# =========  4 - BACKWARD COMPATIBILITY WRAPPERS =================
def calculate_total_mission_distance_km(
    climb_result: MinFuelSchedule,
    cruise_result: CruiseResults,
    descent_result: DescentResults
) -> Tuple[float, Dict[str, float]]:
    """Backward compatibility wrapper for RangeOptimizationCore.DistanceCalculator.calculate_total_mission_distance_km"""
    return RangeOptimizationCore.DistanceCalculator.calculate_total_mission_distance_km(
        climb_result, cruise_result, descent_result
    )

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
