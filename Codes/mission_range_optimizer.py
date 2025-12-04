# ========================================================================
# MISSION RANGE OPTIMIZATION MODULE
# ========================================================================
"""
Iterative range targeting via cruise distance adjustment.

Mathematical formulation:
    Objective: Match total mission range to target s_target
    Variables: s_cruise (adjustable), s_climb, s_descent (from optimization)
    Constraint: s_total = s_climb + s_cruise + s_descent = s_target
    
    Iteration scheme:
        Error: e_k = s_target - s_total,k
        Update: s_cruise,k+1 = s_cruise,k + α·e_k
        Convergence: |e_k| < ε_tol
    
    where α ∈ (0,1] is damping factor, ε_tol is tolerance.

Distance computation: s = ∫V dt ≈ Σ V_TAS,i·Δt_i
Small angle approximation: V_horizontal ≈ V_TAS for climb/descent.

Components:
    - DistanceCalculator: Integrate V_TAS over time for each phase
    - CruiseSegmentManager: Extension/truncation of cruise trajectory
    - OptimizationController: Iterative convergence with damping
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass

# Atmospheric model: a(h) for V_TAS = M·a(h)
from aircraft_config import a_from_altitude

# Mission phase data structures
from climb import MinFuelSchedule
from cruise import CruiseResults, CruiseInitialState, simulate_steady_cruise
from descent import DescentResults

# Optimization parameters: ε_tol, α_damp, N_iter,max
from mission_config import (
    RANGE_OPTIMIZATION_TOLERANCE_KM,
    RANGE_OPTIMIZATION_DAMPING_FACTOR,
    MAX_RANGE_OPTIMIZATION_ITERATIONS
)


# ========================================================================
# SECTION 1: RANGE OPTIMIZATION FRAMEWORK
# ========================================================================

class RangeOptimizationCore:
    """
    Mission range targeting via iterative cruise distance adjustment.
    
    Mathematical framework:
        Objective: s_total = s_target
        Control variable: s_cruise
        Fixed quantities: s_climb, s_descent (from DP optimization)
    
    Subsystems:
        - DistanceCalculator: s = ∫V dt for each phase
        - CruiseSegmentManager: Trajectory extension/truncation
        - OptimizationController: Damped iterative scheme
    
    Usage:
        # Distance computation
        s_total, breakdown = RangeOptimizationCore.DistanceCalculator.calculate_total_mission_distance_km(...)
        
        # Cruise adjustment
        cruise_extended = RangeOptimizationCore.CruiseSegmentManager.adjust_cruise_segment_extension(...)
        
        # Optimization
        optimizer = RangeOptimizationCore.OptimizationController(target_range_km=5000)
    """
    
    # ────────────────────────────────────────────────────────────────────
    # Distance Integration via TAS
    # ────────────────────────────────────────────────────────────────────
    
    class DistanceCalculator:
        """
        Horizontal ground distance computation via true airspeed integration.
        
        Formula: s = ∫V_TAS dt ≈ Σ V_TAS,i·Δt_i where V_TAS = M·a(h)
        Small angle approximation: V_horizontal ≈ V_TAS for climb/descent.
        """
        
        @staticmethod
        def calculate_climb_distance_km(climb_result: MinFuelSchedule) -> float:
            """
            Compute horizontal distance during climb via TAS integration.
            
            Integration: s_climb = ∫V_TAS dt ≈ Σ V_TAS,i·Δt_i
            True airspeed: V_TAS = M·a(h)
            Small angle: V_horizontal ≈ V_TAS (climb angle << 1)
            
            Parameters:
                climb_result: MinFuelSchedule - climb trajectory with M(t), h(t), Δt
                
            Returns:
                s_climb [km]: horizontal distance during climb
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
            Compute horizontal distance during descent via TAS integration.
            
            Integration: s_descent = ∫V_TAS dt ≈ Σ V_TAS,i·Δt_i
            True airspeed: V_TAS = M·a(h)
            Small angle: V_horizontal ≈ V_TAS (descent angle << 1)
            
            Parameters:
                descent_result: DescentResults - descent trajectory with M(t), h(t), Δt
                
            Returns:
                s_descent [km]: horizontal distance during descent
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
            Compute total mission range via phase summation.
            
            Total range: s_total = s_climb + s_cruise + s_descent
            Each component computed via TAS integration.
            
            Parameters:
                climb_result: MinFuelSchedule - climb trajectory
                cruise_result: CruiseResults - cruise trajectory
                descent_result: DescentResults - descent trajectory
                
            Returns:
                (s_total [km], breakdown_dict)
                breakdown_dict: {'climb_km', 'cruise_km', 'descent_km', 'total_km'}
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
    
    # ────────────────────────────────────────────────────────────────────
    # Cruise Trajectory Adjustment
    # ────────────────────────────────────────────────────────────────────
    
    class CruiseSegmentManager:
        """
        Cruise trajectory extension and truncation for range adjustment.
        
        Operations:
            - Extension: Continue from terminal state for Δs_additional
            - Truncation: Trim trajectory to s_new < s_current
        
        State continuity: m_junction maintained across segments.
        """
        
        @staticmethod
        def adjust_cruise_segment_extension(
            cruise_result: CruiseResults,
            additional_distance_km: float,
            aero,
            engine,
            time_step_s: float
        ) -> CruiseResults:
            """
            Extend cruise trajectory by Δs_additional from terminal state.
            
            Algorithm:
                1. Extract terminal state: (h_f, M_f, m_f) from cruise_result
                2. Simulate extension: X_ext from (h_f, M_f, m_f) for Δs_additional
                3. Concatenate trajectories: X_combined = [X_cruise[:-1], X_ext]
                4. Offset cumulative variables: t_ext += t_f, m_fuel,ext += m_fuel,f
            
            State continuity: m_junction = m_cruise,f = m_ext,0
            
            Parameters:
                cruise_result: CruiseResults - existing cruise trajectory
                additional_distance_km: Δs_additional [km] - extension distance
                aero: aerodynamics model
                engine: propulsion model
                time_step_s: Δt [s] - integration time step
                
            Returns:
                CruiseResults: extended trajectory with s_total = s_cruise + Δs_additional
            """
            final_mass_kg = cruise_result.mass_kg[-1]
            final_altitude_m = cruise_result.altitude_m[-1]
            final_mach = cruise_result.mach_number[-1]
            previous_distance = cruise_result.target_distance_km
            previous_fuel = cruise_result.total_fuel_consumed_kg
            
            print(f"      [EXTENSION] Starting extension from previous cruise endpoint:")
            print(f"        Previous cruise distance: {previous_distance:.2f} km")
            print(f"        Mass at endpoint: {final_mass_kg:.1f} kg")
            print(f"        Fuel consumed in previous cruise: {previous_fuel:.1f} kg")
            print(f"        Extension distance: {additional_distance_km:.2f} km")
            
            extension_initial_state = CruiseInitialState(
                altitude_m=final_altitude_m,
                mach=final_mach,
                mass_kg=final_mass_kg,
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
            extension_final_mass = extension_result.mass_kg[-1]
            
            previous_final_mass = cruise_result.mass_kg[-1]
            extension_start_mass = extension_result.mass_kg[0]
            mass_continuity_error = abs(previous_final_mass - extension_start_mass)
            
            print(f"      [EXTENSION] Extension segment results:")
            print(f"        Fuel consumed in extension: {extension_fuel:.1f} kg")
            print(f"        Final mass after extension: {extension_final_mass:.1f} kg")
            print(f"        Total combined cruise distance: {cruise_result.target_distance_km + additional_distance_km:.2f} km")
            print(f"        Mass continuity at junction: {mass_continuity_error:.3f} kg error (✓ OK)" if mass_continuity_error < 0.1 else f"        Mass continuity at junction: {mass_continuity_error:.3f} kg error (⚠ WARNING)")
            
            # Create combined cruise result (remove last point from previous cruise, keep all extension points)
            combined_result = CruiseResults(
                initial_state=cruise_result.initial_state,
                target_distance_km=cruise_result.target_distance_km + additional_distance_km,
                time_step_s=time_step_s,
                time_s=np.concatenate([cruise_result.time_s[:-1], extension_time_offset]),
                distance_km=np.concatenate([cruise_result.distance_km[:-1], 
                                           cruise_result.distance_km[-1] + extension_result.distance_km]),
                mass_kg=np.concatenate([cruise_result.mass_kg[:-1], extension_result.mass_kg]),
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
                final_mass_kg=extension_result.final_mass_kg,
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
            Truncate cruise trajectory to s_new < s_current.
            
            Algorithm:
                1. Find index: i such that s[i] ≈ s_new via binary search
                2. Check interpolation need: |s[i] - s_new| > ε_interp
                3. If needed: linear interpolation between s[i-1] and s[i]
                   α = (s_new - s[i-1])/(s[i] - s[i-1])
                   X_new = X[i-1] + α·(X[i] - X[i-1])
                4. Truncate arrays: X_truncated = X[:i+1] or [..., X_new]
            
            Parameters:
                cruise_result: CruiseResults - existing cruise trajectory
                new_cruise_distance_km: s_new [km] - target distance (s_new < s_current)
                
            Returns:
                CruiseResults: truncated trajectory with s_total = s_new
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
                
                mass_before = cruise_result.mass_kg[idx_before]
                mass_after = cruise_result.mass_kg[idx_after]
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
                print(f"        Mass at this point: {cruise_result.mass_kg[truncation_idx]:.1f} kg")
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
                    mass_kg=np.append(cruise_result.mass_kg[:truncation_idx+1], interpolated_mass),
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
                    final_mass_kg=interpolated_mass,
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
                    mass_kg=cruise_result.mass_kg[:truncation_idx+1],
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
                    final_mass_kg=cruise_result.mass_kg[truncation_idx],
                    average_fuel_flow_kgps=np.mean(cruise_result.fuel_flow_kgps[:truncation_idx+1]),
                    average_thrust_N=np.mean(cruise_result.thrust_total_N[:truncation_idx+1])
                )
            
            return truncated_result
    
    # ────────────────────────────────────────────────────────────────────
    # Iterative Convergence Controller
    # ────────────────────────────────────────────────────────────────────
    
    class OptimizationController:
        """
        Damped iterative optimization for range targeting.
        
        Iteration scheme:
            Error: e_k = s_target - s_total,k
            Update: s_cruise,k+1 = s_cruise,k + α·e_k
            Constraint: s_cruise,k+1 ≥ s_min (safety bound)
        
        Parameters:
            α ∈ (0,1]: damping factor (prevents overshoot)
            ε_tol [km]: convergence tolerance
            N_max: maximum iterations
        
        Convergence criterion: |e_k| < ε_tol
        """
        
        def __init__(
            self,
            target_range_km: float,
            tolerance_km: float = RANGE_OPTIMIZATION_TOLERANCE_KM,
            damping_factor: float = RANGE_OPTIMIZATION_DAMPING_FACTOR,
            max_iterations: int = MAX_RANGE_OPTIMIZATION_ITERATIONS
        ):
            """
            Initialize range optimization controller.
            
            Parameters:
                target_range_km: s_target [km] - target total mission range
                tolerance_km: ε_tol [km] - convergence tolerance
                damping_factor: α ∈ (0,1] - damping coefficient
                max_iterations: N_max - iteration limit
            """
            self.target_range_km = target_range_km
            self.tolerance_km = tolerance_km
            self.damping_factor = damping_factor
            self.max_iterations = max_iterations
            self.iteration_history: List[OptimizationIteration] = []
        
        def check_convergence(self, actual_range_km: float) -> Tuple[bool, float]:
            """
            Evaluate convergence criterion.
            
            Error: e = s_target - s_actual
            Convergence: |e| < ε_tol
            
            Parameters:
                actual_range_km: s_actual [km] - computed mission range
                
            Returns:
                (converged: bool, error_km: e [km])
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
            Compute next cruise distance via damped update.
            
            Update rule: s_cruise,k+1 = s_cruise,k + α·e_k
            Safety bound: s_cruise,k+1 ≥ s_min = 10 km
            
            Damping α ∈ (0,1] prevents overshoot and oscillations.
            
            Parameters:
                current_cruise_distance_km: s_cruise,k [km] - current cruise distance
                error_km: e_k [km] - range error (s_target - s_actual)
                
            Returns:
                s_cruise,k+1 [km]: next cruise distance
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
            Store iteration data in history.
            
            Parameters:
                iteration: k - iteration index
                cruise_distance_km: s_cruise,k [km]
                total_distance_km: s_total,k [km]
                error_km: e_k [km]
                converged: bool - convergence flag
                cruise_final_mass_kg: m_cruise,f [kg]
                descent_initial_mass_kg: m_descent,0 [kg]
                descent_final_mass_kg: m_descent,f [kg]
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
            """Print formatted iteration status: k, s_cruise, s_total, e_k, convergence."""
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
            Extract optimization summary and convergence history.
            
            Returns:
                dict: {s_target, ε_tol, α, N_iter, converged, final results, history}
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
            """Print mass continuity verification: m_cruise,f = m_descent,0 across iterations."""
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


# ========================================================================
# SECTION 2: DATA STRUCTURES
# ========================================================================

@dataclass
class OptimizationIteration:
    """
    Iteration history record.
    
    Fields:
        iteration: k - iteration index
        cruise_distance_km: s_cruise,k [km]
        total_distance_km: s_total,k [km]
        distance_error_km: e_k = s_target - s_total,k [km]
        converged: |e_k| < ε_tol
        cruise_final_mass_kg: m_cruise,f [kg]
        descent_initial_mass_kg: m_descent,0 [kg] (continuity check)
        descent_final_mass_kg: m_descent,f [kg]
    """
    iteration: int
    cruise_distance_km: float
    total_distance_km: float
    distance_error_km: float
    converged: bool
    cruise_final_mass_kg: float = 0.0
    descent_initial_mass_kg: float = 0.0
    descent_final_mass_kg: float = 0.0


# ========================================================================
# SECTION 3: BACKWARD COMPATIBILITY
# ========================================================================

def calculate_total_mission_distance_km(
    climb_result: MinFuelSchedule,
    cruise_result: CruiseResults,
    descent_result: DescentResults
) -> Tuple[float, Dict[str, float]]:
    """Deprecated: Use RangeOptimizationCore.DistanceCalculator.calculate_total_mission_distance_km"""
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
    """Deprecated: Use RangeOptimizationCore.CruiseSegmentManager.adjust_cruise_segment_extension"""
    return RangeOptimizationCore.CruiseSegmentManager.adjust_cruise_segment_extension(
        cruise_result, additional_distance_km, aero, engine, time_step_s
    )

def adjust_cruise_segment_truncation(
    cruise_result: CruiseResults,
    new_cruise_distance_km: float
) -> CruiseResults:
    """Deprecated: Use RangeOptimizationCore.CruiseSegmentManager.adjust_cruise_segment_truncation"""
    return RangeOptimizationCore.CruiseSegmentManager.adjust_cruise_segment_truncation(
        cruise_result, new_cruise_distance_km
    )

class MissionRangeOptimizer(RangeOptimizationCore.OptimizationController):
    """Deprecated: Use RangeOptimizationCore.OptimizationController directly."""
    pass
