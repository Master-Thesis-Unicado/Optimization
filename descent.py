# =========  1 - MODULE INITIALIZATION =================
# ========= IMPORTS AND BASIC SETUP ===========================================
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, NamedTuple, Callable, List, Dict, Any, Tuple
import time
from atmosphere import Atmosphere
import pyengine as engine

# Import aircraft configuration from centralized module
from aircraft_config import (
    SystemConfiguration, AtmosphericProperties,
    G_C, a_from_altitude, isa_properties,
    M_MIN_EFFECTIVE, M_MMO, INITIAL_MASS_KG, N_ENGINES, S_REF_M2
)

# Import necessary components from climb module
import climb
from climb import AeroTables, EngineWrapper, dbg, GridAndPlotting

# Import cruise module for initial state extraction
import cruise
from cruise import CruiseResults

# =========  2 - DATA STRUCTURES =============================================
@dataclass
class DescentInitialState:
    """Initial state for descent phase extracted from cruise results."""
    altitude_m: float
    mach: float
    weight_kg: float
    fuel_consumed_total_kg: float  # Total fuel consumed in climb + cruise
    total_time_s: float           # Total time for climb + cruise
    
    def __post_init__(self):
        """Validate initial descent state."""
        if self.altitude_m <= 300.0:  # TARGET_DESCENT_ALT_M
            raise ValueError(f"Descent altitude {self.altitude_m:.0f}m must be above target 300.0m")
        if not (0.2 <= self.mach <= M_MMO):
            raise ValueError(f"Descent Mach {self.mach:.3f} outside safe range [0.2-{M_MMO}]")
        if self.weight_kg <= 0:
            raise ValueError(f"Aircraft weight must be positive, got {self.weight_kg:.1f} kg")

@dataclass
class DescentResults:
    """Complete results from descent DP optimization."""
    # Metadata
    strategy_name: str
    
    # Trajectory arrays (similar to MinFuelSchedule from climb)
    alt_m: np.ndarray         # Altitude profile
    mach: np.ndarray          # Mach profile
    lever: np.ndarray         # Lever profile
    cumFuel_kg: np.ndarray    # Cumulative fuel consumed
    dt_s: np.ndarray          # Time steps
    dFuel_kg: np.ndarray      # Fuel increments
    
    # Additional performance arrays
    thrust_total_N: np.ndarray
    drag_N: np.ndarray
    fuel_flow_kgps: np.ndarray
    descent_rate_mps: np.ndarray
    temperature_K: np.ndarray
    density_kgpm3: np.ndarray
    true_airspeed_mps: np.ndarray
    specific_excess_power_mps: np.ndarray
    
    # Time array
    time_s: np.ndarray
    
    # Weight evolution
    weight_kg: np.ndarray
    
    # Summary statistics
    total_time_s: float
    total_fuel_consumed_kg: float
    final_weight_kg: float
    average_descent_rate_mps: float
    average_fuel_flow_kgps: float
    
    # Initial state for reference
    initial_altitude_m: float
    initial_mach: float
    initial_weight_kg: float
    target_altitude_m: float
    target_mach: float
    
    def get_summary_dict(self) -> Dict[str, Any]:
        """Get summary statistics as dictionary."""
        return {
            'strategy': self.strategy_name,
            'descent_altitude_change_m': self.initial_altitude_m - self.target_altitude_m,
            'descent_time_hours': self.total_time_s / 3600.0,
            'descent_time_minutes': self.total_time_s / 60.0,
            'descent_fuel_kg': self.total_fuel_consumed_kg,
            'avg_descent_rate_mps': self.average_descent_rate_mps,
            'avg_descent_rate_mpm': self.average_descent_rate_mps * 60.0,  # Convert to m/min
            'avg_fuel_flow_kg_h': self.average_fuel_flow_kgps * 3600.0,
            'initial_altitude_m': self.initial_altitude_m,
            'final_altitude_m': self.target_altitude_m,
            'initial_weight_kg': self.initial_weight_kg,
            'final_weight_kg': self.final_weight_kg,
            'initial_mach': self.initial_mach,
            'final_mach': self.target_mach,
        }

# =========  3 - MINIMUM MACH CALCULATION ========================
def calculate_min_descent_mach(altitude_m: float, weight_kg: float, 
                               cl_max: Optional[float] = None,
                               s_ref_m2: Optional[float] = None,
                               safety_margin: float = None) -> float:
    """
    Calculate minimum safe Mach number for descent based on stall speed.
    
    Uses aerodynamic stall speed formula with safety margin:
    V_stall = sqrt(2 * W / (rho * S_ref * CL_max))
    V_min = V_stall * safety_margin
    M_min = V_min / speed_of_sound
    
    Args:
        altitude_m: Current altitude in meters
        weight_kg: Current aircraft weight in kg
        cl_max: Maximum lift coefficient (defaults to CL_MAX from climb module)
        s_ref_m2: Reference wing area in m² (defaults to S_REF_M2)
        safety_margin: Safety factor above stall speed (defaults to 1.3)
        
    Returns:
        float: Minimum safe Mach number for the given conditions
        
    Notes:
        - Returns 0.15 as fallback if CLmax is not available
        - Applies reasonable bounds (0.15 to 0.40) to prevent unrealistic values
    """
    # Use defaults from climb module if not provided
    if cl_max is None:
        cl_max = climb.CL_MAX  # Access dynamically from climb module
    if s_ref_m2 is None:
        s_ref_m2 = S_REF_M2
    if safety_margin is None:
        safety_margin = 1.3  # 30% above stall
    
    # Validate inputs
    if cl_max is None or cl_max <= 0:
        dbg(f"[DESCENT] Warning: CLmax not available or invalid ({cl_max}), using fallback minimum Mach 0.15")
        return 0.15
    
    if s_ref_m2 is None or s_ref_m2 <= 0:
        dbg(f"[DESCENT] Warning: S_ref not available or invalid ({s_ref_m2}), using fallback minimum Mach 0.15")
        return 0.15
    
    try:
        # Get atmospheric properties at current altitude
        T, p, rho = isa_properties(altitude_m)
        a = a_from_altitude(altitude_m)
        
        if rho <= 0 or a <= 0:
            dbg(f"[DESCENT] Warning: Invalid atmospheric properties at {altitude_m:.0f}m, using fallback")
            return 0.15
        
        # Calculate stall speed: V_stall = sqrt(2*W/(rho*S*CL_max))
        weight_N = weight_kg * G_C
        q_min = weight_N / (s_ref_m2 * cl_max)  # Minimum dynamic pressure
        v_stall_mps = np.sqrt(2 * q_min / rho)  # Stall speed in m/s
        
        # Apply safety margin
        v_min_mps = v_stall_mps * safety_margin
        
        # Convert to Mach number
        m_min = v_min_mps / a
        
        # Apply reasonable bounds 
        m_min_bounded = np.clip(m_min, 0.15, 0.40)
        


        return float(m_min_bounded)
        
    except Exception as e:
        dbg(f"[DESCENT] Error calculating min Mach: {e}, using fallback")
        return 0.15

# =========  4 - DESCENDING CORE SYSTEM =============================================
class DescentCore:
    """
    Comprehensive aircraft descent optimization framework for mission analysis.
    
    This class implements a complete computational framework for aircraft descent performance
    analysis through integrated subsystems: dynamic programming optimization and penalty-based
    guidance. The system serves as the primary interface for descent trajectory analysis and
    optimization.
    
    System Components:
    - DynamicProgrammingOptimizer: Computes optimal descent trajectories through 3D state space (altitude, Mach, lever)
      to minimize fuel consumption while satisfying aircraft constraints
    - PenaltySystem: Provides Mach trajectory guidance and lever position penalties to direct optimization
      toward physically realizable flight paths and avoid infeasible solutions
    
    Computational Features:
    - Fuel-optimal descent path computation using dynamic programming in 3D state space
    - Penalty-based guidance to ensure physically realizable Mach trajectories and lever schedules
    - Integration with aerodynamic tables and engine models for accurate calculations
    
    Implementation:
        # Optimal descent calculation
        optimal_path = DescentCore.DynamicProgrammingOptimizer.solve_descent_dp(aero=aero, eng=engine, ...)
        
        # Cost evaluation
        cost = DescentCore.compute_descent_cost(aero=aero, eng=engine, altitude=10000, mach=0.8, lever=0.1, ...)
    """
    
    # ========= PENALTY SYSTEM (Nested Class) =========
    class PenaltySystem:
        """System for computing various penalties in the optimization process."""
        
        # Feature flags
        MACH_TRAJECTORY_GUIDANCE = True  # Enable reachability-constrained Mach guidance
        LEVER_PENALTY_GUIDANCE = True  # Enable penalties for high lever positions
        
        # Mach targeting constants
        TARGET_MACH_TOLERANCE = 0.015  # Tolerance for target Mach constraint in DP
        
        # Mach trajectory guidance constants
        MACH_PENALTY_BASE_WEIGHT = 0.3  # Base penalty weight (kg per Mach² deviation)
        MAX_REASONABLE_MACH_RATE = 0.02  # Max reasonable Mach change per optimization step
        TOTAL_DESCENT_STEPS_ESTIMATE = 50  # Matches N_PLOT_STEPS - actual DP grid steps
        URGENCY_MULTIPLIER = 2.0  # How much urgency scales with descent progress
        GUIDANCE_PENALTY_WEIGHT = 0.5  # Strong guidance penalty when inside reachable corridor
        
        # Lever penalty guidance constants
        LEVER_PENALTY_WEIGHT = 3.0  # Base weight for lever penalty (kg per lever unit above threshold)
        LEVER_PENALTY_THRESHOLD = 0.85  # Lever threshold above which penalties apply (85% = realistic descent limit)
        LEVER_PENALTY_EXPONENT = 3.0  # Exponent for penalty curve (higher = more aggressive)
        LEVER_PENALTY_CRITICAL_THRESHOLD = 0.90  # Critical threshold for very high penalties (90%+)
        LEVER_PENALTY_CRITICAL_MULTIPLIER = 5.0  # Extra penalty multiplier for critical range
        LEVER_PENALTY_ULTRA_CRITICAL_THRESHOLD = 0.95  # Ultra-critical threshold for maximum penalties (95%+)
        LEVER_PENALTY_ULTRA_CRITICAL_MULTIPLIER = 20.0  # Ultra-critical penalty multiplier (emergency thrust only)
        
        @staticmethod
        def compute_mach_penalty(current_mach: float, target_mach: float, prev_mach: float = None, 
                                descent_fraction: float = None) -> float:
            """
            Compute penalty using reachability-constrained approach FOR DESCENT.
            
            INVERTED FROM CLIMB: For descent, target is LOW Mach (0.25) at LOW altitude (300m).
            Creates a dynamic safety corridor that ensures target remains achievable
            with realistic Mach change rates.
            
            Args:
                current_mach: Current Mach number
                target_mach: Final target Mach number (0.25 for approach)
                prev_mach: Previous Mach number (unused - kept for API compatibility)
                descent_fraction: Fraction of descent progress (0.0 = start, 1.0 = target)
            
            Returns:
                penalty: Penalty value in kg per meter
            """
            if descent_fraction is None:
                descent_fraction = 0.0
            
            # Calculate remaining descent fraction and steps
            remaining_fraction = 1.0 - descent_fraction
            estimated_steps_remaining = remaining_fraction * DescentCore.PenaltySystem.TOTAL_DESCENT_STEPS_ESTIMATE
            
            # Calculate maximum achievable Mach change with reasonable rates
            max_achievable_change = DescentCore.PenaltySystem.MAX_REASONABLE_MACH_RATE * estimated_steps_remaining
            
            # Define reachability corridor bounds
            # For descent: target is LOWER than start, so corridor is around target
            min_reachable_mach = target_mach - max_achievable_change
            max_reachable_mach = target_mach + max_achievable_change
            
            # Calculate urgency factor (increases as we approach target altitude)
            urgency = (1.0 - remaining_fraction) * DescentCore.PenaltySystem.URGENCY_MULTIPLIER
            
            # Apply penalties based on position relative to corridor
            if current_mach < min_reachable_mach:
                # Below corridor - too slow, risk of stall
                deviation = min_reachable_mach - current_mach
                penalty = urgency * DescentCore.PenaltySystem.MACH_PENALTY_BASE_WEIGHT * (deviation ** 2)
                
            elif current_mach > max_reachable_mach:
                # Above corridor - too fast, won't slow down in time
                deviation = current_mach - max_reachable_mach  
                penalty = urgency * DescentCore.PenaltySystem.MACH_PENALTY_BASE_WEIGHT * (deviation ** 2)
                
            else:
                # Within corridor - apply progressive guidance toward target
                if descent_fraction > 0.7:
                    # Strong final phase guidance (70-100% descent)
                    final_phase_strength = (descent_fraction - 0.7) / 0.3  # 0 to 1 scaling
                    mach_deviation = current_mach - target_mach
                    
                    # Extra penalty boost for final 10% of descent
                    if descent_fraction > 0.9:
                        final_boost = ((descent_fraction - 0.9) / 0.1) * 2.0  # 0 to 2x multiplier
                        final_phase_strength *= (1.0 + final_boost)
                        
                    penalty = final_phase_strength * DescentCore.PenaltySystem.GUIDANCE_PENALTY_WEIGHT * (mach_deviation ** 2)
                else:
                    penalty = 0.0  # No penalty in early descent phase
            
            return penalty
        
        @staticmethod
        def compute_lever_penalty(current_lever: float, descent_fraction: float = None) -> float:
            """
            Compute penalty for high lever positions to encourage realistic engine usage.
            
            Engine limits are altitude-independent - high thrust settings cause the same
            thermal and mechanical stress regardless of altitude.
            
            Real-world considerations:
            - 85% lever = Maximum Continuous Thrust (MCT) - unlimited duration
            - 90%+ lever = Takeoff/Go-around thrust - limited duration, high wear
            - 95%+ lever = Maximum Takeoff Thrust - emergency use only, severe penalties
            
            Args:
                current_lever: Current lever position (0.0 to 1.0)
                descent_fraction: Unused parameter (kept for backward compatibility)
            
            Returns:
                penalty: Penalty value in kg (altitude-independent)
            """
            penalty = 0.0
            
            # Only apply penalty if lever exceeds MCT threshold (85%)
            if current_lever > DescentCore.PenaltySystem.LEVER_PENALTY_THRESHOLD:
                # Calculate excess lever above MCT threshold
                excess_lever = current_lever - DescentCore.PenaltySystem.LEVER_PENALTY_THRESHOLD
                
                # Base penalty using exponential curve for realistic behavior
                lever_penalty = excess_lever ** DescentCore.PenaltySystem.LEVER_PENALTY_EXPONENT
                
                # Apply critical penalty for very high lever positions (90%+)
                if current_lever > DescentCore.PenaltySystem.LEVER_PENALTY_CRITICAL_THRESHOLD:
                    critical_excess = current_lever - DescentCore.PenaltySystem.LEVER_PENALTY_CRITICAL_THRESHOLD
                    critical_penalty = critical_excess ** (DescentCore.PenaltySystem.LEVER_PENALTY_EXPONENT + 1.0)
                    lever_penalty += critical_penalty * DescentCore.PenaltySystem.LEVER_PENALTY_CRITICAL_MULTIPLIER
                
                # Apply ultra-critical penalty for maximum thrust positions (95%+)
                if current_lever > DescentCore.PenaltySystem.LEVER_PENALTY_ULTRA_CRITICAL_THRESHOLD:
                    ultra_critical_excess = current_lever - DescentCore.PenaltySystem.LEVER_PENALTY_ULTRA_CRITICAL_THRESHOLD
                    ultra_critical_penalty = ultra_critical_excess ** (DescentCore.PenaltySystem.LEVER_PENALTY_EXPONENT + 2.0)
                    lever_penalty += ultra_critical_penalty * DescentCore.PenaltySystem.LEVER_PENALTY_ULTRA_CRITICAL_MULTIPLIER
                
                # Use constant penalty weight - engine limits are altitude-independent
                penalty_weight = DescentCore.PenaltySystem.LEVER_PENALTY_WEIGHT
                
                penalty = penalty_weight * lever_penalty
            
            return penalty
    
    # ========= DYNAMIC PROGRAMMING OPTIMIZATION SYSTEM =========
    class DynamicProgrammingOptimizer:
        """3D Dynamic Programming optimizer for minimum fuel descent with penalty guidance."""
        
        @staticmethod
        def solve_descent_dp(aero: AeroTables, eng: EngineWrapper,
                            M_grid: np.ndarray, H_sched: np.ndarray,
                            initial_state: DescentInitialState,
                            lever_samples: int = 10,
                            target_mach: float = None,
                            target_mach_tolerance: float = 0.015):
            """
            3D Dynamic Programming solver for minimum fuel descent optimization WITH PENALTY GUIDANCE.
            
            Similar to climb DP but optimized for descent (minimizing fuel consumption).
            Uses penalty system to guide toward target Mach (0.25) at approach altitude (300m).
            
            Args:
                aero: Aerodynamics tables
                eng: Engine wrapper
                M_grid: Mach number grid
                H_sched: Altitude schedule (descending from high to low)
                initial_state: Initial state from cruise
                lever_samples: Number of lever positions to consider
                target_mach: Target Mach number at final altitude (0.25)
                target_mach_tolerance: Tolerance for target Mach constraint (0.015)
            
            Returns:
                DescentResults: Optimal descent schedule
                dict: Additional information (costs, path, etc.)
            """
            if target_mach is None:
                target_mach = 0.25  # Default target approach Mach
            
            print(f"[DP-DESCENT] Starting 3D Dynamic Programming optimization with penalty guidance...")
            print(f"[DP-DESCENT] Target: {target_mach:.3f} Mach at {H_sched[-1]:.0f}m altitude")
            print(f"[DP-DESCENT] Grid: {len(M_grid)} Mach × {len(H_sched)} Alt × {lever_samples} Lever")
            
            K, I = len(H_sched), len(M_grid)
            L = lever_samples
            
            # Create lever grid (same as climb: full range 0-100%)
            lever_grid = np.linspace(0.0, 1.0, L)
            
            # Initialize 3D cost matrix and predecessor array
            F = np.full((K, I, L), np.inf)
            prv = np.full((K, I, L, 3), -1, dtype=int)  # [alt_idx, mach_idx, lever_idx]
            
            # Set starting point (from cruise altitude)
            start_mach_idx = np.argmin(np.abs(M_grid - initial_state.mach))
            start_lever_idx = 0  # Start at idle
            actual_start_mach = M_grid[start_mach_idx]
            
            dbg(f"[DP-DESCENT] Starting from cruise: h={H_sched[0]:.0f}m, M={actual_start_mach:.3f}")
            
            # Verify starting point is in bounds
            if (actual_start_mach >= M_MIN_EFFECTIVE and 
                actual_start_mach <= M_MMO):
                F[0, start_mach_idx, start_lever_idx] = 0.0  # Starting cost is 0
                dbg(f"[DP-DESCENT] Starting point verified at Mach {actual_start_mach:.3f}")
            else:
                raise RuntimeError(f"[DP-DESCENT] Starting point out of bounds: M={actual_start_mach:.3f}")
            
            # Forward pass - 3D Dynamic Programming (descending)
            for k in range(K - 1):  # For each altitude level
                current_alt = H_sched[k]
                next_alt = H_sched[k + 1]
                dh = next_alt - current_alt  # Negative for descent
                
                # Calculate descent fraction for penalty system
                descent_fraction = k / (K - 1.0) if K > 1 else 0.0
                
                if k % 10 == 0:
                    dbg(f"[DP-DESCENT] Processing altitude {current_alt:.0f}m -> {next_alt:.0f}m "
                        f"(progress: {descent_fraction*100:.1f}%)")
                
                # Find all feasible current states
                feasible_states = np.where(np.isfinite(F[k]))
                feasible_count = 0
                
                for state_idx in range(len(feasible_states[0])):
                    i = feasible_states[0][state_idx]  # Mach index
                    j = feasible_states[1][state_idx]  # Lever index
                    
                    if not np.isfinite(F[k, i, j]):
                        continue
                    
                    current_mach = M_grid[i]
                    current_lever = lever_grid[j]
                    
                    # Consider neighboring states (5x5 grid in Mach-Lever space)
                    for di in [-2, -1, 0, 1, 2]:  # Mach change
                        for dj in [-2, -1, 0, 1, 2]:  # Lever change
                            next_mach_idx = i + di
                            next_lever_idx = j + dj
                            
                            # Check bounds
                            if (0 <= next_mach_idx < I and 
                                0 <= next_lever_idx < L and
                                k + 1 < K):
                                
                                next_mach = M_grid[next_mach_idx]
                                next_lever = lever_grid[next_lever_idx]
                                
                                # Calculate dynamic minimum Mach at next altitude
                                min_mach_next = calculate_min_descent_mach(next_alt, initial_state.weight_kg)
                                
                                # Check feasibility
                                if (next_mach >= min_mach_next and 
                                    next_mach <= M_MMO and
                                    next_lever >= 0.0 and 
                                    next_lever <= 1.0):
                                    
                                    # Compute fuel costs WITH PENALTIES
                                    current_cost = DescentCore.compute_descent_cost(
                                        aero, eng, current_alt, current_mach, current_lever,
                                        initial_state.weight_kg, target_mach, descent_fraction
                                    )
                                    next_cost = DescentCore.compute_descent_cost(
                                        aero, eng, next_alt, next_mach, next_lever,
                                        initial_state.weight_kg, target_mach, (k+1)/(K-1.0) if K > 1 else 1.0
                                    )
                                    
                                    if (np.isfinite(current_cost) and np.isfinite(next_cost) and
                                        current_cost > 0 and next_cost > 0):
                                        
                                        # Trapezoidal integration for fuel cost
                                        step_cost = 0.5 * (current_cost + next_cost) * abs(dh)
                                        total_cost = F[k, i, j] + step_cost
                                        
                                        # Update if this path is better
                                        if total_cost < F[k + 1, next_mach_idx, next_lever_idx]:
                                            F[k + 1, next_mach_idx, next_lever_idx] = total_cost
                                            prv[k + 1, next_mach_idx, next_lever_idx] = [k, i, j]
                                            feasible_count += 1
                
                if feasible_count == 0 and k < K - 2:
                    dbg(f"[DP-DESCENT] WARNING: No feasible transitions at altitude {current_alt:.0f}m")
            
            # Apply terminal Mach constraint
            dbg(f"[DP-DESCENT] Applying target Mach constraint: {target_mach:.3f} ± {target_mach_tolerance:.3f}")
            valid_final = np.abs(M_grid - target_mach) < target_mach_tolerance
            
            if not valid_final.any():
                dbg(f"[DP-DESCENT] Warning: No Mach within tolerance, using closest")
                closest_idx = np.argmin(np.abs(M_grid - target_mach))
                valid_final = np.zeros_like(valid_final, dtype=bool)
                valid_final[closest_idx] = True
            
            for i in range(I):
                if not valid_final[i]:
                    F[-1, i, :] = np.inf
            
            # Check if any path reached final altitude
            if not np.isfinite(F[-1]).any():
                raise RuntimeError("[DP-DESCENT] No feasible path reached final altitude with target constraints")
            
            # Find optimal final state
            final_flat_idx = np.nanargmin(F[-1])
            final_mach_idx, final_lever_idx = np.unravel_index(final_flat_idx, F[-1].shape)
            final_alt_idx = K - 1
            
            final_mach = M_grid[final_mach_idx]
            final_lever = lever_grid[final_lever_idx]
            final_alt = H_sched[final_alt_idx]
            final_cost = F[final_alt_idx, final_mach_idx, final_lever_idx]
            
            dbg(f"[DP-DESCENT] Optimal final state:")
            dbg(f"  Altitude: {final_alt:.0f}m (target: {H_sched[-1]:.0f}m)")
            dbg(f"  Mach: {final_mach:.3f} (target: {target_mach:.3f})")
            dbg(f"  Lever: {final_lever:.3f}")
            dbg(f"  Total fuel cost: {final_cost:.2f} kg")
            
            # Backtrack to find optimal path
            path_alt = []
            path_mach = []
            path_lever = []
            path_costs = []
            
            current_state = [final_alt_idx, final_mach_idx, final_lever_idx]
            
            while current_state[0] >= 0:
                alt_idx, mach_idx, lever_idx = current_state
                
                path_alt.append(H_sched[alt_idx])
                path_mach.append(M_grid[mach_idx])
                path_lever.append(lever_grid[lever_idx])
                path_costs.append(F[alt_idx, mach_idx, lever_idx])
                
                if alt_idx > 0:
                    current_state = prv[alt_idx, mach_idx, lever_idx].tolist()
                else:
                    break
            
            # Reverse to get correct order (start to finish)
            path_alt = path_alt[::-1]
            path_mach = path_mach[::-1]
            path_lever = path_lever[::-1]
            path_costs = path_costs[::-1]
            
            # Convert to arrays
            alt_array = np.array(path_alt)
            mach_array = np.array(path_mach)
            lever_array = np.array(path_lever)
            fuel_array = np.array(path_costs)
            
            # Calculate detailed trajectory data
            n_points = len(alt_array)
            time_array = np.zeros(n_points)
            dt_array = np.zeros(n_points)
            dFuel_array = np.diff(fuel_array, prepend=0)
            
            thrust_array = np.zeros(n_points)
            drag_array = np.zeros(n_points)
            fuel_flow_array = np.zeros(n_points)
            descent_rate_array = np.zeros(n_points)
            T_array = np.zeros(n_points)
            rho_array = np.zeros(n_points)
            V_array = np.zeros(n_points)
            Ps_array = np.zeros(n_points)
            weight_array = np.full(n_points, initial_state.weight_kg)
            
            # Calculate time increments and performance data
            for i in range(n_points):
                h = alt_array[i]
                M = mach_array[i]
                lever = lever_array[i]
                
                # Atmospheric properties
                T, p, rho = isa_properties(h)
                a = a_from_altitude(h)
                V = M * a
                
                T_array[i] = T
                rho_array[i] = rho
                V_array[i] = V
                
                # Thrust and drag
                T_per = eng.thrust_with_lever(lever, M, h)
                if T_per is None or T_per < 0:
                    T_per = 0.0
                T_tot = T_per * N_ENGINES
                D = aero.get_drag(M, h)
                
                thrust_array[i] = T_tot
                drag_array[i] = D
                
                # Fuel flow
                tsfc = eng.tsfc_current()
                if tsfc is None or tsfc < 0:
                    tsfc = 0.0
                mdot = tsfc * T_per * N_ENGINES
                fuel_flow_array[i] = mdot
                
                # Calculate Ps and descent rate
                W = weight_array[i] * G_C
                Ps = ((T_tot - D) * V) / W if W > 0 else 0.0
                Ps_array[i] = Ps
                
                # Time increment for next step
                if i < n_points - 1:
                    h_next = alt_array[i + 1]
                    dh = h_next - h  # Negative for descent
                    
                    if abs(Ps) > 0.1:
                        dt = abs(dh) / abs(Ps)
                    else:
                        dt = 1.0
                    
                    dt_array[i + 1] = dt
                    time_array[i + 1] = time_array[i] + dt
                    descent_rate_array[i] = Ps
                    
                    # Update weight
                    fuel_burned = mdot * dt
                    if i + 1 < n_points:
                        weight_array[i + 1] = weight_array[i] - fuel_burned
            
            # Final statistics
            total_time = time_array[-1]
            total_fuel = fuel_array[-1]
            final_weight = weight_array[-1]
            avg_descent_rate = np.mean(np.abs(descent_rate_array[descent_rate_array != 0])) if np.any(descent_rate_array != 0) else 0.0
            avg_fuel_flow = total_fuel / total_time if total_time > 0 else 0.0
            
            # Create result object (similar to MinFuelSchedule)
            descent_result = DescentResults(
                strategy_name="3D DP Optimal Descent (with Penalty Guidance)",
                alt_m=alt_array,
                mach=mach_array,
                lever=lever_array,
                cumFuel_kg=fuel_array,
                dt_s=dt_array,
                dFuel_kg=dFuel_array,
                thrust_total_N=thrust_array,
                drag_N=drag_array,
                fuel_flow_kgps=fuel_flow_array,
                descent_rate_mps=descent_rate_array,
                temperature_K=T_array,
                density_kgpm3=rho_array,
                true_airspeed_mps=V_array,
                specific_excess_power_mps=Ps_array,
                time_s=time_array,
                weight_kg=weight_array,
                total_time_s=total_time,
                total_fuel_consumed_kg=total_fuel,
                final_weight_kg=final_weight,
                average_descent_rate_mps=avg_descent_rate,
                average_fuel_flow_kgps=avg_fuel_flow,
                initial_altitude_m=initial_state.altitude_m,
                initial_mach=initial_state.mach,
                initial_weight_kg=initial_state.weight_kg,
                target_altitude_m=H_sched[-1],
                target_mach=target_mach
            )
            
            info = {
                'total_fuel_kg': float(total_fuel),
                'total_time_s': float(total_time),
                'total_time_min': float(total_time / 60.0),
                'final_mach': float(final_mach),
                'final_altitude': float(final_alt),
                'target_mach': target_mach,
                'target_altitude': H_sched[-1],
                'mach_deviation': abs(final_mach - target_mach),
                'path_length': len(alt_array),
                'cost_matrix_3d': F,
                'predecessor_matrix': prv,
                'avg_descent_rate_mps': avg_descent_rate,
                'avg_descent_rate_mpm': avg_descent_rate * 60.0,
            }
            
            print(f"[DP-DESCENT] Optimization complete!")
            print(f"  Total fuel: {total_fuel:.2f} kg")
            print(f"  Total time: {total_time/60:.1f} min")
            print(f"  Final Mach: {final_mach:.3f} (target: {target_mach:.3f}, deviation: {abs(final_mach-target_mach):.4f})")
            print(f"  Avg descent rate: {avg_descent_rate:.2f} m/s ({avg_descent_rate*60.0:.0f} m/min)")
            
            return descent_result, info

    # ========= COST COMPUTATION (Similar to ClimbingCore.compute_3d_cost) =========
    @staticmethod
    def compute_descent_cost(aero: AeroTables, eng: EngineWrapper,
                            altitude: float, mach: float, lever: float,
                            mass_kg: float,
                            target_mach: float = None,
                            descent_fraction: float = None) -> float:
        """
        Compute fuel cost density J = mdot/|Ps| + penalties for a given 3D state.
        
        Args:
            aero: Aerodynamics tables
            eng: Engine wrapper
            altitude: Altitude in meters
            mach: Mach number
            lever: Throttle lever position (0-1)
            mass_kg: Aircraft mass in kg
            target_mach: Target Mach number for penalty calculation
            descent_fraction: Fraction of descent progress for adaptive penalties
        
        Returns:
            float: Fuel cost density in kg/m + penalties, or inf if infeasible
        """
        if target_mach is None:
            target_mach = 0.25  # Default target approach Mach
        
        try:
            # Get atmospheric properties
            a = a_from_altitude(altitude)
            V = mach * a
            
            # Get thrust
            T_per = eng.thrust_with_lever(lever, mach, altitude)
            if T_per is None or not np.isfinite(T_per) or T_per < 0:
                return np.inf
            
            T_tot = T_per * N_ENGINES
            
            # Get drag
            D = aero.get_drag(mach, altitude)
            if not np.isfinite(D) or D < 0:
                return np.inf
            
            # Calculate specific excess power
            W = mass_kg * G_C
            Ps = ((T_tot - D) * V) / W
            
            # For descent, Ps should be negative (energy dissipation)
            if Ps >= 0:  # Can't descend with positive/zero Ps
                return np.inf
            
            # Get fuel flow
            tsfc = eng.tsfc_current()
            if tsfc is None or not np.isfinite(tsfc) or tsfc < 0:
                return np.inf
            
            mdot = tsfc * T_per * N_ENGINES
            
            # Base fuel cost density (fuel per meter of descent)
            J = mdot / abs(Ps)
            
            if not np.isfinite(J) or J <= 0:
                return np.inf
            
            # Add Mach penalty if guidance is enabled
            if target_mach is not None and DescentCore.PenaltySystem.MACH_TRAJECTORY_GUIDANCE:
                mach_penalty = DescentCore.PenaltySystem.compute_mach_penalty(
                    mach, target_mach, None, descent_fraction
                )
                J += mach_penalty
            
            # Add lever penalty if guidance is enabled
            if DescentCore.PenaltySystem.LEVER_PENALTY_GUIDANCE:
                lever_penalty = DescentCore.PenaltySystem.compute_lever_penalty(lever, descent_fraction)
                J += lever_penalty
            
            return J
            
        except Exception:
            return np.inf
    
    # ========= ENGINE ENVELOPE SYSTEM =========
    @staticmethod
    def compute_full_descent_envelope(aero: AeroTables, eng: EngineWrapper,
                                      M_grid: np.ndarray, H_sched: np.ndarray,
                                      initial_weight_kg: float,
                                      lever_samples: int = 50,
                                      target_mach: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the full descent envelope showing all possible J points (fuel cost density)
        that the engine can achieve across the entire Mach-Altitude-Lever space.
        
        This method provides comprehensive engine performance analysis for visualization
        and optimization purposes.
        
        Args:
            aero: Aerodynamics tables
            eng: Engine wrapper
            M_grid: Mach number grid
            H_sched: Altitude schedule (descending from high to low)
            initial_weight_kg: Initial weight at start of descent
            lever_samples: Number of lever positions to sample
            target_mach: Target Mach for penalty calculation
            
        Returns:
            tuple: (J_envelope_transposed, lever_grid) - Engine envelope data and lever grid
        """
        if target_mach is None:
            target_mach = 0.25  # Default target approach Mach
        
        print("[ENVELOPE-DESCENT] Computing full descent envelope...")
        print(f"  Grid: {len(M_grid)} Mach × {len(H_sched)} Alt × {lever_samples} Lever")
        
        # Create lever grid (FULL ENGINE ENVELOPE: 0-100% for visualization)
        lever_grid = np.linspace(0.0, 1.0, lever_samples)
        
        # Initialize 3D cost matrix
        I, K, L = len(M_grid), len(H_sched), len(lever_grid)
        J_envelope = np.full((I, K, L), np.nan)
        
        # Compute J = mdot/|Ps| for all feasible points
        feasible_count = 0
        total_points = I * K * L
        
        for k, h in enumerate(H_sched):
            # Calculate descent fraction for penalty system
            descent_fraction = k / (K - 1.0) if K > 1 else 0.0
            
            # Calculate dynamic minimum Mach at this altitude
            min_mach_h = calculate_min_descent_mach(h, initial_weight_kg)
            
            for i, M in enumerate(M_grid):
                # Skip if Mach is out of valid range
                if M < min_mach_h or M > M_MMO:
                    continue
                    
                for j, lever in enumerate(lever_grid):
                    # Compute cost using the same function as DP
                    cost = DescentCore.compute_descent_cost(
                        aero, eng, h, M, lever,
                        initial_weight_kg,
                        target_mach=target_mach,
                        descent_fraction=descent_fraction
                    )
                    
                    if np.isfinite(cost) and cost > 0:
                        J_envelope[i, k, j] = cost
                        feasible_count += 1
            
            # Progress update
            if k % 10 == 0 or k == K - 1:
                progress = (k + 1) / K * 100
                print(f"  Progress: {progress:.1f}% ({k+1}/{K} altitudes)")
        
        print(f"[ENVELOPE-DESCENT] Computed {feasible_count}/{total_points} feasible points ({100*feasible_count/total_points:.1f}%)")
        
        # Don't transpose - keep as (Mach, Altitude, Lever) for consistency with climb
        return J_envelope, lever_grid

# Create global descent core instance
_descent_core = DescentCore()

# =========  5 - HELPER FUNCTIONS ========================
def extract_descent_initial_state(cruise_results: CruiseResults,
                                  climb_fuel_kg: float,
                                  climb_time_s: float) -> DescentInitialState:
    """
    Extract initial descent state from cruise results.
    
    Args:
        cruise_results: Results from cruise simulation
        climb_fuel_kg: Fuel consumed during climb
        climb_time_s: Time spent in climb
        
    Returns:
        DescentInitialState object with extracted parameters
    """
    # Get final state from cruise
    final_altitude = float(cruise_results.altitude_m[-1])
    final_mach = float(cruise_results.mach_number[-1])
    final_weight = float(cruise_results.weight_kg[-1])
    
    # Total fuel consumed (climb + cruise)
    total_fuel_consumed = climb_fuel_kg + cruise_results.total_fuel_consumed_kg
    
    # Total time (climb + cruise)
    total_time = climb_time_s + cruise_results.total_time_s
    
    print(f"[DESCENT] Extracted initial state:")
    print(f"  Altitude: {final_altitude:.0f} m")
    print(f"  Mach: {final_mach:.3f}")
    print(f"  Weight: {final_weight:.1f} kg")
    print(f"  Total fuel consumed (climb+cruise): {total_fuel_consumed:.1f} kg")
    print(f"  Total time (climb+cruise): {total_time:.0f} s ({total_time/60:.1f} min)")
    
    return DescentInitialState(
        altitude_m=final_altitude,
        mach=final_mach,
        weight_kg=final_weight,
        fuel_consumed_total_kg=total_fuel_consumed,
        total_time_s=total_time
    )

# =========  6 - MAIN INTERFACE ========================
def run_descent_dp_optimization(cruise_results: CruiseResults,
                                climb_fuel_kg: float,
                                climb_time_s: float,
                                aero: AeroTables,
                                engine: EngineWrapper,
                                target_altitude_m: float = None,
                                target_mach: float = None,
                                n_altitude_steps: int = 50,
                                n_mach_samples: int = 81,
                                lever_samples: int = 10) -> Tuple[DescentResults, Dict]:
    """
    Run 3D Dynamic Programming optimization for descent with penalty guidance.
    
    Main interface function similar to climb's solve_3d_fixed_mass.
    
    Args:
        cruise_results: Results from cruise simulation
        climb_fuel_kg: Fuel consumed during climb
        climb_time_s: Time spent in climb
        aero: Aerodynamics tables
        engine: Engine wrapper
        target_altitude_m: Target altitude for descent end (300m for approach)
        target_mach: Target Mach number at final altitude (0.25 for approach)
        n_altitude_steps: Number of altitude steps for DP grid
        n_mach_samples: Number of Mach samples for DP grid
        lever_samples: Number of lever samples for DP grid
        
    Returns:
        Tuple of (DescentResults, info_dict) with optimal descent and optimization info
    """
    if target_altitude_m is None:
        target_altitude_m = 300.0  # Default target altitude
    if target_mach is None:
        target_mach = 0.25  # Default target Mach
    
    print(f"\n{'='*80}")
    print("3D DYNAMIC PROGRAMMING DESCENT OPTIMIZATION (with Penalty Guidance)")
    print(f"{'='*80}")
    print(f"Target: Mach {target_mach:.3f} at {target_altitude_m:.0f}m altitude")
    print(f"Penalty System: Mach guidance + Lever penalties (same as climb)")
    print(f"{'='*80}")
    
    # Extract initial state
    initial_state = extract_descent_initial_state(cruise_results, climb_fuel_kg, climb_time_s)
    
    # Create descent altitude schedule (from high to low)
    H_descent = np.linspace(initial_state.altitude_m, target_altitude_m, n_altitude_steps)
    
    # Calculate dynamic minimum Mach at highest altitude
    min_mach_start = calculate_min_descent_mach(initial_state.altitude_m, initial_state.weight_kg)
    dbg(f"[DP-DESCENT] Dynamic min Mach at start: {min_mach_start:.3f} "
        f"(h={initial_state.altitude_m:.0f}m, W={initial_state.weight_kg:.0f}kg)")
    
    # Create Mach grid (from dynamic minimum to high Mach)
    # Ensure target_mach is in the grid
    M_max = min(0.85, initial_state.mach + 0.05)  # Slightly above cruise Mach
    M_min = max(min_mach_start, target_mach - 0.1)  # Ensure we can reach target
    M_grid = np.linspace(M_min, M_max, n_mach_samples)
    
    dbg(f"[DP-DESCENT] Mach grid: {M_min:.3f} to {M_max:.3f} ({n_mach_samples} samples)")
    
    # Run DP optimization
    dp_result, dp_info = DescentCore.DynamicProgrammingOptimizer.solve_descent_dp(
        aero=aero,
        eng=engine,
        M_grid=M_grid,
        H_sched=H_descent,
        initial_state=initial_state,
        lever_samples=lever_samples,
        target_mach=target_mach,
        target_mach_tolerance=DescentCore.PenaltySystem.TARGET_MACH_TOLERANCE
    )
    
    print(f"{'='*80}")
    print("DP OPTIMIZATION COMPLETED")
    print(f"{'='*80}\n")
    
    return dp_result, dp_info

# =========  4.7 - DESCENT CONFIGURATION ========================
class DescentConfiguration:
    """Configuration constants for descent phase analysis."""
    
    # Descent targets
    TARGET_DESCENT_ALT_M = 300.0        # Approach altitude (~1000 ft)
    TARGET_APPROACH_MACH = 0.25         # Target Mach at approach
    
    # Grid resolution for DP optimization
    DESCENT_MACH_SAMPLES = 81           # Number of Mach samples for DP grid (same as climb MACH_COLS)
    DESCENT_ALT_SAMPLES = 50            # Number of altitude samples for DP grid
    DESCENT_LEVER_SAMPLES = 10          # Number of lever samples for DP grid (same as climb default)
    
    # Speed constraints
    STALL_SPEED_SAFETY_MARGIN = 1.3     # Safety margin above stall speed (1.3 = 30% above stall)
    ABSOLUTE_MIN_DESCENT_MACH = 0.15    # Absolute minimum Mach (safety fallback)
    MAX_DESCENT_MACH = M_MMO            # Maximum Mach for descent

# Create global descent configuration
_descent_config = DescentConfiguration()

# Backward compatibility constants
TARGET_DESCENT_ALT_M = DescentConfiguration.TARGET_DESCENT_ALT_M
TARGET_APPROACH_MACH = DescentConfiguration.TARGET_APPROACH_MACH
MACH_TRAJECTORY_GUIDANCE = DescentCore.PenaltySystem.MACH_TRAJECTORY_GUIDANCE
LEVER_PENALTY_GUIDANCE = DescentCore.PenaltySystem.LEVER_PENALTY_GUIDANCE
TARGET_MACH_TOLERANCE = DescentCore.PenaltySystem.TARGET_MACH_TOLERANCE

# Backward compatibility for old PenaltySystem class
PenaltySystem = DescentCore.PenaltySystem

# Backward compatibility functions
def compute_descent_cost(aero: AeroTables, eng: EngineWrapper, 
                        altitude: float, mach: float, lever: float,
                        mass_kg: float,
                        target_mach: float = None,
                        descent_fraction: float = None) -> float:
    """Backward compatibility wrapper for DescentCore.compute_descent_cost"""
    return DescentCore.compute_descent_cost(aero, eng, altitude, mach, lever, mass_kg, target_mach, descent_fraction)

def compute_mach_penalty(current_mach: float, target_mach: float, prev_mach: float = None, 
                         descent_fraction: float = None) -> float:
    """Backward compatibility wrapper for DescentCore.PenaltySystem.compute_mach_penalty"""
    return DescentCore.PenaltySystem.compute_mach_penalty(current_mach, target_mach, prev_mach, descent_fraction)

def compute_lever_penalty(current_lever: float, descent_fraction: float = None) -> float:
    """Backward compatibility wrapper for DescentCore.PenaltySystem.compute_lever_penalty"""
    return DescentCore.PenaltySystem.compute_lever_penalty(current_lever, descent_fraction)

# Backward compatibility wrappers
def compute_full_descent_envelope(aero, eng, M_grid, H_sched, initial_weight_kg, lever_samples=50, target_mach=None):
    """Backward compatibility wrapper for DescentCore.compute_full_descent_envelope"""
    return DescentCore.compute_full_descent_envelope(aero, eng, M_grid, H_sched, initial_weight_kg, lever_samples, target_mach)

def solve_descent_dp(aero: AeroTables, eng: EngineWrapper,
                    M_grid: np.ndarray, H_sched: np.ndarray,
                    initial_state: DescentInitialState,
                    lever_samples: int = 10,
                    target_mach: float = None,
                    target_mach_tolerance: float = 0.015):
    """Backward compatibility wrapper for DescentCore.DynamicProgrammingOptimizer.solve_descent_dp"""
    return DescentCore.DynamicProgrammingOptimizer.solve_descent_dp(
        aero, eng, M_grid, H_sched, initial_state, lever_samples, target_mach, target_mach_tolerance
    )
