# ========================================================================
# SECTION 1: MODULE INITIALIZATION
# ========================================================================
# Standard imports
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, List, Dict, Any, Tuple

# Aircraft configuration: M limits, lever limits, N_ENGINES, S_ref [m²], atmospheric models
from aircraft_config import (
    a_from_altitude, isa_properties,
    M_MIN_EFFECTIVE, M_MMO, N_ENGINES, S_REF_M2,
    LEVER_MIN, LEVER_MAX
)

# Mission parameters: altitude steps, penalty coefficients, consolidated parameters
from mission_config import (
    N_ALTITUDE_STEPS_DESCENT, N_MACH_SAMPLES_DESCENT, N_LEVER_SAMPLES_DESCENT, TARGET_DESCENT_ALT_M, MIN_DESCENT_MACH,
    PENALTY_DESCENT_MACH_TRAJECTORY_GUIDANCE, PENALTY_DESCENT_LEVER_PENALTY_GUIDANCE,
    PENALTY_DESCENT_TARGET_MACH_TOLERANCE, PENALTY_DESCENT_MACH_PENALTY_BASE_WEIGHT,
    PENALTY_DESCENT_MAX_REASONABLE_MACH_RATE, PENALTY_DESCENT_TOTAL_STEPS_ESTIMATE,
    PENALTY_DESCENT_URGENCY_MULTIPLIER, PENALTY_DESCENT_GUIDANCE_PENALTY_WEIGHT,
    PENALTY_DESCENT_LEVER_PENALTY_WEIGHT, PENALTY_DESCENT_LEVER_PENALTY_THRESHOLD,
    PENALTY_DESCENT_LEVER_PENALTY_EXPONENT, PENALTY_DESCENT_LEVER_PENALTY_CRITICAL_THRESHOLD,
    PENALTY_DESCENT_LEVER_PENALTY_CRITICAL_MULTIPLIER, PENALTY_DESCENT_LEVER_PENALTY_ULTRA_CRITICAL_THRESHOLD,
    PENALTY_DESCENT_LEVER_PENALTY_ULTRA_CRITICAL_MULTIPLIER,
    STALL_SPEED_SAFETY_MARGIN, ABSOLUTE_MIN_DESCENT_MACH,
    MACH_GUIDANCE_FINAL_PHASE_START, MACH_GUIDANCE_FINAL_PHASE_RANGE,
    MACH_GUIDANCE_TERMINAL_PHASE_START, MACH_GUIDANCE_TERMINAL_PHASE_RANGE,
    MACH_GUIDANCE_TERMINAL_BOOST_MULTIPLIER,
    DP_MIN_ALTITUDE_SEGMENT_M, DP_MIN_VELOCITY_CHANGE_MPS,
    DP_MIN_TIME_STEP_S, DP_MIN_SEGMENT_DISTANCE_M,
    DP_PROGRESS_REPORT_INTERVAL
)

# External model interfaces: aerodynamics D(M,h,m) and propulsion T(δ,M,h)
from pyaerodynamics_wrapper import PyAerodynamicsWrapper
from pyengine_wrapper import EngineWrapper

# Cruise phase results for descent initialization
from cruise import CruiseResults

# Utility functions: kinematics, energy, time integration
from mission_utils import (
    find_lever_for_thrust,
    velocity_from_mach,
    mach_from_velocity,
    calculate_stall_mach as calculate_stall_mach_util,
    calculate_specific_excess_power,
    calculate_fuel_flow_rate_safe,
    validate_tsfc,
    thrust_per_engine_to_total,
    calculate_acceleration_rate,
    calculate_time_from_altitude_change,
    calculate_time_from_velocity_change,
    build_time_array_from_segments,
    find_closest_index
)

# ========================================================================
# SECTION 2: DATA STRUCTURES
# ========================================================================

@dataclass
class DescentInitialState:
    """
    Initial state vector for descent phase from cruise endpoint.
    
    State: X_0 = (h_0, M_0, m_0) at cruise termination
    Cumulative quantities: Σm_fuel (climb+cruise), Σt (climb+cruise)
    """
    altitude_m: float                  # h_0 [m]: initial altitude
    mach: float                        # M_0 [-]: initial Mach number
    mass_kg: float                     # m_0 [kg]: aircraft mass at descent start
    fuel_consumed_total_kg: float      # Σm_fuel [kg]: cumulative fuel (climb+cruise)
    total_time_s: float                # Σt [s]: cumulative time (climb+cruise)
    
    def __post_init__(self):
        """Validate state: h > h_target, M ∈ safe range, m > 0."""
        if self.altitude_m <= TARGET_DESCENT_ALT_M:
            raise ValueError(f"Initial altitude {self.altitude_m:.0f}m must exceed target {TARGET_DESCENT_ALT_M}m")
        if not (MIN_DESCENT_MACH <= self.mach <= M_MMO):
            raise ValueError(f"Initial Mach {self.mach:.3f} outside safe range [{MIN_DESCENT_MACH}, {M_MMO}]")
        if self.mass_kg <= 0:
            raise ValueError(f"Mass must be positive: {self.mass_kg:.1f} kg")

@dataclass
class DescentResults:
    """
    Optimal descent trajectory from dynamic programming solution.
    
    State variables: h(t) [m], M(t) [-], δ(t) [-]
    Performance: J [kg/m], Ps [m/s], ṁ [kg/s], T [N], D [N]
    Atmospheric: T [K], ρ [kg/m³], V [m/s]
    """
    # Metadata
    strategy_name: str                          # Strategy identifier
    
    # State trajectory arrays
    alt_m: np.ndarray                           # h(t) [m]: altitude profile
    mach: np.ndarray                            # M(t) [-]: Mach profile
    lever: np.ndarray                           # δ(t) [-]: throttle lever [0,1]
    cumFuel_kg: np.ndarray                      # m_fuel(t) [kg]: cumulative fuel
    dt_s: np.ndarray                            # Δt [s]: time increments
    dFuel_kg: np.ndarray                        # Δm_fuel [kg]: fuel increments
    J_kg_per_m: np.ndarray                      # J [kg/m]: fuel cost density (ṁ/|Ps|)
    
    # Performance arrays
    thrust_total_N: np.ndarray                  # T(t) [N]: total thrust
    drag_N: np.ndarray                          # D(t) [N]: drag
    fuel_flow_kgps: np.ndarray                  # ṁ(t) [kg/s]: fuel flow rate
    descent_rate_mps: np.ndarray                # Ps(t) [m/s]: specific excess power
    temperature_K: np.ndarray                   # T_atm(t) [K]: atmospheric temperature
    density_kgpm3: np.ndarray                   # ρ(t) [kg/m³]: air density
    true_airspeed_mps: np.ndarray               # V(t) [m/s]: true airspeed
    specific_excess_power_mps: np.ndarray       # Ps(t) [m/s]: specific excess power
    
    # Temporal array
    time_s: np.ndarray                          # t [s]: time array
    
    # Mass evolution
    mass_kg: np.ndarray                         # m(t) [kg]: aircraft mass
    
    # Summary statistics
    total_time_s: float                         # t_total [s]: total descent time
    total_fuel_consumed_kg: float               # m_fuel,total [kg]: total fuel consumed
    final_mass_kg: float                        # m_f [kg]: final mass
    average_descent_rate_mps: float             # <Ps> [m/s]: mean descent rate
    average_fuel_flow_kgps: float               # <ṁ> [kg/s]: mean fuel flow
    
    # Boundary conditions
    initial_altitude_m: float                   # h_0 [m]: initial altitude
    initial_mach: float                         # M_0 [-]: initial Mach
    initial_mass_kg: float                      # m_0 [kg]: initial mass
    target_altitude_m: float                    # h_f [m]: target altitude
    target_mach: float                          # M_f [-]: target Mach
    
    def get_summary_dict(self) -> Dict[str, Any]:
        """Extract summary statistics for reporting."""
        return {
            'strategy': self.strategy_name,
            'descent_altitude_change_m': self.initial_altitude_m - self.target_altitude_m,
            'descent_time_hours': self.total_time_s / 3600.0,
            'descent_time_minutes': self.total_time_s / 60.0,
            'descent_fuel_kg': self.total_fuel_consumed_kg,
            'avg_descent_rate_mps': self.average_descent_rate_mps,
            'avg_descent_rate_mpm': self.average_descent_rate_mps * 60.0,
            'avg_fuel_flow_kg_h': self.average_fuel_flow_kgps * 3600.0,
            'initial_altitude_m': self.initial_altitude_m,
            'final_altitude_m': self.target_altitude_m,
            'initial_mass_kg': self.initial_mass_kg,
            'final_mass_kg': self.final_mass_kg,
            'initial_mach': self.initial_mach,
            'final_mach': self.target_mach,
        }

# ========================================================================
# SECTION 3: SYSTEM UTILITIES
# ========================================================================

class SystemUtilities:
    """Diagnostic and logging utilities."""
    
    @staticmethod
    def dbg(msg: str):
        """Debug output method (currently disabled)."""
        pass

def dbg(msg: str):
    """Module-level debug function for backward compatibility."""
    SystemUtilities.dbg(msg)

# ========================================================================
# SECTION 4: STALL CONSTRAINTS
# ========================================================================

def calculate_min_descent_mach(altitude_m: float, mass_kg: float, 
                               cl_max: Optional[float] = None,
                               s_ref_m2: Optional[float] = None,
                               safety_margin: float = None,
                               aero: Optional[PyAerodynamicsWrapper] = None) -> float:
    """
    Compute minimum safe Mach for descent operations.
    
    Formulation: M_min = M_stall · safety_margin
    where M_stall from L = mg at CL_max
    
    Descent-specific bounds: M ∈ [0.15, 0.25] (approach speed range)
    
    Parameters:
        altitude_m: h [m] - altitude
        mass_kg: m [kg] - aircraft mass
        cl_max: CL_max [-] - maximum lift coefficient (default 1.6)
        s_ref_m2: S_ref [m²] - wing reference area (default S_REF_M2)
        safety_margin: factor above stall (default 1.3)
        aero: unused (kept for backward compatibility)
        
    Returns:
        M_min [-]: minimum safe Mach, bounded to [0.15, 0.25]
    """
    try:
        # Compute stall Mach via centralized utility
        m_min = calculate_stall_mach_util(
            mass_kg, altitude_m, 
            cl_max or 1.6, 
            s_ref_m2 or S_REF_M2, 
            safety_margin or 1.3
        )
        # Apply descent-specific bounds (approach speed constraints)
        return float(np.clip(m_min, 0.15, 0.25))
    except Exception as e:
        dbg(f"[DESCENT] Stall Mach calculation failed: {e}, using M_min=0.15")
        return 0.15

# ========================================================================
# SECTION 5: COMPUTATIONAL PRIMITIVES
# ========================================================================

class DescentCore:
    """
    Descent trajectory optimization framework via dynamic programming.
    
    Mathematical Formulation:
        State space: X = (h, M, δ) ∈ ℝ³
        Cost functional: J[X(·)] = ∫ (ṁ/|Ps|) dh  (descent: Ps < 0)
        Optimization: min J subject to flight envelope constraints
    
    Subsystems:
        - DynamicProgrammingOptimizer: 3D Bellman recursion over (h,M,δ)
        - PenaltySystem: Soft constraints via augmented cost J_aug = J + penalties
        - compute_cost: Point-wise cost evaluation J(h,M,δ,m) + penalties
    
    Usage:
        # Compute optimal trajectory
        result, info = DescentCore.DynamicProgrammingOptimizer.solve_3d_dp(
            aero, engine, mach_grid, altitude_sched, initial_state, target_mach=0.25
        )
        
        # Evaluate point cost
        J = DescentCore.compute_cost(aero, engine, h=5000, M=0.6, lever=0.1, mass_kg=70000)
    """
    
    # ────────────────────────────────────────────────────────────────────
    # Penalty Functions for Trajectory Guidance
    # ────────────────────────────────────────────────────────────────────
    
    class PenaltySystem:
        """
        Soft constraint penalties for descent trajectory optimization.
        
        Purpose: Guide DP optimizer toward physically realizable descent trajectories
        via augmented cost J_aug = J + Σ penalties.
        
        Penalty types:
            1. Mach trajectory guidance: ensures reachability of M_target (approach speed)
            2. Lever penalty: discourages sustained high thrust operation
        
        Note: Descent-specific formulation with target M_final < M_initial
        """
        
        # Configuration flags (from mission_config)
        MACH_TRAJECTORY_GUIDANCE = PENALTY_DESCENT_MACH_TRAJECTORY_GUIDANCE
        LEVER_PENALTY_GUIDANCE = PENALTY_DESCENT_LEVER_PENALTY_GUIDANCE
        TARGET_MACH_TOLERANCE = PENALTY_DESCENT_TARGET_MACH_TOLERANCE
        
        # Mach penalty coefficients
        MACH_PENALTY_BASE_WEIGHT = PENALTY_DESCENT_MACH_PENALTY_BASE_WEIGHT
        MAX_REASONABLE_MACH_RATE = PENALTY_DESCENT_MAX_REASONABLE_MACH_RATE
        TOTAL_DESCENT_STEPS_ESTIMATE = PENALTY_DESCENT_TOTAL_STEPS_ESTIMATE
        URGENCY_MULTIPLIER = PENALTY_DESCENT_URGENCY_MULTIPLIER
        GUIDANCE_PENALTY_WEIGHT = PENALTY_DESCENT_GUIDANCE_PENALTY_WEIGHT
        
        # Lever penalty coefficients
        LEVER_PENALTY_WEIGHT = PENALTY_DESCENT_LEVER_PENALTY_WEIGHT
        LEVER_PENALTY_THRESHOLD = PENALTY_DESCENT_LEVER_PENALTY_THRESHOLD
        LEVER_PENALTY_EXPONENT = PENALTY_DESCENT_LEVER_PENALTY_EXPONENT
        LEVER_PENALTY_CRITICAL_THRESHOLD = PENALTY_DESCENT_LEVER_PENALTY_CRITICAL_THRESHOLD
        LEVER_PENALTY_CRITICAL_MULTIPLIER = PENALTY_DESCENT_LEVER_PENALTY_CRITICAL_MULTIPLIER
        LEVER_PENALTY_ULTRA_CRITICAL_THRESHOLD = PENALTY_DESCENT_LEVER_PENALTY_ULTRA_CRITICAL_THRESHOLD
        LEVER_PENALTY_ULTRA_CRITICAL_MULTIPLIER = PENALTY_DESCENT_LEVER_PENALTY_ULTRA_CRITICAL_MULTIPLIER
        
        @staticmethod
        def compute_mach_penalty(current_mach: float, target_mach: float, prev_mach: float = None, 
                                descent_fraction: float = None) -> float:
            """
            Mach trajectory penalty via reachability corridor (descent-specific).
            
            Method: Define corridor [M_min(h), M_max(h)] ensuring M_target is reachable
            with physically reasonable deceleration rates. Apply quadratic penalty outside corridor.
            
            Descent-specific: Target M_final < M_initial (deceleration from cruise to approach)
            
            Corridor bounds:
                ΔM_max = dM/dk_max · k_remaining
                M_min = M_target - ΔM_max
                M_max = M_target + ΔM_max
            
            Penalty structure:
                - M < M_min: P = urgency · w_base · (M_min - M)² (too slow, stall risk)
                - M > M_max: P = urgency · w_base · (M - M_max)² (too fast, won't decelerate)
                - M_min ≤ M ≤ M_max: Progressive guidance toward M_target
            
            Parameters:
                current_mach: M [-] - current Mach
                target_mach: M_target [-] - terminal approach Mach (typically 0.25)
                prev_mach: unused (kept for API compatibility)
                descent_fraction: h/h_total [-] - descent progress ∈ [0,1]
            
            Returns:
                penalty [kg/m]: Mach deviation penalty
            """
            if descent_fraction is None:
                descent_fraction = 0.0
            
            # Remaining descent fraction: ξ_rem = 1 - progress
            remaining_fraction = 1.0 - descent_fraction
            estimated_steps_remaining = remaining_fraction * DescentCore.PenaltySystem.TOTAL_DESCENT_STEPS_ESTIMATE
            
            # Maximum reachable Mach deviation: ΔM_max = dM/dk_max · k_rem
            max_achievable_change = DescentCore.PenaltySystem.MAX_REASONABLE_MACH_RATE * estimated_steps_remaining
            
            # Reachability corridor: [M_min, M_max]
            # For descent: target M_final < M_initial, so corridor surrounds target
            min_reachable_mach = target_mach - max_achievable_change
            max_reachable_mach = target_mach + max_achievable_change
            
            # Urgency factor: increases as h → h_target
            urgency = (1.0 - remaining_fraction) * DescentCore.PenaltySystem.URGENCY_MULTIPLIER
            
            # Penalty computation based on corridor position
            if current_mach < min_reachable_mach:
                # Too slow: stall risk, cannot reach target
                deviation = min_reachable_mach - current_mach
                penalty = urgency * DescentCore.PenaltySystem.MACH_PENALTY_BASE_WEIGHT * (deviation ** 2)
                
            elif current_mach > max_reachable_mach:
                # Too fast: insufficient deceleration distance to reach M_target
                deviation = current_mach - max_reachable_mach  
                penalty = urgency * DescentCore.PenaltySystem.MACH_PENALTY_BASE_WEIGHT * (deviation ** 2)
                
            else:
                # Within corridor: apply progressive guidance
                if descent_fraction > MACH_GUIDANCE_FINAL_PHASE_START:
                    # Final phase: strong convergence to M_target
                    final_phase_strength = (descent_fraction - MACH_GUIDANCE_FINAL_PHASE_START) / MACH_GUIDANCE_FINAL_PHASE_RANGE  # ξ ∈ [0,1]
                    mach_deviation = current_mach - target_mach
                    
                    # Terminal phase boost: extra convergence
                    if descent_fraction > MACH_GUIDANCE_TERMINAL_PHASE_START:
                        final_boost = ((descent_fraction - MACH_GUIDANCE_TERMINAL_PHASE_START) / MACH_GUIDANCE_TERMINAL_PHASE_RANGE) * MACH_GUIDANCE_TERMINAL_BOOST_MULTIPLIER
                        final_phase_strength *= (1.0 + final_boost)
                        
                    penalty = final_phase_strength * DescentCore.PenaltySystem.GUIDANCE_PENALTY_WEIGHT * (mach_deviation ** 2)
                else:
                    penalty = 0.0  # Early phase: no guidance penalty
            
            return penalty
        
        @staticmethod
        def compute_lever_penalty(current_lever: float, descent_fraction: float = None) -> float:
            """
            Lever penalty to discourage sustained high thrust operation.
            
            Physical basis: Engine thermal and mechanical limits are altitude-independent.
            High thrust operation increases wear, fuel consumption, and operational costs.
            
            Thrust regimes:
                δ ≤ 0.85: Maximum Continuous Thrust (MCT) - unlimited duration, no penalty
                0.85 < δ ≤ 0.90: Climb thrust - time-limited, moderate penalty
                0.90 < δ ≤ 0.95: Maximum climb thrust - high wear, significant penalty
                δ > 0.95: Emergency thrust - severe penalty
            
            Penalty structure: P(δ) = w · [(δ - δ_MCT)^p + critical terms]
            where p = exponent, δ_MCT = MCT threshold
            
            Parameters:
                current_lever: δ [-] - throttle position ∈ [0,1]
                descent_fraction: unused (kept for API compatibility)
            
            Returns:
                penalty [kg/m]: lever position penalty (altitude-independent)
            """
            penalty = 0.0
            
            # Apply penalty only if δ > δ_MCT
            if current_lever > DescentCore.PenaltySystem.LEVER_PENALTY_THRESHOLD:
                # Excess thrust: Δδ = δ - δ_MCT
                excess_lever = current_lever - DescentCore.PenaltySystem.LEVER_PENALTY_THRESHOLD
                
                # Base penalty: P_base = (Δδ)^p
                lever_penalty = excess_lever ** DescentCore.PenaltySystem.LEVER_PENALTY_EXPONENT
                
                # Critical regime (δ > 0.90): additional quadratic term
                if current_lever > DescentCore.PenaltySystem.LEVER_PENALTY_CRITICAL_THRESHOLD:
                    critical_excess = current_lever - DescentCore.PenaltySystem.LEVER_PENALTY_CRITICAL_THRESHOLD
                    critical_penalty = critical_excess ** (DescentCore.PenaltySystem.LEVER_PENALTY_EXPONENT + 1.0)
                    lever_penalty += critical_penalty * DescentCore.PenaltySystem.LEVER_PENALTY_CRITICAL_MULTIPLIER
                
                # Ultra-critical regime (δ > 0.95): emergency thrust penalty
                if current_lever > DescentCore.PenaltySystem.LEVER_PENALTY_ULTRA_CRITICAL_THRESHOLD:
                    ultra_critical_excess = current_lever - DescentCore.PenaltySystem.LEVER_PENALTY_ULTRA_CRITICAL_THRESHOLD
                    ultra_critical_penalty = ultra_critical_excess ** (DescentCore.PenaltySystem.LEVER_PENALTY_EXPONENT + 2.0)
                    lever_penalty += ultra_critical_penalty * DescentCore.PenaltySystem.LEVER_PENALTY_ULTRA_CRITICAL_MULTIPLIER
                
                # Scale by constant weight (altitude-independent)
                penalty_weight = DescentCore.PenaltySystem.LEVER_PENALTY_WEIGHT
                penalty = penalty_weight * lever_penalty
            
            return penalty
    
    # ────────────────────────────────────────────────────────────────────
    # Dynamic Programming Optimization (Bellman Recursion)
    # ────────────────────────────────────────────────────────────────────
    
    class DynamicProgrammingOptimizer:
        """
        3D dynamic programming solver for minimum-fuel descent trajectories.
        
        Formulation:
            State: X_k = (h_k, M_k, δ_k) at altitude level k
            Cost-to-go: F[k, i, j] = min fuel from X_k to target
            Recursion: F[k+1, i', j'] = min_{i,j} {F[k, i, j] + J(X_k→X_{k+1}) · |Δh|}
            Transition kernel: 7×7 grid in (M, δ) space at each altitude step
            
        Descent-specific: Ps < 0 (energy dissipation), h_k > h_{k+1} (descending)
        """
        
        @staticmethod
        def solve_3d_dp(aero: PyAerodynamicsWrapper, engine: EngineWrapper,
                       mach_grid: np.ndarray, altitude_sched: np.ndarray,
                       initial_state: DescentInitialState,
                       lever_samples: int = N_LEVER_SAMPLES_DESCENT,
                       target_mach: float = None,
                       target_mach_tolerance: float = PENALTY_DESCENT_TARGET_MACH_TOLERANCE):
            """
            Solve 3D Bellman equation for optimal descent trajectory.
            
            Algorithm:
                1. Initialize: F[0, M_cruise, δ_idle] = 0
                2. Forward pass: Compute F[k+1] from F[k] via Bellman recursion
                3. Terminal constraint: Enforce |M_final - M_target| < tol
                4. Backtrack: Recover optimal path from predecessor matrix
                5. Post-process: Compute time, fuel, and performance arrays
            
            Parameters:
                aero: PyAerodynamicsWrapper - drag model D(M,h,m)
                engine: EngineWrapper - thrust model T(δ,M,h)
                mach_grid: np.ndarray - M_i, i=1..I (Mach discretization)
                altitude_sched: np.ndarray - h_k, k=1..K (descending: h_k > h_{k+1})
                initial_state: DescentInitialState - X_0 from cruise
                lever_samples: int - number of throttle positions L
                target_mach: float - M_target at h_final (default 0.25)
                target_mach_tolerance: float - terminal Mach tolerance (default 0.015)
            
            Returns:
                DescentResults: optimal trajectory X*(k)
                dict: optimization metadata (costs, path length, time)
            """
            if target_mach is None:
                target_mach = 0.25  # Default approach Mach
            
            print("[DESCENT] Starting 3D DP optimization with penalty guidance")
            print(f"[DESCENT] Target: M={target_mach:.3f} at h={altitude_sched[-1]:.0f}m")
            print(f"[DESCENT] Grid: {len(mach_grid)} Mach × {len(altitude_sched)} Alt × {lever_samples} Lever")
            
            # Grid dimensions: K altitude levels, I Mach points, L lever positions
            K, I = len(altitude_sched), len(mach_grid)
            L = lever_samples
            
            # Throttle discretization: δ ∈ [δ_min, δ_max]
            lever_grid = np.linspace(LEVER_MIN, LEVER_MAX, L)
            
            # Initialize DP arrays
            F = np.full((K, I, L), np.inf)              # F[k,i,j]: cost-to-go [kg]
            weight_matrix = np.full((K, I, L), np.nan)  # m[k,i,j]: mass [kg]
            prv = np.full((K, I, L, 3), -1, dtype=int)  # predecessor[k,i,j] = [k',i',j']
            
            # Determine initial state indices from cruise endpoint
            start_mach_idx = find_closest_index(initial_state.mach, mach_grid)
            start_lever_idx = 0  # Initialize at idle thrust
            actual_start_mach = mach_grid[start_mach_idx]
            
            dbg(f"[DP-DESCENT] Initial state: h_0={altitude_sched[0]:.0f}m, M_0={actual_start_mach:.3f}, m_0={initial_state.mass_kg:.0f}kg")
            
            # Validate initial state X_0 = (h_0, M_0, δ_0)
            if (actual_start_mach >= M_MIN_EFFECTIVE and 
                actual_start_mach <= M_MMO):
                F[0, start_mach_idx, start_lever_idx] = 0.0  # Boundary condition: F[0] = 0
                weight_matrix[0, start_mach_idx, start_lever_idx] = initial_state.mass_kg
                dbg(f"[DP-DESCENT] Initial state validated: M_0={actual_start_mach:.3f}, m_0={initial_state.mass_kg:.0f}kg")
            else:
                raise RuntimeError(f"[DP-DESCENT] Initial state violates bounds: M={actual_start_mach:.3f}")
            
            # ================================================================
            # Forward Pass: Bellman Recursion (Descending)
            # ================================================================
            # Compute F[k+1] from F[k] for k = 0, 1, ..., K-2
            for k in range(K - 1):
                current_alt = altitude_sched[k]      # h_k [m]
                next_alt = altitude_sched[k + 1]     # h_{k+1} [m] (< h_k)
                dh = next_alt - current_alt          # Δh [m] (negative)
                
                # Descent progress for penalty system
                descent_fraction = k / (K - 1.0) if K > 1 else 0.0
                
                if k % DP_PROGRESS_REPORT_INTERVAL == 0:
                    dbg(f"[DP-DESCENT] Altitude level k={k}/{K-1}: h={current_alt:.0f}m → {next_alt:.0f}m (progress: {descent_fraction*100:.1f}%)")
                
                # Identify feasible states at current level: F[k,i,j] < ∞
                feasible_states = np.where(np.isfinite(F[k]))
                feasible_count = 0
                
                for state_idx in range(len(feasible_states[0])):
                    i = feasible_states[0][state_idx]  # Mach index i
                    j = feasible_states[1][state_idx]  # Lever index j
                    
                    if not np.isfinite(F[k, i, j]):
                        continue
                    
                    # Current state: X_k = (h_k, M_i, δ_j)
                    current_weight = weight_matrix[k, i, j]  # m_k [kg]
                    if not np.isfinite(current_weight) or current_weight <= 0:
                        continue
                    
                    current_mach = mach_grid[i]      # M_i [-]
                    current_lever = lever_grid[j]    # δ_j [-]
                    
                    # ────────────────────────────────────────────────────────
                    # Transition Kernel: Explore 7×7 neighborhood in (M, δ)
                    # ────────────────────────────────────────────────────────
                    # Transitions: (h_k, M_i, δ_j) → (h_{k+1}, M_{i±Δi}, δ_{j±Δj})
                    # Range: Δi, Δj ∈ {-3, -2, -1, 0, +1, +2, +3} → 49 candidates
                    for di in [-3, -2, -1, 0, 1, 2, 3]:  # Mach index offset
                        for dj in [-3, -2, -1, 0, 1, 2, 3]:  # Lever index offset
                            next_mach_idx = i + di
                            next_lever_idx = j + dj
                            
                            # Bounds check: indices must be valid and transition only to k+1
                            if (0 <= next_mach_idx < I and 
                                0 <= next_lever_idx < L and
                                k + 1 < K):
                                
                                next_mach = mach_grid[next_mach_idx]    # M_{i'} [-]
                                next_lever = lever_grid[next_lever_idx]  # δ_{j'} [-]
                                
                                # Dynamic stall constraint: M ≥ M_stall(h, m)
                                min_mach_next = calculate_min_descent_mach(next_alt, current_weight, aero=aero)
                                
                                # Flight envelope constraints: M ∈ [M_stall, M_MMO], δ ∈ [δ_min, δ_max]
                                if (next_mach >= min_mach_next and 
                                    next_mach <= M_MMO and
                                    next_lever >= LEVER_MIN and 
                                    next_lever <= LEVER_MAX):
                                    
                                    # ────────────────────────────────────────────────────────
                                    # Cost Computation with Mass Coupling
                                    # ────────────────────────────────────────────────────────
                                    # Descent fraction for penalty system
                                    descent_fraction_curr = k / (K - 1.0) if K > 1 else 0.0
                                    descent_fraction_next = (k + 1) / (K - 1.0) if K > 1 else 1.0
                                    
                                    # Current state cost: J_k = J(h_k, M_i, δ_j, m_k)
                                    current_cost = DescentCore.compute_cost(
                                        aero, engine, current_alt, current_mach, current_lever,
                                        current_weight, target_mach, descent_fraction_curr
                                    )
                                    
                                    if not (np.isfinite(current_cost) and current_cost > 0):
                                        continue
                                    
                                    # Next state cost (initial): J_{k+1}^(0) using m_k
                                    next_cost_initial = DescentCore.compute_cost(
                                        aero, engine, next_alt, next_mach, next_lever,
                                        current_weight, target_mach, descent_fraction_next
                                    )
                                    
                                    if not (np.isfinite(next_cost_initial) and next_cost_initial > 0):
                                        continue
                                    
                                    # Fuel burn estimate (trapezoidal rule): Δm = (J_k + J_{k+1})/2 · |Δh|
                                    step_cost_initial = 0.5 * (current_cost + next_cost_initial) * abs(dh)
                                    fuel_burned_initial = step_cost_initial
                                    
                                    # Mass update: m_{k+1} = m_k - Δm
                                    next_weight = current_weight - fuel_burned_initial
                                    if next_weight <= 0:
                                        continue
                                    
                                    # Cost refinement: Recompute J_{k+1} with updated mass m_{k+1}
                                    # Accounts for Ps ∝ 1/m → J = ṁ/|Ps| ∝ m
                                    next_cost_refined = DescentCore.compute_cost(
                                        aero, engine, next_alt, next_mach, next_lever,
                                        next_weight, target_mach, descent_fraction_next
                                    )
                                    
                                    if not (np.isfinite(next_cost_refined) and next_cost_refined > 0):
                                        next_cost = next_cost_initial  # Fallback
                                    else:
                                        next_cost = next_cost_refined
                                    
                                    # Final step cost with refined J_{k+1}
                                    step_cost = 0.5 * (current_cost + next_cost) * abs(dh)
                                    total_cost = F[k, i, j] + step_cost
                                    
                                    # Final mass update
                                    fuel_burned = step_cost
                                    next_weight = current_weight - fuel_burned
                                    if next_weight <= 0:
                                        continue
                                    
                                    # ────────────────────────────────────────────────────────
                                    # Bellman Update: F[k+1] = min{F[k] + cost}
                                    # ────────────────────────────────────────────────────────
                                    if total_cost < F[k + 1, next_mach_idx, next_lever_idx]:
                                        F[k + 1, next_mach_idx, next_lever_idx] = total_cost
                                        weight_matrix[k + 1, next_mach_idx, next_lever_idx] = next_weight
                                        prv[k + 1, next_mach_idx, next_lever_idx] = [k, i, j]
                                        feasible_count += 1
                
                if feasible_count == 0 and k < K - 2:
                    dbg(f"[DP-DESCENT] WARNING: No feasible transitions at level k={k}")
            
            # ================================================================
            # Terminal Constraint: |M_final - M_target| < tolerance
            # ================================================================
            dbg(f"[DP-DESCENT] Enforcing terminal constraint: M_final ∈ [{target_mach - target_mach_tolerance:.3f}, {target_mach + target_mach_tolerance:.3f}]")
            
            # Identify valid terminal Mach indices
            valid_final = np.abs(mach_grid - target_mach) < target_mach_tolerance
            
            if not valid_final.any():
                dbg(f"[DP-DESCENT] Warning: No Mach within tolerance, using closest")
                closest_idx = find_closest_index(target_mach, mach_grid)
                valid_final = np.zeros_like(valid_final, dtype=bool)
                valid_final[closest_idx] = True
            
            # Mask infeasible terminal states: F[K-1, i, :] = ∞ for i ∉ valid_final
            for i in range(I):
                if not valid_final[i]:
                    F[-1, i, :] = np.inf
            
            # Verify solution existence
            if not np.isfinite(F[-1]).any():
                raise RuntimeError("[DP-DESCENT] No feasible path to terminal altitude")
            
            # ================================================================
            # Optimal Terminal State: X* = argmin F[K-1]
            # ================================================================
            final_flat_idx = np.nanargmin(F[-1])
            final_mach_idx, final_lever_idx = np.unravel_index(final_flat_idx, F[-1].shape)
            final_alt_idx = K - 1
            
            final_mach = mach_grid[final_mach_idx]
            final_lever = lever_grid[final_lever_idx]
            final_alt = altitude_sched[final_alt_idx]
            final_cost = F[final_alt_idx, final_mach_idx, final_lever_idx]
            
            dbg(f"[DP-DESCENT] Optimal terminal state:")
            dbg(f"  h_f={final_alt:.0f}m, M_f={final_mach:.3f}, δ_f={final_lever:.3f}")
            dbg(f"  Minimum fuel: {final_cost:.2f} kg")
            
            # ================================================================
            # Backtracking: Recover Optimal Path via Predecessor Chain
            # ================================================================
            # Start from X*[K-1] and follow predecessor links to X*[0]
            path_alt = []
            path_mach = []
            path_lever = []
            path_costs = []
            path_weights = []
            
            current_state = [final_alt_idx, final_mach_idx, final_lever_idx]
            
            while current_state[0] >= 0:
                alt_idx, mach_idx, lever_idx = current_state
                
                # Append current state to path (reverse order)
                path_alt.append(altitude_sched[alt_idx])
                path_mach.append(mach_grid[mach_idx])
                path_lever.append(lever_grid[lever_idx])
                path_costs.append(F[alt_idx, mach_idx, lever_idx])
                path_weights.append(weight_matrix[alt_idx, mach_idx, lever_idx])
                
                # Follow predecessor link
                if alt_idx > 0:
                    current_state = prv[alt_idx, mach_idx, lever_idx].tolist()
                else:
                    break
            
            # Reverse to chronological order: X*[0] → X*[1] → ... → X*[K-1]
            path_alt = path_alt[::-1]
            path_mach = path_mach[::-1]
            path_lever = path_lever[::-1]
            path_costs = path_costs[::-1]
            path_weights = path_weights[::-1]
            
            # Convert to arrays
            alt_array = np.array(path_alt)        # h[k] [m]
            mach_array = np.array(path_mach)      # M[k] [-]
            lever_array = np.array(path_lever)    # δ[k] [-]
            fuel_array = np.array(path_costs)     # Cumulative cost [kg]
            weight_array = np.array(path_weights)  # m[k] [kg]
            
            # Calculate detailed trajectory data
            n_points = len(alt_array)
            time_array = np.zeros(n_points)
            dt_array = np.zeros(n_points)
            # Note: dFuel_array will be calculated during reconstruction with consistent method
            
            thrust_array = np.zeros(n_points)
            drag_array = np.zeros(n_points)
            fuel_flow_array = np.zeros(n_points)
            descent_rate_array = np.zeros(n_points)
            T_array = np.zeros(n_points)
            rho_array = np.zeros(n_points)
            V_array = np.zeros(n_points)
            Ps_array = np.zeros(n_points)
            J_cost_array = np.zeros(n_points)  # J [kg/m]: fuel cost density
            # weight_array already set from backtracking with dynamic mass values
            
            # Calculate time increments and performance data with enhanced accuracy
            n_segments = n_points - 1  # Number of segments between points
            dt_segment_array = np.zeros(n_segments)  # Time for each segment
            
            for i in range(n_points):
                h = alt_array[i]
                M = mach_array[i]
                lever = lever_array[i]
                
                # Atmospheric properties
                temp_K, pressure_Pa, density_kgpm3 = isa_properties(h)
                V = velocity_from_mach(M, h)
                
                T_array[i] = temp_K  # Temperature in Kelvin
                rho_array[i] = density_kgpm3
                V_array[i] = V
                
                # Thrust and drag
                T_per = engine.thrust_with_lever(lever, M, h)
                if T_per is None or T_per < 0:
                    T_per = 0.0
                T_tot = T_per * N_ENGINES
                D = aero.get_drag(M, h, weight_array[i])
                
                thrust_array[i] = T_tot
                drag_array[i] = D
                
                # Fuel flow with safety guards
                tsfc = validate_tsfc(engine.tsfc_current(), fallback=0.0)
                mdot = calculate_fuel_flow_rate_safe(tsfc, T_per, N_ENGINES)
                fuel_flow_array[i] = mdot
                
                # Calculate Ps and descent rate using centralized function
                Ps = calculate_specific_excess_power(T_tot, D, weight_array[i], V)
                Ps_array[i] = Ps
                descent_rate_array[i] = Ps
                
                # Calculate J cost density: J = ṁ/|Ps| [kg/m]
                descent_fraction_i = i / (n_points - 1.0) if n_points > 1 else 0.0
                J_cost_i = DescentCore.compute_cost(
                    aero, engine, h, M, lever,
                    weight_array[i], target_mach, descent_fraction_i
                )
                J_cost_array[i] = J_cost_i if np.isfinite(J_cost_i) and J_cost_i > 0 else 0.0
            
            # Enhanced time and fuel calculation for segments (consistent method, similar to climb.py)
            dFuel_segment_array = np.zeros(n_segments)  # Fuel for each segment
            
            for i in range(n_segments):
                h_curr, h_next = alt_array[i], alt_array[i + 1]
                M_curr, M_next = mach_array[i], mach_array[i + 1]
                lever_curr, lever_next = lever_array[i], lever_array[i + 1]
                weight_curr, weight_next = weight_array[i], weight_array[i + 1]
                
                # Calculate altitude difference for segment
                dh = abs(h_next - h_curr) if abs(h_next - h_curr) > DP_MIN_ALTITUDE_SEGMENT_M else 0.0
                
                # Calculate descent fraction for penalty system
                descent_fraction_curr = i / (n_points - 1.0) if n_points > 1 else 0.0
                descent_fraction_next = (i + 1) / (n_points - 1.0) if n_points > 1 else 1.0
                
                # Calculate current cost with current mass (consistent with forward pass)
                current_cost = DescentCore.compute_cost(
                    aero, engine, h_curr, M_curr, lever_curr,
                    weight_curr, target_mach, descent_fraction_curr
                )
                
                if not (np.isfinite(current_cost) and current_cost > 0):
                    dt_segment_array[i] = 0.0
                    dFuel_segment_array[i] = 0.0
                    continue
                
                # Cost calculation using DP mass values (weight_next from DP forward pass)
                # This ensures consistency with DP optimization where weight_next was
                # calculated considering dynamic drag D(M,h,m) effects
                next_cost = DescentCore.compute_cost(
                    aero, engine, h_next, M_next, lever_next,
                    weight_next, target_mach, descent_fraction_next
                )
                
                if not (np.isfinite(next_cost) and next_cost > 0):
                    dt_segment_array[i] = 0.0
                    dFuel_segment_array[i] = 0.0
                    continue
                
                # Fuel burn calculation via trapezoidal rule using DP mass values
                # Δm = (J_k + J_{k+1})/2 · |Δh| where J evaluated at DP-optimized mass values
                if dh > DP_MIN_ALTITUDE_SEGMENT_M:  # Vertical move (altitude change)
                    fuel_burned = 0.5 * (current_cost + next_cost) * dh
                else:  # Horizontal move - use small distance approximation
                    # For horizontal moves, approximate distance from velocity change
                    a = a_from_altitude(h_curr)
                    V_curr = M_curr * a
                    V_next = M_next * a
                    ds = 0.5 * (V_curr + V_next) * DP_MIN_TIME_STEP_S if abs(V_next - V_curr) > DP_MIN_VELOCITY_CHANGE_MPS else DP_MIN_SEGMENT_DISTANCE_M
                    fuel_burned = 0.5 * (current_cost + next_cost) * ds
                
                dFuel_segment_array[i] = fuel_burned
                
                # Use DP mass values for time calculation (no recalculation needed)
                weight_avg = 0.5 * (weight_curr + weight_next)
                
                # Calculate time for this segment using DP mass values
                if dh > DP_MIN_ALTITUDE_SEGMENT_M:  # Vertical move (altitude change)
                    h_avg = 0.5 * (h_curr + h_next)
                    M_avg = 0.5 * (M_curr + M_next)
                    lever_avg = 0.5 * (lever_curr + lever_next)
                    V_avg = velocity_from_mach(M_avg, h_avg)
                    D_avg = aero.get_drag(M_avg, h_avg, weight_avg)
                    T_per_avg = engine.thrust_with_lever(lever_avg, M_avg, h_avg)
                    if T_per_avg is None or T_per_avg < 0:
                        T_per_avg = 0.0
                    T_tot_avg = T_per_avg * N_ENGINES
                    Ps_avg = calculate_specific_excess_power(T_tot_avg, D_avg, weight_avg, V_avg)
                    
                    # Calculate time from altitude change using centralized function
                    dt_segment_array[i] = calculate_time_from_altitude_change(dh, Ps_avg)
                    if dt_segment_array[i] > 0 and abs(Ps_avg) > 0.1:
                        dbg(f"[DP-DESCENT] Vertical move {i}: h={h_curr:.0f}->{h_next:.0f}m, dt={dt_segment_array[i]:.3f}s, fuel={fuel_burned:.3f}kg, weight={weight_avg:.0f}kg")
                    else:
                        if dt_segment_array[i] == 0:
                            dt_segment_array[i] = 1.0
                            dFuel_segment_array[i] = 0.0
                        dbg(f"[DP-DESCENT] Vertical move {i}: h={h_curr:.0f}->{h_next:.0f}m, dt={dt_segment_array[i]:.3f}s, fuel={fuel_burned:.3f}kg (low Ps)")
                else:  # Horizontal move (same altitude, different Mach/lever)
                    # Calculate time based on velocity change using DP mass values
                    V_curr = velocity_from_mach(M_curr, h_curr)
                    V_next = velocity_from_mach(M_next, h_curr)
                    
                    if abs(V_next - V_curr) > 0.1:  # Significant velocity change
                        # Use deceleration rate: dt = dV / a_decel with DP average mass
                        D = aero.get_drag(M_curr, h_curr, weight_avg)
                        T_per = engine.thrust_with_lever(lever_curr, M_curr, h_curr)
                        if T_per is None or T_per < 0:
                            T_per = 0.0
                        T_tot = thrust_per_engine_to_total(T_per, N_ENGINES)
                        a_decel = calculate_acceleration_rate(T_tot, D, weight_avg)
                        dt_segment_array[i] = calculate_time_from_velocity_change(V_next - V_curr, a_decel)
                        
                        if dt_segment_array[i] > 0 and abs(a_decel) > 0.1:
                            dbg(f"[DP-DESCENT] Horizontal move {i}: M={M_curr:.3f}->{M_next:.3f}, V={V_curr:.1f}->{V_next:.1f}m/s, dt={dt_segment_array[i]:.3f}s, fuel={fuel_burned:.3f}kg")
                        else:
                            dt_segment_array[i] = 0.1
                            dbg(f"[DP-DESCENT] Horizontal move {i}: M={M_curr:.3f}->{M_next:.3f}, dt={dt_segment_array[i]:.3f}s, fuel={fuel_burned:.3f}kg (small step)")
                    else:
                        dt_segment_array[i] = 0.1
                        dbg(f"[DP-DESCENT] Horizontal move {i}: M={M_curr:.3f}->{M_next:.3f}, dt={dt_segment_array[i]:.3f}s, fuel={fuel_burned:.3f}kg (minimal change)")
                
                # Debug the final segment
                if i >= n_segments - 2:  # Last two segments
                    dbg(f"[DP-DESCENT] Segment {i} (step {i}->{i+1}): h={h_curr:.0f}->{h_next:.0f}m, dt={dt_segment_array[i]:.3f}s, fuel={dFuel_segment_array[i]:.3f}kg")
            
            # Build time array from segments using centralized function
            time_array, dt_array = build_time_array_from_segments(dt_segment_array, n_points)
            
            # Create dFuel_array from segments (consistent with time calculation)
            dFuel_array = np.zeros(n_points)
            dFuel_array[1:] = dFuel_segment_array  # Fuel consumed in each segment
            dFuel_array[0] = 0.0  # First point has no fuel consumed
            
            # Recalculate fuel_array from dFuel_array using cumsum (consistent with mass calculation)
            # This ensures fuel_array matches the recalculated fuel segments, not the DP cost matrix
            fuel_array = np.cumsum(dFuel_array)
            
            dbg(f"[DP-DESCENT] Enhanced time array constructed: start=0, end={time_array[-1]:.3f}s")
            dbg(f"[DP-DESCENT] Sample time progression: {time_array[:5]} (first 5 points)")
            dbg(f"[DP-DESCENT] Final time progression: {time_array[-3:]} (last 3 points)")
            dbg(f"[DP-DESCENT] Final dt values: {dt_segment_array[-3:]} (last 3 segments)")
            
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
                J_kg_per_m=J_cost_array,  # J [kg/m]: fuel cost density
                thrust_total_N=thrust_array,
                drag_N=drag_array,
                fuel_flow_kgps=fuel_flow_array,
                descent_rate_mps=descent_rate_array,
                temperature_K=T_array,
                density_kgpm3=rho_array,
                true_airspeed_mps=V_array,
                specific_excess_power_mps=Ps_array,
                time_s=time_array,
                mass_kg=weight_array,  # Renamed for physics accuracy
                total_time_s=total_time,
                total_fuel_consumed_kg=total_fuel,
                final_mass_kg=final_weight,  # Renamed for physics accuracy
                average_descent_rate_mps=avg_descent_rate,
                average_fuel_flow_kgps=avg_fuel_flow,
                initial_altitude_m=initial_state.altitude_m,
                initial_mach=initial_state.mach,
                initial_mass_kg=initial_state.mass_kg,  # Renamed for physics accuracy
                target_altitude_m=altitude_sched[-1],
                target_mach=target_mach
            )
            
            info = {
                'total_fuel_kg': float(total_fuel),
                'total_time_s': float(total_time),
                'total_time_min': float(total_time / 60.0),
                'final_mach': float(final_mach),
                'final_altitude': float(final_alt),
                'target_mach': target_mach,
                'target_altitude': altitude_sched[-1],
                'mach_deviation': abs(final_mach - target_mach),
                'path_length': len(alt_array),
                'cost_matrix_3d': F,
                'predecessor_matrix': prv,
                'avg_descent_rate_mps': avg_descent_rate,
                'avg_descent_rate_mpm': avg_descent_rate * 60.0,
            }
            
            print("[DESCENT] Descent optimization completed")
            print(f"  Total fuel: {total_fuel:.2f} kg")
            print(f"  Total time: {total_time/60:.1f} min")
            print(f"  Final Mach: {final_mach:.3f} (target: {target_mach:.3f}, deviation: {abs(final_mach-target_mach):.4f})")
            print(f"  Average descent rate: {avg_descent_rate:.2f} m/s ({avg_descent_rate*60.0:.0f} m/min)")
            
            return descent_result, info
        
        @staticmethod
        def solve_descent_dp(*args, **kwargs):
            """Deprecated: Use solve_3d_dp() instead. Maintained for backward compatibility."""
            import warnings
            warnings.warn("solve_descent_dp() is deprecated, use solve_3d_dp() instead", DeprecationWarning, stacklevel=2)
            return DescentCore.DynamicProgrammingOptimizer.solve_3d_dp(*args, **kwargs)

    # ────────────────────────────────────────────────────────────────────
    # Cost Evaluation
    # ────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def compute_cost(aero: PyAerodynamicsWrapper, engine: EngineWrapper,
                    altitude: float, mach: float, lever: float,
                    mass_kg: float,
                    target_mach: float = None,
                    descent_fraction: float = None) -> float:
        """
        Evaluate augmented fuel cost density at state (h, M, δ) for descent.
        
        Formulation: J_aug = J + penalties
        where J = ṁ/|Ps| is fuel consumption per unit altitude descent.
        
        Physical relations (descent-specific):
            Ps = (T-D)V/m < 0  [m/s] - specific excess power (negative for descent)
            ṁ = TSFC · T  [kg/s] - fuel flow rate
            J = ṁ/|Ps|  [kg/m] - fuel cost density per meter descended
        
        Feasibility: Requires Ps < 0 for descent capability
        
        Penalties (optional):
            - Mach trajectory guidance: guides toward target M_final (approach speed)
            - Lever penalty: discourages high thrust settings
        
        Parameters:
            aero: PyAerodynamicsWrapper - drag model
            engine: EngineWrapper - thrust and TSFC model
            altitude: h [m] - altitude
            mach: M [-] - Mach number
            lever: δ [-] - throttle position ∈ [0,1]
            mass_kg: m [kg] - aircraft mass
            target_mach: M_target [-] - terminal approach Mach (optional, default 0.25)
            descent_fraction: progress ∈ [0,1] for adaptive penalties (optional)
        
        Returns:
            J_aug [kg/m]: augmented fuel cost density, or ∞ if infeasible
        """
        if target_mach is None:
            target_mach = 0.25  # Default approach Mach
        
        try:
            # Kinematics: V = M · a(h)
            V = velocity_from_mach(mach, altitude)
            
            # Propulsion: T = T(δ, M, h)
            T_per = engine.thrust_with_lever(lever, mach, altitude)
            if T_per is None or not np.isfinite(T_per) or T_per < 0:
                return np.inf
            T_tot = T_per * N_ENGINES
            
            # Aerodynamics: D = D(M, h, m)
            D = aero.get_drag(mach, altitude, mass_kg)
            if not np.isfinite(D) or D < 0:
                return np.inf
            
            # Specific excess power: Ps = (T-D)V/m
            Ps = calculate_specific_excess_power(T_tot, D, mass_kg, V)
            
            # Descent feasibility: Ps < 0 (energy dissipation required)
            if Ps >= 0:  # Cannot descend with positive/zero Ps
                return np.inf
            
            # Fuel flow: ṁ = TSFC · T
            tsfc = engine.tsfc_current()
            if tsfc is None or not np.isfinite(tsfc) or tsfc < 0:
                return np.inf
            mdot = tsfc * T_per * N_ENGINES
            
            # Base cost: J = ṁ/|Ps| [kg/m] (fuel per meter descended)
            J = mdot / abs(Ps)
            
            if not np.isfinite(J) or J <= 0:
                return np.inf
            
            # Augmented cost: J_aug = J + penalties
            if target_mach is not None and DescentCore.PenaltySystem.MACH_TRAJECTORY_GUIDANCE:
                mach_penalty = DescentCore.PenaltySystem.compute_mach_penalty(
                    mach, target_mach, None, descent_fraction
                )
                J += mach_penalty
            
            if DescentCore.PenaltySystem.LEVER_PENALTY_GUIDANCE:
                lever_penalty = DescentCore.PenaltySystem.compute_lever_penalty(lever, descent_fraction)
                J += lever_penalty
            
            return J
            
        except Exception:
            return np.inf
    
    @staticmethod
    def compute_descent_cost(*args, **kwargs):
        """Deprecated: Use compute_cost() instead. Maintained for backward compatibility."""
        import warnings
        warnings.warn("compute_descent_cost() is deprecated, use compute_cost() instead", DeprecationWarning, stacklevel=2)
        return DescentCore.compute_cost(*args, **kwargs)
    
    # ────────────────────────────────────────────────────────────────────
    # Performance Envelope Analysis
    # ────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def compute_full_envelope(aero: PyAerodynamicsWrapper, engine: EngineWrapper,
                             mach_grid: np.ndarray, altitude_sched: np.ndarray,
                             lever_grid: np.ndarray,
                             mass_kg: float,
                             target_mach: float = None) -> np.ndarray:
        """
        Compute 3D performance envelope J(M, h, δ) over feasible descent state space.
        
        Purpose: Map fuel cost density J = ṁ/|Ps| across entire flight envelope
        for visualization and feasibility analysis.
        
        Computation: Evaluate J at all grid points (M_i, h_k, δ_j) ∈ feasible region.
        Feasible region: M ∈ [M_stall(h,m), M_MMO], δ ∈ [0,1], Ps < 0.
        
        Parameters:
            aero: PyAerodynamicsWrapper - aerodynamic model
            engine: EngineWrapper - propulsion model
            mach_grid: np.ndarray - M_i, i=1..I (Mach discretization)
            altitude_sched: np.ndarray - h_k, k=1..K (altitude discretization, descending)
            lever_grid: np.ndarray - δ_j, j=1..L (throttle lever discretization)
            mass_kg: m_0 [kg] - reference mass for envelope computation
            target_mach: M_target [-] - approach Mach for penalties (optional)
            
        Returns:
            J_envelope: np.ndarray (I×K×L) - fuel cost density [kg/m]
        """
        if target_mach is None:
            target_mach = 0.25  # Default approach Mach
        
        print(f"[DESCENT] Computing performance envelope J(M,h,δ) at reference mass={mass_kg:.0f}kg")
        print(f"[DESCENT] Grid: {len(mach_grid)} Mach × {len(altitude_sched)} Alt × {len(lever_grid)} Lever")
        
        # Initialize envelope array: J[i,k,j] at (M_i, h_k, δ_j)
        I, K, L = len(mach_grid), len(altitude_sched), len(lever_grid)
        J_envelope = np.full((I, K, L), np.nan)
        
        # Evaluate J at all grid points
        feasible_count = 0
        total_points = I * K * L
        
        for k, h in enumerate(altitude_sched):
            # Descent progress for penalty system
            descent_fraction = k / (K - 1.0) if K > 1 else 0.0
            
            # Dynamic stall constraint: M ≥ M_stall(h, m)
            min_mach_h = calculate_min_descent_mach(h, mass_kg, aero=aero)
            
            for i, M in enumerate(mach_grid):
                # Flight envelope check: M ∈ [M_stall, M_MMO]
                if M < min_mach_h or M > M_MMO:
                    continue
                    
                for j, lever in enumerate(lever_grid):
                    # Compute J = ṁ/|Ps| + penalties
                    cost = DescentCore.compute_cost(
                        aero, engine, h, M, lever,
                        mass_kg,
                        target_mach=target_mach,
                        descent_fraction=descent_fraction
                    )
                    
                    if np.isfinite(cost) and cost > 0:
                        J_envelope[i, k, j] = cost
                        feasible_count += 1
            
            # Progress reporting
            if k % 10 == 0 or k == K - 1:
                progress = (k + 1) / K * 100
                print(f"  Progress: {progress:.1f}% ({k+1}/{K} altitudes)")
        
        print(f"[DESCENT] Envelope complete: {feasible_count}/{total_points} feasible ({100*feasible_count/total_points:.1f}%)")
        
        # Return as (M, h, δ) ordering
        return J_envelope
    
    @staticmethod
    def compute_full_descent_envelope(*args, **kwargs):
        """Deprecated: Use compute_full_envelope() instead. Maintained for backward compatibility."""
        import warnings
        warnings.warn("compute_full_descent_envelope() is deprecated, use compute_full_envelope() instead", DeprecationWarning, stacklevel=2)
        return DescentCore.compute_full_envelope(*args, **kwargs)

# ========================================================================
# SECTION 6: INTERFACE FUNCTIONS
# ========================================================================

def extract_descent_initial_state(cruise_results: CruiseResults,
                                  climb_fuel_kg: float,
                                  climb_time_s: float) -> DescentInitialState:
    """
    Extract initial state vector X_0 for descent from cruise endpoint.
    
    State extraction: X_0 = (h_f^cruise, M_f^cruise, m_f^cruise)
    Cumulative accounting: Σm_fuel = m_climb + m_cruise, Σt = t_climb + t_cruise
    
    Parameters:
        cruise_results: CruiseResults - cruise phase terminal state
        climb_fuel_kg: m_climb [kg] - fuel consumed in climb
        climb_time_s: t_climb [s] - time spent in climb
        
    Returns:
        DescentInitialState: initial state for descent optimization
    """
    # Extract terminal cruise state: X_f^cruise = (h_f, M_f, m_f)
    final_altitude = float(cruise_results.altitude_m[-1])     # h_f [m]
    final_mach = float(cruise_results.mach_number[-1])        # M_f [-]
    final_mass = float(cruise_results.mass_kg[-1])            # m_f [kg]
    
    # Cumulative fuel: Σm_fuel = m_climb + m_cruise
    total_fuel_consumed = climb_fuel_kg + cruise_results.total_fuel_consumed_kg
    
    # Cumulative time: Σt = t_climb + t_cruise
    total_time = climb_time_s + cruise_results.total_time_s
    
    print("[DESCENT] Initial state extraction:")
    print(f"  h_0 = {final_altitude:.0f} m")
    print(f"  M_0 = {final_mach:.3f}")
    print(f"  m_0 = {final_mass:.1f} kg")
    print(f"  Σm_fuel (climb+cruise) = {total_fuel_consumed:.1f} kg")
    print(f"  Σt (climb+cruise) = {total_time:.0f} s ({total_time/60:.1f} min)")
    
    return DescentInitialState(
        altitude_m=final_altitude,
        mach=final_mach,
        mass_kg=final_mass,
        fuel_consumed_total_kg=total_fuel_consumed,
        total_time_s=total_time
    )

# ========================================================================
# SECTION 7: MAIN OPTIMIZATION INTERFACE
# ========================================================================

def run_optimization(cruise_results: CruiseResults,
                    climb_fuel_kg: float,
                    climb_time_s: float,
                    aero: PyAerodynamicsWrapper,
                    engine: EngineWrapper,
                    target_altitude_m: float = None,
                    target_mach: float = None,
                    n_altitude_steps: int = N_ALTITUDE_STEPS_DESCENT,
                    n_mach_samples: int = N_MACH_SAMPLES_DESCENT,
                    lever_samples: int = N_LEVER_SAMPLES_DESCENT) -> Tuple[DescentResults, Dict]:
    """
    Primary interface for descent trajectory optimization.
    
    Algorithm: 3D dynamic programming over (h, M, δ) state space.
    Objective: Minimize ∫ J dh where J = ṁ/|Ps| (descent: Ps < 0).
    
    Provides consistent interface with climb.run_optimization() for
    unified mission analysis workflow.
    
    Parameters:
        cruise_results: CruiseResults - cruise phase terminal state
        climb_fuel_kg: m_climb [kg] - fuel consumed in climb
        climb_time_s: t_climb [s] - time spent in climb
        aero: PyAerodynamicsWrapper - drag model
        engine: EngineWrapper - thrust model
        target_altitude_m: h_target [m] - final altitude (default 300m)
        target_mach: M_target [-] - approach Mach (default 0.25)
        n_altitude_steps: K - altitude discretization
        n_mach_samples: I - Mach discretization
        lever_samples: L - throttle discretization
        
    Returns:
        DescentResults: optimal trajectory X*(k)
        dict: optimization metadata
    """
    if target_altitude_m is None:
        target_altitude_m = 300.0  # Approach altitude
    if target_mach is None:
        target_mach = 0.25  # Approach Mach
    
    print(f"\n{'='*80}")
    print("3D DYNAMIC PROGRAMMING DESCENT OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Target: M={target_mach:.3f} at h={target_altitude_m:.0f}m")
    print(f"Penalty guidance: Mach trajectory + Lever limits")
    print(f"{'='*80}")
    
    # Extract initial state
    initial_state = extract_descent_initial_state(cruise_results, climb_fuel_kg, climb_time_s)
    
    # Create descent altitude schedule (from high to low)
    H_descent = np.linspace(initial_state.altitude_m, target_altitude_m, n_altitude_steps)
    
    # Calculate dynamic minimum Mach at highest altitude
    min_mach_start = calculate_min_descent_mach(initial_state.altitude_m, initial_state.mass_kg)
    dbg(f"[DP-DESCENT] Dynamic min Mach at start: {min_mach_start:.3f} "
        f"(h={initial_state.altitude_m:.0f}m, m={initial_state.mass_kg:.0f}kg)")
    
    # Create Mach grid (from target approach speed to cruise Mach)
    # CRITICAL: Grid must include target_mach to allow optimizer to reach it
    # At high altitude, low Mach states will be naturally rejected as infeasible (stall)
    # At low altitude, they become feasible as mass decreases
    M_max = min(0.85, initial_state.mach + 0.05)  # Slightly above cruise Mach
    
    # Start grid from target Mach (or slightly below) to ensure it's reachable
    # The DP cost function will return np.inf for infeasible states (e.g., below stall speed)
    M_min = max(0.20, target_mach - 0.03)  # Minimum of 0.20 or target - 0.03
    
    # Ensure the grid captures both ends of the descent
    # Low Mach states at high altitude will be naturally infeasible (Ps >= 0 or stall)
    mach_grid = np.linspace(M_min, M_max, n_mach_samples)
    
    dbg(f"[DP-DESCENT] Note: Low Mach states at high altitude will be infeasible (stall-limited)")
    dbg(f"[DP-DESCENT] These states become feasible at lower altitudes as weight decreases")
    
    dbg(f"[DP-DESCENT] Mach grid: {M_min:.3f} to {M_max:.3f} ({n_mach_samples} samples)")
    
    # Run DP optimization
    dp_result, dp_info = DescentCore.DynamicProgrammingOptimizer.solve_3d_dp(
        aero=aero,
        engine=engine,
        mach_grid=mach_grid,
        altitude_sched=H_descent,
        initial_state=initial_state,
        lever_samples=lever_samples,
        target_mach=target_mach,
        target_mach_tolerance=DescentCore.PenaltySystem.TARGET_MACH_TOLERANCE
    )
    
    print(f"{'='*80}")
    print("DP OPTIMIZATION COMPLETED")
    print(f"{'='*80}\n")
    
    return dp_result, dp_info

def run_descent_dp_optimization(*args, **kwargs):
    """Deprecated wrapper. Use run_optimization() instead."""
    import warnings
    warnings.warn("run_descent_dp_optimization() is deprecated, use run_optimization() instead", DeprecationWarning, stacklevel=2)
    return run_optimization(*args, **kwargs)

# ========================================================================
# SECTION 8: MODULE EXPORTS
# ========================================================================

# Phase-agnostic type alias for mission analysis
OptimalTrajectory = DescentResults