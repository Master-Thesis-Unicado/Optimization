# ========================================================================
# SECTION 1: MODULE INITIALIZATION
# ========================================================================
# Standard imports
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, List, Dict, Any, Tuple

# Aircraft configuration: N_ENGINES, m_0 [kg], S_ref [m²], Mach limits, lever limits, g_c [m/s²]
from aircraft_config import (
    SystemConfiguration,
    N_ENGINES, INITIAL_MASS_KG, S_REF_M2,
    M_MIN_DEFAULT, M_MIN_EFFECTIVE, M_MMO,
    LEVER_MIN, LEVER_MAX,
    G_C,
    a_from_altitude, _atmospheric_properties
)

# Mission parameters: altitude steps, energy rate, penalty coefficients, consolidated parameters
from mission_config import (
    N_ALTITUDE_STEPS_CLIMB, N_LEVER_SAMPLES_CLIMB, E_DOT_CMD_CLIMB,
    START_ALTITUDE_CLIMB_M, START_VELOCITY_CLIMB_MS, START_LEVER_CLIMB,
    TARGET_ALT_CLIMB_M, TARGET_MACH_CRUISE,
    PENALTY_CLIMB_MACH_TRAJECTORY_GUIDANCE, PENALTY_CLIMB_LEVER_PENALTY_GUIDANCE,
    PENALTY_CLIMB_TARGET_MACH_TOLERANCE,
    PENALTY_CLIMB_MAX_REASONABLE_MACH_RATE, PENALTY_CLIMB_TOTAL_STEPS_ESTIMATE,
    PENALTY_CLIMB_URGENCY_MULTIPLIER, PENALTY_CLIMB_GUIDANCE_PENALTY_WEIGHT,
    PENALTY_CLIMB_LEVER_PENALTY_WEIGHT, PENALTY_CLIMB_LEVER_PENALTY_THRESHOLD,
    PENALTY_CLIMB_LEVER_PENALTY_EXPONENT, PENALTY_CLIMB_LEVER_PENALTY_CRITICAL_THRESHOLD,
    PENALTY_CLIMB_LEVER_PENALTY_CRITICAL_MULTIPLIER, PENALTY_CLIMB_LEVER_PENALTY_ULTRA_CRITICAL_THRESHOLD,
    PENALTY_CLIMB_LEVER_PENALTY_ULTRA_CRITICAL_MULTIPLIER,
    CLIMB_ALTITUDE_UNIFORMITY_TOLERANCE, CLIMB_THRUST_LIMITED_ATOL,
    CLIMB_MIN_RESAMPLE_POINTS, CLIMB_STALL_SAFETY_MARGIN,
    TARGET_MACH_TOLERANCE,
    MACH_GUIDANCE_FINAL_PHASE_START, MACH_GUIDANCE_FINAL_PHASE_RANGE,
    MACH_GUIDANCE_TERMINAL_PHASE_START, MACH_GUIDANCE_TERMINAL_PHASE_RANGE,
    MACH_GUIDANCE_TERMINAL_BOOST_MULTIPLIER,
    DP_MIN_ALTITUDE_SEGMENT_M, DP_MIN_VELOCITY_CHANGE_MPS, DP_MIN_TIME_STEP_S,
    DP_MIN_SEGMENT_DISTANCE_M,
    DP_PROGRESS_REPORT_INTERVAL, DP_TRAJECTORY_DEBUG_LIMIT
)

# External model interfaces: aerodynamics D(M,h,m) and propulsion T(δ,M,h)
from pyaerodynamics_wrapper import PyAerodynamicsWrapper
from pyengine_wrapper import EngineWrapper

# Utility functions: kinematics, energy, time integration
from mission_utils import (
    find_lever_for_thrust,
    pad_array_to_length,
    calculate_specific_excess_power,
    calculate_stall_mach,
    calculate_fuel_flow_rate_safe,
    validate_tsfc,
    thrust_per_engine_to_total,
    calculate_acceleration_rate,
    calculate_time_from_altitude_change,
    calculate_time_from_velocity_change,
    build_time_array_from_segments,
    find_closest_index,
    mach_from_velocity
)

# ========================================================================
# SECTION 2: DATA STRUCTURES
# ========================================================================

@dataclass
class ClimbInitialState:
    """
    Initial state vector for climb phase from takeoff.
    
    State: X_0 = (h_0, M_0, m_0) at takeoff
    """
    altitude_m: float                  # h_0 [m]: initial altitude
    mach: float                        # M_0 [-]: initial Mach number
    mass_kg: float                     # m_0 [kg]: aircraft mass at climb start
    lever: float                       # δ_0 [-]: initial throttle position
    
    def __post_init__(self):
        """Validate state: h >= 0, M in safe range, m > 0, lever in [0,1]."""
        if self.altitude_m < 0:
            raise ValueError(f"Initial altitude {self.altitude_m:.0f}m must be non-negative")
        if not (M_MIN_EFFECTIVE <= self.mach <= M_MMO):
            raise ValueError(f"Initial Mach {self.mach:.3f} outside safe range [{M_MIN_EFFECTIVE}, {M_MMO}]")
        if self.mass_kg <= 0:
            raise ValueError(f"Mass must be positive: {self.mass_kg:.1f} kg")
        if not (LEVER_MIN <= self.lever <= LEVER_MAX):
            raise ValueError(f"Initial lever {self.lever:.3f} outside range [{LEVER_MIN}, {LEVER_MAX}]")

@dataclass
class MinFuelSchedule:
    """
    Optimal climb trajectory from dynamic programming solution.
    
    State variables: h(t) [m], M(t) [-], δ(t) [-]
    Performance: J [kg/m], Ps [m/s], ṁ [kg/s]
    Forces: T [N], D [N]
    """
    alt_m: np.ndarray              # h [m]: altitude profile
    mach: np.ndarray               # M [-]: Mach number profile
    fuel_est_kg: float             # m_fuel [kg]: total fuel consumed
    J_kg_per_m: np.ndarray         # J [kg/m]: fuel cost density
    mdot_kgps: np.ndarray          # ṁ [kg/s]: fuel flow rate
    Ps_mps: np.ndarray             # Ps [m/s]: specific excess power
    thrust_total_N: np.ndarray     # T_total [N]: total thrust
    D_N: np.ndarray                # D [N]: drag
    lever: np.ndarray              # δ [-]: throttle lever position [0,1]
    T_per_engine_N: np.ndarray     # T_eng [N]: thrust per engine
    mass_kg: np.ndarray            # m(t) [kg]: aircraft mass
    thrust_limited: np.ndarray     # Boolean: thrust at maximum (δ ≈ 1)
    dt_s: np.ndarray               # Δt [s]: time increments
    dFuel_kg: np.ndarray           # Δm_fuel [kg]: fuel increments
    cumFuel_kg: np.ndarray         # m_fuel(t) [kg]: cumulative fuel
    
    def __post_init__(self):
        """Validate trajectory data: finite values, positive fuel, consistent array lengths."""
        if len(self.alt_m) == 0:
            raise ValueError("Altitude array cannot be empty")
        if self.fuel_est_kg < 0:
            raise ValueError(f"Fuel estimate cannot be negative: {self.fuel_est_kg:.1f} kg")
        if not np.all(np.isfinite(self.alt_m)):
            raise ValueError("Altitude array contains non-finite values")
        if not np.all(np.isfinite(self.mach)):
            raise ValueError("Mach array contains non-finite values")
        if len(self.alt_m) != len(self.mach):
            raise ValueError(f"Array length mismatch: alt_m={len(self.alt_m)}, mach={len(self.mach)}")

# ========================================================================
# SECTION 3: COMPUTATIONAL PRIMITIVES
# ========================================================================

class ClimbingCore:
    """
    Climb trajectory optimization framework via dynamic programming.
    
    Mathematical Formulation:
        State space: X = (h, M, δ) ∈ ℝ³
        Cost functional: J[X(·)] = ∫ (ṁ/Ps) dh
        Optimization: min J subject to flight envelope constraints
    
    Subsystems:
        - EnergyCalculator: Power balance T·V - D·V = mg·ḣ + mV·V̇
        - DynamicProgrammingOptimizer: 3D Bellman recursion over (h,M,δ)
        - PenaltySystem: Soft constraints via augmented cost J_aug = J + penalties
        - Resampling: Trajectory interpolation for uniform grids
    
    Usage:
        # Compute optimal trajectory
        schedule, info = ClimbingCore.DynamicProgrammingOptimizer.solve_3d_dp(
            aero, engine, mach_grid, altitude_sched, lever_samples=10, target_mach=0.78
        )
        
        # Evaluate point cost
        J = ClimbingCore.compute_cost(aero, engine, h=5000, M=0.6, lever=0.8)
    """
    
    # ────────────────────────────────────────────────────────────────────
    # Trajectory Resampling
    # ────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def resample_strategy_run(sr: 'ClimbingCore.EnergyCalculator.StrategyRun', n_samples: int) -> 'ClimbingCore.EnergyCalculator.StrategyRun':
        """
        Resample trajectory onto uniform altitude grid via linear interpolation.
        
        Method: Given trajectory X(h) with non-uniform altitude spacing, interpolate
        all state and performance variables onto h_new = linspace(h_0, h_f, n_samples).
        
        Algorithm:
            1. Pad arrays to maximum length for consistency
            2. Generate uniform altitude grid: h_new ∈ [h_min, h_max]
            3. Interpolate: X_new(h_new) = interp1d(h_old, X_old, h_new)
            4. Recompute derived quantities: Δt, Δm_fuel
        
        Parameters:
            sr: StrategyRun - input trajectory data
            n_samples: int - number of points in resampled grid (n ≥ 2)
            
        Returns:
            StrategyRun: resampled trajectory with uniform Δh
        """
        if n_samples < CLIMB_MIN_RESAMPLE_POINTS:
            raise ValueError(f"n_samples must be >= {CLIMB_MIN_RESAMPLE_POINTS}, got {n_samples}")
        n = int(n_samples)
        
        # Step 1: Pad arrays to consistent length
        padded_arrays = ClimbingCore._pad_arrays_to_max_length(sr)
        alt_old = padded_arrays['alt_m']  # h_old [m]
        
        # Step 2: Generate uniform altitude grid h_new = [h_0, ..., h_f]
        alt_new = np.linspace(alt_old[0], alt_old[-1], n)
        
        def safe_interp(y):
            """Linear interpolation: y_new = interp1d(h_old, y_old, h_new)."""
            if len(y) != len(alt_old):
                dbg(f"[ERROR] Array length mismatch in resample_strategy_run: {len(y)} vs {len(alt_old)} for strategy '{sr.label}'")
                if len(y) == 0:
                    return np.zeros_like(alt_new)
                return np.full_like(alt_new, y[-1])
            return np.interp(alt_new, alt_old, y)
        
        # Step 3: Interpolate state and performance variables
        time_new = safe_interp(padded_arrays['time_s'])      # t(h) [s]
        mach_new = safe_interp(padded_arrays['mach'])        # M(h) [-]
        lever_new = safe_interp(padded_arrays['lever'])      # δ(h) [-]
        Ttot_new = safe_interp(padded_arrays['thrust_total_N'])   # T(h) [N]
        D_new = safe_interp(padded_arrays['D_N'])            # D(h) [N]
        Ps_new = safe_interp(padded_arrays['Ps_mps'])        # Ps(h) [m/s]
        mdot_new = safe_interp(padded_arrays['mdot_kgps'])   # ṁ(h) [kg/s]
        cumF_new = safe_interp(padded_arrays['cumFuel_kg'])  # m_fuel(h) [kg]
        
        # Boolean interpolation: thrust_limited → {0,1} via threshold at 0.5
        limited_f = safe_interp(padded_arrays['thrust_limited'].astype(float))
        limited_new = (limited_f >= 0.5)
        
        # Step 4: Compute incremental arrays via finite differences
        dt_new = np.diff(time_new, prepend=time_new[0])      # Δt [s]
        dFuel_new = np.diff(cumF_new, prepend=cumF_new[0])   # Δm_fuel [kg]
        fuel_tot = float(cumF_new[-1] - cumF_new[0])         # Total fuel [kg]
        
        return ClimbingCore.EnergyCalculator.StrategyRun(
            label=sr.label,
            alt_m=alt_new,
            mach=mach_new,
            time_s=time_new,
            lever=lever_new,
            thrust_total_N=Ttot_new,
            D_N=D_new,
            Ps_mps=Ps_new,
            mdot_kgps=mdot_new,
            dt_s=dt_new,
            dFuel_kg=dFuel_new,
            cumFuel_kg=cumF_new,
            thrust_limited=limited_new,
            fuel_total_kg=fuel_tot
        )
    
    @staticmethod
    def _pad_arrays_to_max_length(sr: 'ClimbingCore.EnergyCalculator.StrategyRun') -> dict:
        """
        Pad trajectory arrays to uniform length via forward-fill extrapolation.
        
        Method: For arrays with length L < L_max, extend via arr[L:L_max] = arr[L-1].
        Creates copies to avoid mutation of input object.
        
        Parameters:
            sr: StrategyRun - trajectory with potentially mismatched array lengths
            
        Returns:
            dict: {name: padded_array} for all trajectory variables
        """
        # Array catalog
        array_names = ['alt_m', 'time_s', 'mach', 'lever', 'thrust_total_N', 'D_N', 
                     'Ps_mps', 'mdot_kgps', 'cumFuel_kg', 'thrust_limited']
        
        # Determine maximum length L_max
        array_lengths = [len(getattr(sr, name)) for name in array_names]
        max_length = max(array_lengths) if array_lengths else 0
        
        # Pad each array to L_max
        padded_arrays = {}
        for name in array_names:
            arr = np.array(getattr(sr, name), copy=True)
            
            if len(arr) < max_length:
                dbg(f"[WARNING] Array {name} has length {len(arr)} but max length is {max_length}. Padding to match.")
                # Select pad value: last element or zero/False for empty arrays
                if name == 'thrust_limited':
                    pad_value = arr[-1] if len(arr) > 0 else False
                else:
                    pad_value = arr[-1] if len(arr) > 0 else 0.0
                
                arr = pad_array_to_length(arr, max_length, pad_value)
            
            padded_arrays[name] = arr
        
        return padded_arrays
    
    # ────────────────────────────────────────────────────────────────────
    # Energy Balance Computations
    # ────────────────────────────────────────────────────────────────────
    
    class EnergyCalculator:
        """
        Energy and thrust calculations via power balance equation.
        
        Fundamental equation: T·V - D·V = mg·ḣ + mV·V̇
        
        Methods:
            - compute_required_thrust: Solve for T given (ḣ, V̇)
            - StrategyRun: Data structure for trajectory storage
        """
        
        @staticmethod
        def compute_required_thrust(mass_kg: float, dh_dt: float, dv_dt: float, 
                                  V: float, D: float) -> float:
            """
            Compute required thrust from energy rate allocation.
            
            Equation: T·V = D·V + mg·ḣ + mV·V̇
            Solution: T = D + (mg·ḣ + mV·V̇)/V
            
            Parameters:
                mass_kg: m [kg] - aircraft mass
                dh_dt: ḣ [m/s] - climb rate
                dv_dt: V̇ [m/s²] - acceleration rate
                V: V [m/s] - true airspeed
                D: D [N] - drag force
                
            Returns:
                T [N]: required total thrust
            """
            F_required_total = D + (mass_kg * G_C * dh_dt + mass_kg * V * dv_dt) / max(V, 1e-9)
            return F_required_total
        
        @dataclass
        class StrategyRun:
            """
            Trajectory data structure for climb strategies and DP solutions.
            
            Variables:
                State: h [m], M [-], δ [-], t [s]
                Forces: T [N], D [N]
                Performance: Ps [m/s], ṁ [kg/s]
                Fuel: m_fuel [kg], Δm_fuel [kg]
            """
            label: str                     # Strategy identifier
            alt_m: np.ndarray              # h [m]: altitude
            mach: np.ndarray               # M [-]: Mach number
            time_s: np.ndarray             # t [s]: time
            lever: np.ndarray              # δ [-]: throttle lever [0,1]
            thrust_total_N: np.ndarray     # T [N]: total thrust
            D_N: np.ndarray                # D [N]: drag
            Ps_mps: np.ndarray             # Ps [m/s]: specific excess power
            mdot_kgps: np.ndarray          # ṁ [kg/s]: fuel flow rate
            dt_s: np.ndarray               # Δt [s]: time increments
            dFuel_kg: np.ndarray           # Δm_fuel [kg]: fuel increments
            cumFuel_kg: np.ndarray         # m_fuel(t) [kg]: cumulative fuel
            thrust_limited: np.ndarray     # Boolean: thrust at maximum
            fuel_total_kg: float           # m_fuel,total [kg]: total fuel consumed
    
    # ────────────────────────────────────────────────────────────────────
    # Dynamic Programming Optimization (Bellman Recursion)
    # ────────────────────────────────────────────────────────────────────
    
    class DynamicProgrammingOptimizer:
        """
        3D dynamic programming solver for minimum-fuel climb trajectories.
        
        Formulation:
            State: X_k = (h_k, M_k, δ_k) at altitude level k
            Cost-to-go: F[k, i, j] = min fuel from X_k to target
            Recursion: F[k+1, i', j'] = min_{i,j} {F[k, i, j] + J(X_k→X_{k+1}) · Δh}
            Transition kernel: 7×7 grid in (M, δ) space at each altitude step
        """
        
        @staticmethod
        def solve_3d_dp(aero: PyAerodynamicsWrapper, engine: EngineWrapper,
                        mach_grid: np.ndarray, altitude_sched: np.ndarray,
                        initial_state: ClimbInitialState,
                        lever_samples: int = N_LEVER_SAMPLES_CLIMB,
                        target_mach: float = None,
                        target_mach_tolerance: float = PENALTY_CLIMB_TARGET_MACH_TOLERANCE):
            """
            Solve 3D Bellman equation for optimal climb trajectory.
            
            Algorithm:
                1. Initialize: F[0, M_start, δ_start] = 0
                2. Forward pass: Compute F[k+1] from F[k] via Bellman recursion
                3. Terminal constraint: Enforce |M_final - M_target| < tol
                4. Backtrack: Recover optimal path from predecessor matrix
                5. Post-process: Compute time, fuel, and performance arrays
            
            Parameters:
                aero: PyAerodynamicsWrapper - drag model D(M,h,m)
                engine: EngineWrapper - thrust model T(δ,M,h)
                mach_grid: np.ndarray - M_i, i=1..I (Mach discretization)
                altitude_sched: np.ndarray - h_k, k=1..K (altitude schedule)
                initial_state: ClimbInitialState - X_0 from takeoff
                lever_samples: int - number of throttle positions L
                target_mach: float - M_target at h_final (optional)
                target_mach_tolerance: float - terminal Mach tolerance
            
            Returns:
                MinFuelSchedule: optimal trajectory X*(k)
                dict: optimization metadata (costs, path length, time)
            """
            # Grid dimensions: K altitude levels, I Mach points, L lever positions
            K, I = len(altitude_sched), len(mach_grid)
            L = lever_samples
            
            # Throttle discretization: δ ∈ [δ_min, δ_max]
            lever_grid = np.linspace(LEVER_MIN, LEVER_MAX, L)
            
            # Initialize DP arrays
            F = np.full((K, I, L), np.inf)              # F[k,i,j]: cost-to-go [kg]
            weight_matrix = np.full((K, I, L), np.nan)  # m[k,i,j]: mass [kg]
            prv = np.full((K, I, L, 3), -1, dtype=int)  # predecessor[k,i,j] = [k',i',j']
        
            # Determine initial state indices from ClimbInitialState
            start_mach_idx = find_closest_index(initial_state.mach, mach_grid)
            start_lever_idx = find_closest_index(initial_state.lever, lever_grid)
            actual_start_mach = mach_grid[start_mach_idx]
            actual_start_lever = lever_grid[start_lever_idx]
            
            dbg(f"[3D-DP] Initial Mach: M_0 = {actual_start_mach:.3f} (requested {initial_state.mach:.3f})")
            dbg(f"[3D-DP] Initial lever: δ_0 = {actual_start_lever:.3f} (requested {initial_state.lever:.3f})")
            
            # Validate initial state X_0 = (h_0, M_0, δ_0)
            if (actual_start_mach >= M_MIN_EFFECTIVE and 
                actual_start_mach <= M_MMO and
                actual_start_lever >= LEVER_MIN and 
                actual_start_lever <= LEVER_MAX):
                
                # Compute climb progress fraction (step-based, consistent with descent)
                climb_fraction = 0.0  # Initial step: k=0
                
                # Verify feasibility via cost computation
                cost = ClimbingCore.compute_cost(aero, engine, altitude_sched[0], actual_start_mach, actual_start_lever,
                                                  initial_state.mass_kg, target_mach=target_mach, prev_mach=None, climb_fraction=climb_fraction)
                if np.isfinite(cost) and cost > 0:
                    F[0, start_mach_idx, start_lever_idx] = 0.0  # Boundary condition: F[0] = 0
                    weight_matrix[0, start_mach_idx, start_lever_idx] = initial_state.mass_kg
                    dbg(f"[3D-DP] Initial state validated: h_0={altitude_sched[0]:.0f}m, M_0={actual_start_mach:.3f}, δ_0={actual_start_lever:.3f}, m_0={initial_state.mass_kg:.0f}kg")
                else:
                    raise RuntimeError(f"[3D-DP] Initial state infeasible: cost={cost}")
            else:
                raise RuntimeError(f"[3D-DP] Initial state violates bounds: M={actual_start_mach:.3f}, δ={actual_start_lever:.3f}")
            
            if not np.isfinite(F[0, start_mach_idx, start_lever_idx]):
                raise RuntimeError("[3D-DP] Initialization failed: no feasible starting point")
        
            # ================================================================
            # Forward Pass: Bellman Recursion
            # ================================================================
            # Compute F[k+1] from F[k] for k = 0, 1, ..., K-2
            for k in range(K - 1):
                current_alt = altitude_sched[k]      # h_k [m]
                next_alt = altitude_sched[k + 1]     # h_{k+1} [m]
                dh = next_alt - current_alt          # Δh [m]
                
                # Progress tracking (step-based, consistent with descent)
                climb_fraction = k / (K - 1.0) if K > 1 else 0.0
                next_climb_fraction = (k + 1) / (K - 1.0) if K > 1 else 1.0
                if k % DP_PROGRESS_REPORT_INTERVAL == 0:
                    dbg(f"[DP-CLIMB] Altitude level k={k}/{K-1}: h={current_alt:.0f}m → {next_alt:.0f}m (progress: {climb_fraction*100:.1f}%)")
                
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
                                
                                # Flight envelope constraints: M ∈ [M_min, M_MMO], δ ∈ [δ_min, δ_max]
                                if (next_mach >= M_MIN_EFFECTIVE and 
                                    next_mach <= M_MMO and
                                    next_lever >= LEVER_MIN and 
                                    next_lever <= LEVER_MAX):
                                    
                                    # ────────────────────────────────────────────────────────
                                    # Cost Computation with Mass Coupling
                                    # ────────────────────────────────────────────────────────
                                    # Climb progress fractions for penalty system (step-based)
                                    prev_mach = mach_grid[i] if k > 0 else None
                                    
                                    # Current state cost: J_k = J(h_k, M_i, δ_j, m_k)
                                    current_cost = ClimbingCore.compute_cost(aero, engine, current_alt, current_mach, current_lever,
                                                                             current_weight, target_mach=target_mach, prev_mach=prev_mach, 
                                                                             climb_fraction=climb_fraction)
                                    
                                    if not (np.isfinite(current_cost) and current_cost > 0):
                                        continue
                                    
                                    # Next state cost (initial): J_{k+1}^(0) using m_k
                                    next_cost_initial = ClimbingCore.compute_cost(aero, engine, next_alt, next_mach, next_lever,
                                                                                   current_weight, target_mach=target_mach, prev_mach=current_mach, 
                                                                                   climb_fraction=next_climb_fraction)
                                    
                                    if not (np.isfinite(next_cost_initial) and next_cost_initial > 0):
                                        continue
                                    
                                    # Fuel burn estimate (trapezoidal rule): Δm = (J_k + J_{k+1})/2 · Δh
                                    step_cost_initial = 0.5 * (current_cost + next_cost_initial) * dh
                                    fuel_burned_initial = step_cost_initial
                                    
                                    # Mass update: m_{k+1} = m_k - Δm
                                    next_weight = current_weight - fuel_burned_initial
                                    if next_weight <= 0:
                                        continue
                                    
                                    # Cost refinement: Recompute J_{k+1} with updated mass m_{k+1}
                                    # Accounts for Ps ∝ 1/m → J = ṁ/Ps ∝ m
                                    next_cost_refined = ClimbingCore.compute_cost(aero, engine, next_alt, next_mach, next_lever,
                                                                                   next_weight, target_mach=target_mach, prev_mach=current_mach, 
                                                                                   climb_fraction=next_climb_fraction)
                                    
                                    if not (np.isfinite(next_cost_refined) and next_cost_refined > 0):
                                        next_cost = next_cost_initial  # Fallback
                                    else:
                                        next_cost = next_cost_refined
                                    
                                    # Final step cost with refined J_{k+1}
                                    step_cost = 0.5 * (current_cost + next_cost) * dh
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
            
                if k % DP_PROGRESS_REPORT_INTERVAL == 0 or feasible_count == 0:
                    dbg(f"[DP-CLIMB] Level k={k}: {feasible_count} feasible transitions")
        
            # ================================================================
            # Terminal Constraint: |M_final - M_target| < tolerance
            # ================================================================
            if target_mach is not None:
                dbg(f"[DP-CLIMB] Enforcing terminal constraint: M_final ∈ [{target_mach - target_mach_tolerance:.3f}, {target_mach + target_mach_tolerance:.3f}]")
                
                # Identify valid terminal Mach indices
                valid_final = np.abs(mach_grid - target_mach) < target_mach_tolerance
                
                if not valid_final.any():
                    dbg(f"[DP-CLIMB] Warning: No Mach within tolerance, using closest")
                    closest_idx = find_closest_index(target_mach, mach_grid)
                    valid_final = np.zeros_like(valid_final, dtype=bool)
                    valid_final[closest_idx] = True
                
                # Mask infeasible terminal states: F[K-1, i, :] = ∞ for i ∉ valid_final
                for i in range(I):
                    if not valid_final[i]:
                        F[-1, i, :] = np.inf
            
            # Verify solution existence
            if not np.isfinite(F[-1]).any():
                raise RuntimeError("[DP-CLIMB] No feasible path to terminal altitude")
            
            # ================================================================
            # Optimal Terminal State: X* = argmin F[K-1]
            # ================================================================
            final_flat_idx = np.nanargmin(F[-1])
            final_mach_idx, final_lever_idx = np.unravel_index(final_flat_idx, F[-1].shape)
            final_alt_idx = K - 1
            
            dbg(f"[DP-CLIMB] Optimal terminal state: h_f={altitude_sched[final_alt_idx]:.0f}m, "
                f"M_f={mach_grid[final_mach_idx]:.3f}, δ_f={lever_grid[final_lever_idx]:.3f}")
            dbg(f"[DP-CLIMB] Minimum fuel: {F[final_alt_idx, final_mach_idx, final_lever_idx]:.1f} kg")
            
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
                
                # Diagnostic: Check altitude step consistency
                if len(path_alt) > 1:
                    alt_diff = path_alt[-1] - path_alt[-2]
                    expected_step = altitude_sched[1] - altitude_sched[0]
                    if abs(alt_diff) > expected_step * 1.5:
                        dbg(f"[DP-CLIMB] WARNING: Large altitude step: {path_alt[-2]:.0f}m → {path_alt[-1]:.0f}m (Δh={alt_diff:.0f}m)")
                
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
            
            # Diagnostic output
            dbg(f"[DP-CLIMB] Optimal trajectory: {len(path_alt)} points")
            for i in range(min(DP_TRAJECTORY_DEBUG_LIMIT, len(path_alt))):
                dbg(f"  k={i}: h={path_alt[i]:.0f}m, M={path_mach[i]:.3f}, δ={path_lever[i]:.3f}, m={path_weights[i]:.0f}kg")
            if len(path_alt) > DP_TRAJECTORY_DEBUG_LIMIT:
                dbg(f"  ... ({len(path_alt)-DP_TRAJECTORY_DEBUG_LIMIT} more points)")
            
            # Validate uniform altitude spacing
            for i in range(1, len(path_alt)):
                alt_diff = path_alt[i] - path_alt[i-1]
                expected_diff = altitude_sched[1] - altitude_sched[0]
                if abs(alt_diff - expected_diff) > expected_diff * CLIMB_ALTITUDE_UNIFORMITY_TOLERANCE:
                    dbg(f"[DP-CLIMB] WARNING: Non-uniform Δh at k={i}: Δh={alt_diff:.0f}m (expected {expected_diff:.0f}m)")
            
            # ================================================================
            # Post-Processing: Compute Time and Fuel Arrays
            # ================================================================
            # Convert path lists to arrays
            alt_array = np.array(path_alt)        # h[k] [m]
            mach_array = np.array(path_mach)      # M[k] [-]
            lever_array = np.array(path_lever)    # δ[k] [-]
            weight_array = np.array(path_weights)  # m[k] [kg]
            
            # Initialize segment arrays (k → k+1 for k=0..K-2)
            n_segments = len(alt_array) - 1       # Number of segments: K-1
            dt_array = np.zeros(n_segments)       # Δt[k] [s] for segment k→k+1
            dF_array = np.zeros(n_segments)       # Δm_fuel[k] [kg] for segment k→k+1
            
            # ────────────────────────────────────────────────────────────────
            # Segment-wise Time and Fuel Integration
            # ────────────────────────────────────────────────────────────────
            # For each segment k→k+1: compute Δt and Δm_fuel via trapezoidal rule
            for i in range(n_segments):
                h_curr, h_next = alt_array[i], alt_array[i + 1]          # h_k, h_{k+1} [m]
                M_curr, M_next = mach_array[i], mach_array[i + 1]        # M_k, M_{k+1} [-]
                lever_curr, lever_next = lever_array[i], lever_array[i + 1]  # δ_k, δ_{k+1} [-]
                weight_curr, weight_next = weight_array[i], weight_array[i + 1]  # m_k, m_{k+1} [kg]
                
                # Altitude increment
                dh = abs(h_next - h_curr) if abs(h_next - h_curr) > DP_MIN_ALTITUDE_SEGMENT_M else 0.0
                
                # Cost at current state: J_k [kg/m]
                current_cost = ClimbingCore.compute_cost(
                    aero, engine, h_curr, M_curr, lever_curr,
                    weight_curr, target_mach=target_mach, prev_mach=None,
                    climb_fraction=None
                )
                
                if not (np.isfinite(current_cost) and current_cost > 0):
                    dt_array[i] = 0.0
                    dF_array[i] = 0.0
                    continue
                
                # ────────────────────────────────────────────────────────────
                # Cost Calculation Using DP Mass Values
                # ────────────────────────────────────────────────────────────
                # Use weight_next directly from DP (already accounts for dynamic mass)
                # This ensures consistency with DP optimization where weight_next was
                # calculated considering dynamic drag D(M,h,m) effects
                
                # Cost at next state using DP mass (weight_next from DP forward pass)
                next_cost = ClimbingCore.compute_cost(
                    aero, engine, h_next, M_next, lever_next,
                    weight_next, target_mach=target_mach, prev_mach=M_curr,
                    climb_fraction=None
                )
                
                if not (np.isfinite(next_cost) and next_cost > 0):
                    dt_array[i] = 0.0
                    dF_array[i] = 0.0
                    continue
                
                # Fuel burn calculation via trapezoidal rule using DP mass values
                # Δm = (J_k + J_{k+1})/2 · Δh where J evaluated at DP-optimized mass values
                if dh > DP_MIN_ALTITUDE_SEGMENT_M:  # Vertical segment: Δm = (J_k + J_{k+1})/2 · Δh
                    fuel_burned = 0.5 * (current_cost + next_cost) * dh
                else:  # Horizontal segment: approximate distance from velocity
                    a = a_from_altitude(h_curr)
                    V_curr = M_curr * a
                    V_next = M_next * a
                    ds = 0.5 * (V_curr + V_next) * DP_MIN_TIME_STEP_S if abs(V_next - V_curr) > DP_MIN_VELOCITY_CHANGE_MPS else DP_MIN_SEGMENT_DISTANCE_M
                    fuel_burned = 0.5 * (current_cost + next_cost) * ds
                
                dF_array[i] = fuel_burned
                
                # Use DP mass values for time calculation (no recalculation needed)
                weight_avg = 0.5 * (weight_curr + weight_next)
                
                # ────────────────────────────────────────────────────────────
                # Time Calculation: Δt = f(Δh, Δv) using DP mass values
                # ────────────────────────────────────────────────────────────
                if dh > DP_MIN_ALTITUDE_SEGMENT_M:  # Vertical segment: Δt = Δh / Ps
                    # Average state for Ps computation using DP mass values
                    h_avg = 0.5 * (h_curr + h_next)
                    M_avg = 0.5 * (M_curr + M_next)
                    lever_avg = 0.5 * (lever_curr + lever_next)
                    
                    # Compute Ps = (T-D)V/m with DP average mass
                    a = a_from_altitude(h_avg)
                    V = M_avg * a
                    D = aero.get_drag(M_avg, h_avg, weight_avg)
                    T_per = engine.thrust_with_lever(lever_avg, M_avg, h_avg)
                    T_tot = T_per * SystemConfiguration.N_ENGINES
                    Ps = calculate_specific_excess_power(T_tot, D, weight_avg, V)
                    
                    # Δt = Δh / Ps
                    dt_array[i] = calculate_time_from_altitude_change(dh, Ps)
                    if dt_array[i] > 0:
                        dbg(f"[DP-CLIMB] Segment {i} (vertical): Δh={dh:.0f}m, Δt={dt_array[i]:.3f}s, Δm={fuel_burned:.3f}kg")
                    else:
                        dt_array[i] = 0.0
                        dF_array[i] = 0.0
                
                else:  # Horizontal segment: Δt = ΔV / a
                    a_curr = a_from_altitude(h_curr)
                    V_curr = M_curr * a_curr
                    V_next = M_next * a_curr
                    
                    if abs(V_next - V_curr) > DP_MIN_VELOCITY_CHANGE_MPS:  # Significant ΔV
                        # Compute acceleration: a = (T-D)/m with DP average mass
                        D = aero.get_drag(M_curr, h_curr, weight_avg)
                        T_per = engine.thrust_with_lever(lever_curr, M_curr, h_curr)
                        T_tot = thrust_per_engine_to_total(T_per, SystemConfiguration.N_ENGINES)
                        a_accel = calculate_acceleration_rate(T_tot, D, weight_avg)
                        
                        # Δt = ΔV / a
                        dt_array[i] = calculate_time_from_velocity_change(V_next - V_curr, a_accel)
                        if dt_array[i] > 0:
                            dbg(f"[DP-CLIMB] Segment {i} (horizontal): ΔV={V_next-V_curr:.1f}m/s, Δt={dt_array[i]:.3f}s, Δm={fuel_burned:.3f}kg")
                        else:
                            dt_array[i] = DP_MIN_TIME_STEP_S  # Minimum time step
                    else:
                        dt_array[i] = DP_MIN_TIME_STEP_S  # Minimal velocity change
                
                # Diagnostic for final segments
                if i >= n_segments - 2:
                    dbg(f"[DP-CLIMB] Segment {i}: h={h_curr:.0f}→{h_next:.0f}m, Δt={dt_array[i]:.3f}s, Δm={dF_array[i]:.3f}kg")
            
            # ════════════════════════════════════════════════════════════════
            # Construct Full Time and Fuel Arrays (n_points elements)
            # ════════════════════════════════════════════════════════════════
            n_points = len(alt_array)
            
            # Time array: t[0]=0, t[k] = t[k-1] + Δt[k-1] for k=1..K-1
            if len(dt_array) > 0:
                time_array, dt_array_full = build_time_array_from_segments(dt_array, n_points)
                
                dbg(f"[DP-CLIMB] Time array: t_0=0, t_final={time_array[-1]:.3f}s")
                dbg(f"[DP-CLIMB] First 5 time points: {time_array[:5]}")
                dbg(f"[DP-CLIMB] Last 3 time points: {time_array[-3:]}")
                dbg(f"[DP-CLIMB] Last 3 time increments: {dt_array[-3:]}")
            else:
                # Fallback: uniform time grid
                dbg(f"[3D-DP] WARNING: Empty dt_array, using uniform time grid")
                time_array = np.linspace(0, n_points * 1.0, n_points)
                dt_array_full = np.ones(n_points)
                dt_array_full[0] = 0.0
            
            # ────────────────────────────────────────────────────────────────
            # Fuel Array Construction
            # ────────────────────────────────────────────────────────────────
            # Structure: Δm_fuel[0]=0, Δm_fuel[k]=dF_array[k-1] for k=1..K-1
            dF_array_full = np.zeros(n_points)
            if len(dF_array) > 0:
                n_segments = len(dF_array)
                n_expected_segments = n_points - 1
                
                if n_segments <= n_expected_segments:
                    # Fill segment fuel increments
                    dF_array_full[1:1+n_segments] = dF_array
                    # Pad if necessary
                    if n_segments < n_expected_segments:
                        last_dF = dF_array[-1] if n_segments > 0 else 0.0
                        dF_array_full[1+n_segments:] = last_dF
                else:
                    # Truncate excess segments
                    dF_array_full[1:] = dF_array[:n_expected_segments]
            
            # Cumulative fuel: m_fuel[k] = Σ_{i=0}^{k} Δm_fuel[i]
            fuel_array = np.cumsum(dF_array_full)
            
            # ────────────────────────────────────────────────────────────────
            # Mass Array Preservation
            # ────────────────────────────────────────────────────────────────
            # weight_array contains exact masses m[k] from DP optimization.
            # These values are preserved to maintain consistency: m[k] affects
            # drag D(M,h,m), which influences trajectory feasibility and cost.
            # Recalculating from fuel would introduce numerical discrepancies.
            
            # Verify array dimensions
            dbg(f"[3D-DP] Array dimensions: alt={len(alt_array)}, time={len(time_array)}, fuel={len(fuel_array)}, mass={len(weight_array)}")
            
            # ────────────────────────────────────────────────────────────────
            # Performance Gradients: ṁ = dm/dt, Ps = dh/dt
            # ────────────────────────────────────────────────────────────────
            if len(time_array) > 1 and len(alt_array) > 1:
                try:
                    mdot_kgps = np.gradient(fuel_array, time_array)  # ṁ [kg/s]
                    Ps_mps = np.gradient(alt_array, time_array)      # Ps [m/s]
                except ValueError as e:
                    dbg(f"[3D-DP] WARNING: Gradient computation failed: {e}")
                    mdot_kgps = np.zeros_like(alt_array)
                    Ps_mps = np.zeros_like(alt_array)
            else:
                mdot_kgps = np.zeros_like(alt_array)
                Ps_mps = np.zeros_like(alt_array)
            
            # ════════════════════════════════════════════════════════════════
            # Construct Output: MinFuelSchedule
            # ════════════════════════════════════════════════════════════════
            schedule = MinFuelSchedule(
                alt_m=alt_array,
                mach=mach_array,
                fuel_est_kg=float(path_costs[-1]),
                J_kg_per_m=np.array(path_costs),
                mdot_kgps=mdot_kgps,
                Ps_mps=Ps_mps,
                thrust_total_N=np.array([engine.thrust_with_lever(lever, mach, alt) * SystemConfiguration.N_ENGINES 
                                   for alt, mach, lever in zip(alt_array, mach_array, lever_array)]),
                D_N=np.array([aero.get_drag(mach, alt, weight) for alt, mach, weight in zip(alt_array, mach_array, weight_array)]),
                lever=lever_array,
                T_per_engine_N=np.array([engine.thrust_with_lever(lever, mach, alt) 
                                        for alt, mach, lever in zip(alt_array, mach_array, lever_array)]),
                mass_kg=weight_array,
                thrust_limited=np.isclose(lever_array, LEVER_MAX, atol=CLIMB_THRUST_LIMITED_ATOL),
                dt_s=dt_array_full,
                dFuel_kg=dF_array_full,
                cumFuel_kg=fuel_array
            )
            
            # Metadata dictionary
            info = {
                'total_fuel_kg': float(path_costs[-1]),
                'total_time_s': float(time_array[-1]),
                'final_mach': float(mach_array[-1]),
                'final_altitude': float(alt_array[-1]),
                'path_length': len(alt_array),
                'cost_matrix_3d': F,
                'predecessor_matrix': prv
            }
            
            return schedule, info
    
    @staticmethod
    def compute_cost(aero: PyAerodynamicsWrapper, engine: EngineWrapper, 
                       altitude: float, mach: float, lever: float,
                       mass_kg: float,
                       target_mach: float = None, prev_mach: float = None,
                       climb_fraction: float = None) -> float:
        """
        Evaluate augmented fuel cost density at state (h, M, δ).
        
        Formulation: J_aug = J + penalties
        where J = ṁ/Ps is fuel consumption per unit altitude.
        
        Physical relations:
            Ps = (T-D)V/m  [m/s] - specific excess power
            ṁ = TSFC · T  [kg/s] - fuel flow rate
            J = ṁ/Ps  [kg/m] - fuel cost density
        
        Penalties (optional):
            - Mach trajectory guidance: guides toward target M_final
            - Lever penalty: discourages high thrust settings
        
        Parameters:
            aero: PyAerodynamicsWrapper - drag model
            engine: EngineWrapper - thrust and TSFC model
            altitude: h [m] - altitude
            mach: M [-] - Mach number
            lever: δ [-] - throttle position ∈ [0,1]
            mass_kg: m [kg] - aircraft mass (required)
            target_mach: M_target [-] - terminal Mach (optional)
            prev_mach: M_prev [-] - previous Mach for smoothness (optional)
            climb_fraction: k/(K-1) [-] - climb progress ∈ [0,1] (step-based)
        
        Returns:
            J_aug [kg/m]: augmented fuel cost density, or ∞ if infeasible
        """
        try:
            
            # Kinematics: V = M · a(h)
            a = a_from_altitude(altitude)
            V = mach * a
            
            # Propulsion: T = T(δ, M, h)
            T_per = engine.thrust_with_lever(lever, mach, altitude)
            if T_per is None or not np.isfinite(T_per) or T_per <= 0:
                return np.inf
            T_tot = T_per * SystemConfiguration.N_ENGINES
            
            # Aerodynamics: D = D(M, h, m)
            D = aero.get_drag(mach, altitude, mass_kg)
            if not np.isfinite(D) or D < 0:
                return np.inf
            
            # Specific excess power: Ps = (T-D)V/m
            Ps = calculate_specific_excess_power(T_tot, D, mass_kg, V)
            if not np.isfinite(Ps) or Ps <= 0:
                return np.inf
            
            # Fuel flow: ṁ = TSFC · T
            engine.thrust_with_lever(lever, mach, altitude)  # Update TSFC state
            tsfc = engine.tsfc_current()
            if tsfc is None or not np.isfinite(tsfc) or tsfc < 0:
                return np.inf
            mdot = tsfc * T_per * SystemConfiguration.N_ENGINES
            
            # Base cost: J = ṁ/Ps [kg/m]
            J = mdot / Ps
            
            # Augmented cost: J_aug = J + penalties
            if target_mach is not None and ClimbingCore.PenaltySystem.MACH_TRAJECTORY_GUIDANCE:
                mach_penalty = ClimbingCore.PenaltySystem.compute_mach_penalty(mach, target_mach, prev_mach, climb_fraction)
                J += mach_penalty
            
            if ClimbingCore.PenaltySystem.LEVER_PENALTY_GUIDANCE:
                lever_penalty = ClimbingCore.PenaltySystem.compute_lever_penalty(lever, climb_fraction)
                J += lever_penalty
            
            return J
            
        except Exception:
            return np.inf
    
    # ────────────────────────────────────────────────────────────────────
    # Penalty Functions for Trajectory Guidance
    # ────────────────────────────────────────────────────────────────────
    
    class PenaltySystem:
        """
        Soft constraint penalties for trajectory optimization.
        
        Purpose: Guide DP optimizer toward physically realizable trajectories
        via augmented cost J_aug = J + Σ penalties.
        
        Penalty types:
            1. Mach trajectory guidance: ensures reachability of M_target
            2. Lever penalty: discourages sustained high thrust operation
        """
        
        # Configuration flags (from mission_config)
        MACH_TRAJECTORY_GUIDANCE = PENALTY_CLIMB_MACH_TRAJECTORY_GUIDANCE
        LEVER_PENALTY_GUIDANCE = PENALTY_CLIMB_LEVER_PENALTY_GUIDANCE
        TARGET_MACH_TOLERANCE = PENALTY_CLIMB_TARGET_MACH_TOLERANCE
        
        # Mach penalty coefficients
        MAX_REASONABLE_MACH_RATE = PENALTY_CLIMB_MAX_REASONABLE_MACH_RATE
        TOTAL_CLIMB_STEPS_ESTIMATE = PENALTY_CLIMB_TOTAL_STEPS_ESTIMATE
        URGENCY_MULTIPLIER = PENALTY_CLIMB_URGENCY_MULTIPLIER
        GUIDANCE_PENALTY_WEIGHT = PENALTY_CLIMB_GUIDANCE_PENALTY_WEIGHT
        
        # Lever penalty coefficients
        LEVER_PENALTY_WEIGHT = PENALTY_CLIMB_LEVER_PENALTY_WEIGHT
        LEVER_PENALTY_THRESHOLD = PENALTY_CLIMB_LEVER_PENALTY_THRESHOLD
        LEVER_PENALTY_EXPONENT = PENALTY_CLIMB_LEVER_PENALTY_EXPONENT
        LEVER_PENALTY_CRITICAL_THRESHOLD = PENALTY_CLIMB_LEVER_PENALTY_CRITICAL_THRESHOLD
        LEVER_PENALTY_CRITICAL_MULTIPLIER = PENALTY_CLIMB_LEVER_PENALTY_CRITICAL_MULTIPLIER
        LEVER_PENALTY_ULTRA_CRITICAL_THRESHOLD = PENALTY_CLIMB_LEVER_PENALTY_ULTRA_CRITICAL_THRESHOLD
        LEVER_PENALTY_ULTRA_CRITICAL_MULTIPLIER = PENALTY_CLIMB_LEVER_PENALTY_ULTRA_CRITICAL_MULTIPLIER
        
        @staticmethod
        def compute_mach_penalty(current_mach: float, target_mach: float, prev_mach: float = None, 
                                 climb_fraction: float = None) -> float:
            """
            Mach trajectory penalty via reachability corridor.
            
            Method: Define corridor [M_min(h), M_max(h)] such that M_target is reachable
            with physically reasonable Mach rates. Apply quadratic penalty outside corridor.
            
            Corridor bounds:
                ΔM_max = dM/dk_max · k_remaining
                M_min = M_target - ΔM_max
                M_max = M_target + ΔM_max
            
            Penalty structure:
                - M < M_min: P = urgency · (M_min - M)²
                - M > M_max: P = urgency · (M - M_max)²
                - M_min ≤ M ≤ M_max: Progressive guidance toward M_target
            
            Parameters:
                current_mach: M [-] - current Mach
                target_mach: M_target [-] - terminal Mach
                prev_mach: unused (kept for API compatibility)
                climb_fraction: k/(K-1) [-] - climb progress ∈ [0,1] (step-based)
            
            Returns:
                penalty [kg/m]: Mach deviation penalty
            """
            if climb_fraction is None:
                climb_fraction = 0.0
            
            # Remaining climb fraction: ξ_rem = 1 - progress
            remaining_fraction = 1.0 - climb_fraction
            estimated_steps_remaining = remaining_fraction * ClimbingCore.PenaltySystem.TOTAL_CLIMB_STEPS_ESTIMATE
            
            # Maximum reachable Mach deviation: ΔM_max = dM/dk_max · k_rem
            max_achievable_change = ClimbingCore.PenaltySystem.MAX_REASONABLE_MACH_RATE * estimated_steps_remaining
            
            # Reachability corridor: [M_min, M_max]
            min_reachable_mach = target_mach - max_achievable_change
            max_reachable_mach = target_mach + max_achievable_change
            
            # Urgency factor: increases as h → h_target
            urgency = (1.0 - remaining_fraction) * ClimbingCore.PenaltySystem.URGENCY_MULTIPLIER
            
            # Penalty computation based on corridor position
            if current_mach < min_reachable_mach:
                # Too low: risk undershooting M_target
                deviation = min_reachable_mach - current_mach
                penalty = urgency * (deviation ** 2)
                
            elif current_mach > max_reachable_mach:
                # Too high: risk overshooting M_target
                deviation = current_mach - max_reachable_mach  
                penalty = urgency * (deviation ** 2)
                
            else:
                # Within corridor: apply progressive guidance
                if climb_fraction > MACH_GUIDANCE_FINAL_PHASE_START:
                    # Final phase: strong convergence to M_target
                    final_phase_strength = (climb_fraction - MACH_GUIDANCE_FINAL_PHASE_START) / MACH_GUIDANCE_FINAL_PHASE_RANGE  # ξ ∈ [0,1]
                    mach_deviation = current_mach - target_mach
                    
                    # Terminal phase boost: extra convergence
                    if climb_fraction > MACH_GUIDANCE_TERMINAL_PHASE_START:
                        final_boost = ((climb_fraction - MACH_GUIDANCE_TERMINAL_PHASE_START) / MACH_GUIDANCE_TERMINAL_PHASE_RANGE) * MACH_GUIDANCE_TERMINAL_BOOST_MULTIPLIER
                        final_phase_strength *= (1.0 + final_boost)
                        
                    penalty = final_phase_strength * ClimbingCore.PenaltySystem.GUIDANCE_PENALTY_WEIGHT * (mach_deviation ** 2)
                else:
                    penalty = 0.0  # Early phase: no guidance penalty
            
            return penalty
        
        @staticmethod
        def compute_lever_penalty(current_lever: float, climb_fraction: float = None) -> float:
            """
            Lever penalty to discourage sustained high thrust operation.
            
            Physical basis: Engine thermal and mechanical limits are altitude-independent.
            Sustained high thrust causes increased wear, fuel consumption, and operational costs.
            
            Thrust regimes:
                δ ≤ 0.75: Maximum Continuous Thrust (MCT) - unlimited duration, no penalty
                0.75 < δ ≤ 0.90: Takeoff/climb thrust - time-limited, moderate penalty
                0.90 < δ ≤ 0.95: Maximum climb thrust - high wear, significant penalty
                δ > 0.95: Emergency thrust - severe penalty
            
            Penalty structure: P(δ) = w · [(δ - δ_MCT)^p + critical terms]
            where p = exponent, δ_MCT = MCT threshold
            
            Parameters:
                current_lever: δ [-] - throttle position ∈ [0,1]
                climb_fraction: unused (kept for API compatibility)
            
            Returns:
                penalty [kg/m]: lever position penalty (altitude-independent)
            """
            penalty = 0.0
            
            # Apply penalty only if δ > δ_MCT
            if current_lever > ClimbingCore.PenaltySystem.LEVER_PENALTY_THRESHOLD:
                # Excess thrust: Δδ = δ - δ_MCT
                excess_lever = current_lever - ClimbingCore.PenaltySystem.LEVER_PENALTY_THRESHOLD
                
                # Base penalty: P_base = (Δδ)^p
                lever_penalty = excess_lever ** ClimbingCore.PenaltySystem.LEVER_PENALTY_EXPONENT
                
                # Critical regime (δ > 0.90): additional quadratic term
                if current_lever > ClimbingCore.PenaltySystem.LEVER_PENALTY_CRITICAL_THRESHOLD:
                    critical_excess = current_lever - ClimbingCore.PenaltySystem.LEVER_PENALTY_CRITICAL_THRESHOLD
                    critical_penalty = critical_excess ** (ClimbingCore.PenaltySystem.LEVER_PENALTY_EXPONENT + 1.0)
                    lever_penalty += critical_penalty * ClimbingCore.PenaltySystem.LEVER_PENALTY_CRITICAL_MULTIPLIER
                
                # Ultra-critical regime (δ > 0.95): emergency thrust penalty
                if current_lever > ClimbingCore.PenaltySystem.LEVER_PENALTY_ULTRA_CRITICAL_THRESHOLD:
                    ultra_critical_excess = current_lever - ClimbingCore.PenaltySystem.LEVER_PENALTY_ULTRA_CRITICAL_THRESHOLD
                    ultra_critical_penalty = ultra_critical_excess ** (ClimbingCore.PenaltySystem.LEVER_PENALTY_EXPONENT + 2.0)
                    lever_penalty += ultra_critical_penalty * ClimbingCore.PenaltySystem.LEVER_PENALTY_ULTRA_CRITICAL_MULTIPLIER
                
                # Scale by constant weight (altitude-independent)
                penalty_weight = ClimbingCore.PenaltySystem.LEVER_PENALTY_WEIGHT
                penalty = penalty_weight * lever_penalty
            
            return penalty
    
    # ────────────────────────────────────────────────────────────────────
    # Performance Envelope Analysis
    # ────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def compute_full_envelope(aero: PyAerodynamicsWrapper, engine: EngineWrapper, mach_grid: np.ndarray, 
                             altitude_sched: np.ndarray, lever_grid: np.ndarray, mass_kg: float):
        """
        Compute 3D performance envelope J(M, h, δ) over feasible state space.
        
        Purpose: Map fuel cost density J = ṁ/Ps across entire flight envelope
        for visualization and feasibility analysis.
        
        Computation: Evaluate J at all grid points (M_i, h_k, δ_j) ∈ feasible region.
        Feasible region defined by: M ∈ [M_min, M_MMO], δ ∈ [0,1], Ps > 0.
        
        Parameters:
            aero: PyAerodynamicsWrapper - aerodynamic model
            engine: EngineWrapper - propulsion model
            mach_grid: np.ndarray - M_i, i=1..I (Mach discretization)
            altitude_sched: np.ndarray - h_k, k=1..K (altitude discretization)
            lever_grid: np.ndarray - δ_j, j=1..L (throttle lever discretization)
            mass_kg: float - reference mass [kg] for envelope (required)
            
        Returns:
            J_envelope: np.ndarray (I×K×L) - fuel cost density [kg/m]
        """
        
        print(f"[CLIMB] Computing performance envelope J(M,h,δ) at reference mass={mass_kg:.0f}kg")
        print(f"[CLIMB] Grid: {len(mach_grid)} Mach × {len(altitude_sched)} Alt × {len(lever_grid)} Lever")
        
        # Initialize envelope array: J[k,i,j] at (h_k, M_i, δ_j)
        K, I, L = len(altitude_sched), len(mach_grid), len(lever_grid)
        J_envelope = np.full((K, I, L), np.nan)
        
        # Evaluate J at all grid points
        feasible_count = 0
        total_points = K * I * L
        
        for k, h in enumerate(altitude_sched):
            for i, M in enumerate(mach_grid):
                for j, lever in enumerate(lever_grid):
                    # Feasibility check: M ∈ [M_min, M_MMO], δ ∈ [δ_min, δ_max]
                    if (M >= M_MIN_EFFECTIVE and M <= M_MMO and 
                        LEVER_MIN <= lever <= LEVER_MAX):
                        
                        # Compute J = ṁ/Ps with reference mass
                        cost = ClimbingCore.compute_cost(aero, engine, h, M, lever, mass_kg, climb_fraction=None)
                        
                        if np.isfinite(cost) and cost > 0:
                            J_envelope[k, i, j] = cost
                            feasible_count += 1
            
            # Progress reporting
            if k % 10 == 0 or k == K - 1:
                progress = (k + 1) / K * 100
                print(f"  Progress: {progress:.1f}% ({k+1}/{K} altitudes)")
        
        print(f"[CLIMB] Envelope complete: {feasible_count}/{total_points} feasible ({100*feasible_count/total_points:.1f}%)")
        
        # Transpose to (M, h, δ) ordering for visualization
        J_envelope_transposed = np.transpose(J_envelope, (1, 0, 2))
        
        return J_envelope_transposed
    
    # ────────────────────────────────────────────────────────────────────
    # Flight Envelope Validation
    # ────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def check_envelope_exceedance(strategy, aero):
        """
        Verify trajectory compliance with flight envelope constraints.
        
        Constraints:
            1. Maximum operating Mach: M ≤ M_MMO
            2. Stall boundary: M ≥ M_stall(h, m, CL_max)
        
        Method: Check each trajectory point against constraints using actual trajectory mass.
        Return first violation encountered or "Within Envelope".
        
        Parameters:
            strategy: StrategyRun or MinFuelSchedule - trajectory data with mass_kg array
            aero: PyAerodynamicsWrapper - for CL_max and stall calculations
            
        Returns:
            str: "Within Envelope", "Exceeds MMO", or "Exceeds CLmax"
        """
        exceeds_mmo = False
        exceeds_clmax = False
        
        # Get mass array if available, otherwise use initial mass
        mass_array = getattr(strategy, 'mass_kg', None)
        if mass_array is None or len(mass_array) == 0:
            # Fallback: use initial mass for all points
            mass_array = [INITIAL_MASS_KG] * len(strategy.mach)
        
        for i, (mach, alt, mass) in enumerate(zip(strategy.mach, strategy.alt_m, mass_array)):
            # Constraint 1: M ≤ M_MMO
            if mach > M_MMO:
                exceeds_mmo = True
                break
                
            # Constraint 2: M ≥ M_stall using actual trajectory mass
            try:
                M_stall = calculate_stall_mach(float(mass), float(alt), 
                                              aero.cl_max, S_REF_M2, safety_margin=CLIMB_STALL_SAFETY_MARGIN)
                if mach < M_stall:
                    exceeds_clmax = True
                    break
            except Exception:
                pass  # Skip if stall calculation unavailable
        
        if exceeds_mmo:
            return "Exceeds MMO"
        elif exceeds_clmax:
            return "Exceeds CLmax"
        else:
            return "Within Envelope"
    
# ========================================================================
# SECTION 4: SYSTEM UTILITIES
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

def create_climb_initial_state(start_mach: float = None,
                               start_lever: float = None,
                               mass_kg: float = None) -> ClimbInitialState:
    """
    Create ClimbInitialState from configuration or provided parameters.
    
    Parameters:
        start_mach: M_0 [-] - initial Mach (optional, computed from velocity if not provided)
        start_lever: δ_0 [-] - initial throttle (optional, uses START_LEVER_CLIMB)
        mass_kg: m_0 [kg] - initial mass (optional, uses INITIAL_MASS_KG)
    
    Returns:
        ClimbInitialState: initial state for climb optimization
    """
    # Compute Mach from velocity if not provided
    if start_mach is None:
        start_mach = mach_from_velocity(START_VELOCITY_CLIMB_MS, START_ALTITUDE_CLIMB_M)
    
    # Use defaults if not provided
    start_lever = start_lever if start_lever is not None else START_LEVER_CLIMB
    mass_kg = mass_kg if mass_kg is not None else INITIAL_MASS_KG
    
    return ClimbInitialState(
        altitude_m=START_ALTITUDE_CLIMB_M,
        mach=start_mach,
        mass_kg=mass_kg,
        lever=start_lever
    )

def run_optimization(aero: PyAerodynamicsWrapper, engine: EngineWrapper,
                    mach_grid: np.ndarray, altitude_sched: np.ndarray,
                    lever_samples: int = N_LEVER_SAMPLES_CLIMB,
                    target_mach: float = None,
                    target_mach_tolerance: float = PENALTY_CLIMB_TARGET_MACH_TOLERANCE,
                    start_mach: float = None,
                    start_lever: float = None,
                    mass_kg: float = None):
    """
    Primary interface for climb trajectory optimization.
    
    Algorithm: 3D dynamic programming over (h, M, δ) state space.
    Objective: Minimize ∫ J dh where J = ṁ/Ps.
    
    Provides consistent interface with descent.run_optimization() for
    unified mission analysis workflow.
    
    Parameters:
        aero: PyAerodynamicsWrapper - drag model
        engine: EngineWrapper - thrust model
        mach_grid: np.ndarray - M_i discretization
        altitude_sched: np.ndarray - h_k schedule
        lever_samples: int - number of δ levels
        target_mach: float - M_target at h_final (optional)
        target_mach_tolerance: float - |M_final - M_target| tolerance
        start_mach: float - M_0 initial Mach (optional)
        start_lever: float - δ_0 initial throttle (optional)
        mass_kg: float - m_0 initial mass (optional)
        
    Returns:
        MinFuelSchedule: optimal trajectory X*(k)
        dict: optimization metadata
    """
    # Create initial state from parameters (backward compatible)
    initial_state = create_climb_initial_state(start_mach, start_lever, mass_kg)
    
    return ClimbingCore.DynamicProgrammingOptimizer.solve_3d_dp(
        aero, engine, mach_grid, altitude_sched,
        initial_state=initial_state,
        lever_samples=lever_samples,
        target_mach=target_mach,
        target_mach_tolerance=target_mach_tolerance
    )

# ========================================================================
# SECTION 5: MODULE EXPORTS
# ========================================================================

# Data structure exports
StrategyRun = ClimbingCore.EnergyCalculator.StrategyRun
OptimalTrajectory = MinFuelSchedule  # Phase-agnostic alias for mission analysis